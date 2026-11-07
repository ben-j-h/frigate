"""Handle processing images for face detection and recognition."""

import base64
import datetime
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from frigate.comms.embeddings_updater import EmbeddingsRequestEnum
from frigate.comms.event_metadata_updater import (
    EventMetadataPublisher,
    EventMetadataTypeEnum,
)
from frigate.comms.inter_process import InterProcessRequestor
from frigate.config import FrigateConfig
from frigate.const import FACE_DIR, MODEL_CACHE_DIR
from frigate.data_processing.common.bench_dump import (
    bench_enabled,
    face_recognize_enabled,
    get_dump,
    setting,
)
from frigate.data_processing.common.face.model import (
    ArcFaceRecognizer,
    FaceNetRecognizer,
    FaceRecognizer,
)
from frigate.types import TrackedObjectUpdateTypesEnum
from frigate.util.builtin import EventsPerSecond, InferenceSpeed
from frigate.util.image import area
from frigate.util.path import safe_join, sanitize_path_component

from ..types import DataProcessorMetrics
from .api import RealTimeProcessorApi

logger = logging.getLogger(__name__)


MAX_DETECTION_HEIGHT = 1080
MAX_FACES_ATTEMPTS_AFTER_REC = 6
MAX_FACE_ATTEMPTS = 12


class FaceRealTimeProcessor(RealTimeProcessorApi):
    def __init__(
        self,
        config: FrigateConfig,
        requestor: InterProcessRequestor,
        sub_label_publisher: EventMetadataPublisher,
        metrics: DataProcessorMetrics,
    ):
        super().__init__(config, metrics)
        self.face_config = config.face_recognition
        self.requestor = requestor
        self.sub_label_publisher = sub_label_publisher
        self.face_detector: cv2.FaceDetectorYN | None = None
        self.requires_face_detection = "face" not in self.config.objects.all_objects
        self.person_face_history: dict[str, list[tuple[str, float, int]]] = {}
        self.camera_current_people: dict[str, list[str]] = {}
        # bench dump: latest box per person per camera, to list other people
        # whose box overlaps a face (camera -> id -> (frame_time, box))
        self.bench_people: dict[str, dict[str, tuple[float, list[int]]]] = {}
        # bench dump sampling state per person: [crops, last frame time, best area]
        self.bench_face_state: dict[str, list] = {}
        self.recognizer: FaceRecognizer
        self.faces_per_second = EventsPerSecond()
        self.inference_speed = InferenceSpeed(self.metrics.face_rec_speed)

        GITHUB_ENDPOINT = os.environ.get("GITHUB_ENDPOINT", "https://github.com")

        download_path = os.path.join(MODEL_CACHE_DIR, "facedet")
        self.model_files = {
            "facedet.onnx": f"{GITHUB_ENDPOINT}/NickM-27/facenet-onnx/releases/download/v1.0/facedet.onnx",
            "landmarkdet.yaml": f"{GITHUB_ENDPOINT}/NickM-27/facenet-onnx/releases/download/v1.0/landmarkdet.yaml",
        }

        if not all(
            os.path.exists(os.path.join(download_path, n))
            for n in self.model_files.keys()
        ):
            # conditionally import ModelDownloader
            from frigate.util.downloader import ModelDownloader

            self.downloader = ModelDownloader(
                model_name="facedet",
                download_path=download_path,
                file_names=list(self.model_files.keys()),
                download_func=self.__download_models,
                complete_func=self.__build_detector,
            )
            self.downloader.ensure_model_files()
        else:
            self.__build_detector()

        self.label_map: dict[int, str] = {}

        if self.face_config.model_size == "small":
            self.recognizer = FaceNetRecognizer(self.config)
        else:
            self.recognizer = ArcFaceRecognizer(self.config)

        self.recognizer.build()

    CONFIG_UPDATE_TOPIC = "config/face_recognition"

    def update_config(self, topic: str, payload: Any) -> None:
        """Update face recognition config at runtime."""
        if topic != self.CONFIG_UPDATE_TOPIC:
            return

        previous_min_area = self.config.face_recognition.min_area
        self.config.face_recognition = payload
        self.face_config = payload

        for camera_config in self.config.cameras.values():
            if camera_config.face_recognition.min_area == previous_min_area:
                camera_config.face_recognition.min_area = payload.min_area

        logger.debug("Face recognition config updated dynamically")

    def __download_models(self, path: str) -> None:
        try:
            file_name = os.path.basename(path)
            # conditionally import ModelDownloader
            from frigate.util.downloader import ModelDownloader

            ModelDownloader.download_from_url(self.model_files[file_name], path)
        except Exception as e:
            logger.error(f"Failed to download {path}: {e}")

    def __build_detector(self) -> None:
        self.face_detector = cv2.FaceDetectorYN.create(
            os.path.join(MODEL_CACHE_DIR, "facedet/facedet.onnx"),
            config="",
            input_size=(320, 320),
            score_threshold=0.5,
            nms_threshold=0.3,
        )
        self.faces_per_second.start()

    def __detect_face(
        self, input: np.ndarray, threshold: float
    ) -> tuple[int, int, int, int] | None:
        """Detect faces in input image."""
        if not self.face_detector:
            return None

        # YN face detector fails at extreme definitions
        # this rescales to a size that can properly detect faces
        # still retaining plenty of detail
        if input.shape[0] > MAX_DETECTION_HEIGHT:
            scale_factor = MAX_DETECTION_HEIGHT / input.shape[0]
            new_width = int(scale_factor * input.shape[1])
            input = cv2.resize(input, (new_width, MAX_DETECTION_HEIGHT))
        else:
            scale_factor = 1

        self.face_detector.setInputSize((input.shape[1], input.shape[0]))
        faces = self.face_detector.detect(input)

        if faces is None or faces[1] is None:
            return None  # type: ignore[unreachable]

        face = None

        for _, potential_face in enumerate(faces[1]):
            if potential_face[-1] < threshold:
                continue

            raw_bbox = potential_face[0:4].astype(np.uint16)
            x: int = int(max(raw_bbox[0], 0) / scale_factor)
            y: int = int(max(raw_bbox[1], 0) / scale_factor)
            w: int = int(raw_bbox[2] / scale_factor)
            h: int = int(raw_bbox[3] / scale_factor)
            bbox = (x, y, x + w, y + h)

            if face is None or area(bbox) > area(face):  # type: ignore[unreachable]
                face = bbox

        return face

    def __update_metrics(self, duration: float) -> None:
        self.faces_per_second.update()
        self.inference_speed.update(duration)

    def process_frame(self, obj_data: dict[str, Any], frame: np.ndarray) -> None:
        """Look for faces in image."""
        self.metrics.face_rec_fps.value = self.faces_per_second.eps()
        camera = obj_data["camera"]

        if not self.config.cameras[camera].face_recognition.enabled:
            logger.debug(f"Face recognition disabled for camera {camera}, skipping")
            return

        start = datetime.datetime.now().timestamp()
        id = obj_data["id"]

        # don't run for non person objects
        if obj_data.get("label") != "person":
            logger.debug("Not processing face for a non person object.")
            return

        if bench_enabled() and obj_data.get("box"):
            self.bench_people.setdefault(camera, {})[id] = (
                obj_data.get("frame_time", start),
                list(obj_data["box"]),
            )

        # Frigate's recognizer keeps its stock limits; past them (or with the
        # recognizer off) the bench dump keeps sampling the whole track
        stock_ok = face_recognize_enabled() and self._stock_attempt_ok(id, obj_data)
        frame_time = obj_data.get("frame_time", start)

        if not stock_ok and not (
            bench_enabled() and self._bench_face_due(id, frame_time, None)
        ):
            return

        face: dict[str, Any] | None = None

        if self.requires_face_detection:
            logger.debug("Running manual face detection.")
            person_box = obj_data.get("box")

            if not person_box:
                logger.debug(f"No person box available for {id}")
                return

            # YuNet (cv2.FaceDetectorYN) is trained on BGR
            bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
            left, top, right, bottom = person_box
            person = bgr[top:bottom, left:right]
            face_box = self.__detect_face(person, self.face_config.detection_threshold)

            if not face_box:
                logger.debug("Detected no faces for person object.")
                return

            face_frame = person[
                max(0, face_box[1]) : min(frame.shape[0], face_box[3]),
                max(0, face_box[0]) : min(frame.shape[1], face_box[2]),
            ]
            frame_face_box = [
                face_box[0] + left,
                face_box[1] + top,
                face_box[2] + left,
                face_box[3] + top,
            ]
            face_score = None

            # check that face is correct size
            if area(face_box) < self.config.cameras[camera].face_recognition.min_area:
                logger.debug(
                    f"Detected face that is smaller than the min_area {face} < {self.config.cameras[camera].face_recognition.min_area}"
                )
                return

        else:
            # don't run for object without attributes
            if not obj_data.get("current_attributes"):
                logger.debug("No attributes to parse.")
                return

            attributes: list[dict[str, Any]] = obj_data.get("current_attributes", [])
            for attr in attributes:
                if attr.get("label") != "face":
                    continue

                if face is None or attr.get("score", 0.0) > face.get("score", 0.0):
                    face = attr

            # no faces detected in this frame
            if not face:
                logger.debug(f"No face attributes found for {id}")
                return

            face_box = face.get("box")

            # check that face is valid
            if (
                not face_box
                or area(face_box)
                < self.config.cameras[camera].face_recognition.min_area
            ):
                logger.debug(f"Invalid face box {face}")
                return

            if not stock_ok and not self._bench_face_due(
                id, frame_time, area(face_box)
            ):
                return

            face_frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
            frame_face_box = list(face_box)
            face_score = face.get("score")

            face_frame = face_frame[
                max(0, face_box[1]) : min(frame.shape[0], face_box[3]),
                max(0, face_box[0]) : min(frame.shape[1], face_box[2]),
            ]

        if face_frame.size == 0:
            logger.debug(f"Empty face crop for {id}")
            return

        res = self.recognizer.classify(face_frame) if stock_ok else None
        face_area = face_frame.shape[0] * face_frame.shape[1]

        if bench_enabled() and (
            stock_ok or self._bench_face_due(id, frame_time, face_area)
        ):
            self._bench_face(
                camera, id, obj_data, start, face_frame, frame_face_box, face_score, res
            )
            st = self.bench_face_state.setdefault(id, [0, 0.0, 0])
            st[0] += 1
            st[1] = frame_time
            st[2] = max(st[2], face_area)

        if not stock_ok:
            self.__update_metrics(datetime.datetime.now().timestamp() - start)
            return

        if not res:
            logger.debug(f"Face recognizer returned no result for {id}")
            self.__update_metrics(datetime.datetime.now().timestamp() - start)
            return

        sub_label, score = res

        if score <= self.face_config.unknown_score:
            sub_label = "unknown"

        logger.debug(
            f"Detected best face for person as: {sub_label} with probability {score}"
        )

        self.write_face_attempt(
            face_frame, id, datetime.datetime.now().timestamp(), sub_label, score
        )

        if id not in self.person_face_history:
            self.person_face_history[id] = []

            if camera not in self.camera_current_people:
                self.camera_current_people[camera] = []

            self.camera_current_people[camera].append(id)

        self.person_face_history[id].append(
            (sub_label, score, face_frame.shape[0] * face_frame.shape[1])
        )
        (weighted_sub_label, weighted_score) = self.weighted_average(
            self.person_face_history[id]
        )

        self.requestor.send_data(
            "tracked_object_update",
            json.dumps(
                {
                    "type": TrackedObjectUpdateTypesEnum.face,
                    "name": weighted_sub_label,
                    "score": weighted_score,
                    "id": id,
                    "camera": camera,
                    "timestamp": start,
                }
            ),
        )

        if weighted_score >= self.face_config.recognition_threshold:
            self.sub_label_publisher.publish(
                (id, weighted_sub_label, weighted_score),
                EventMetadataTypeEnum.sub_label.value,
            )

        self.__update_metrics(datetime.datetime.now().timestamp() - start)

    def handle_request(
        self, topic: str, request_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        if topic == EmbeddingsRequestEnum.clear_face_classifier.value:
            self.recognizer.clear()
            return {"success": True, "message": "Face classifier cleared."}
        elif topic == EmbeddingsRequestEnum.recognize_face.value:
            img = cv2.imdecode(
                np.frombuffer(base64.b64decode(request_data["image"]), dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )

            # detect faces with lower confidence since we expect the face
            # to be visible in uploaded images
            face_box = self.__detect_face(img, 0.5)

            if not face_box:
                return {"message": "No face was detected.", "success": False}

            face = img[face_box[1] : face_box[3], face_box[0] : face_box[2]]
            res = self.recognizer.classify(face)

            if not res:
                return {"success": False, "message": "No face was recognized."}

            sub_label, score = res

            if score <= self.face_config.unknown_score:
                sub_label = "unknown"

            return {"success": True, "score": score, "face_name": sub_label}
        elif topic == EmbeddingsRequestEnum.register_face.value:
            label = request_data["face_name"]

            if request_data.get("cropped"):
                thumbnail = request_data["image"]
            else:
                img = cv2.imdecode(
                    np.frombuffer(
                        base64.b64decode(request_data["image"]), dtype=np.uint8
                    ),
                    cv2.IMREAD_COLOR,
                )

                # detect faces with lower confidence since we expect the face
                # to be visible in uploaded images
                face_box = self.__detect_face(img, 0.5)

                if not face_box:
                    return {
                        "message": "No face was detected.",
                        "success": False,
                    }

                face = img[face_box[1] : face_box[3], face_box[0] : face_box[2]]
                _, thumbnail = cv2.imencode(
                    ".webp", face, [int(cv2.IMWRITE_WEBP_QUALITY), 100]
                )

            # write face to library
            sanitized_label = sanitize_path_component(label)
            folder = safe_join(FACE_DIR, label)

            if sanitized_label is None or folder is None:
                return {
                    "message": f"Invalid face name: {label}",
                    "success": False,
                }

            file = os.path.join(
                folder, f"{sanitized_label}_{datetime.datetime.now().timestamp()}.webp"
            )
            os.makedirs(folder, exist_ok=True)

            # save face image
            with open(file, "wb") as output:
                output.write(thumbnail.tobytes())

            self.recognizer.clear()
            return {
                "message": "Successfully registered face.",
                "success": True,
            }
        elif topic == EmbeddingsRequestEnum.reprocess_face.value:
            current_file: str = request_data["image_file"]
            (id_time, id_rand, timestamp, _, _) = current_file.split("-")
            img = None
            id = f"{id_time}-{id_rand}"

            if current_file:
                img = cv2.imread(current_file)

            if img is None:
                return {  # type: ignore[unreachable]
                    "message": "Invalid image file.",
                    "success": False,
                }

            res = self.recognizer.classify(img)

            if not res:
                return {
                    "message": "Model is still training, please try again in a few moments.",
                    "success": False,
                }

            sub_label, score = res

            if score <= self.face_config.unknown_score:
                sub_label = "unknown"

            if "-" in sub_label:
                sub_label = sub_label.replace("-", "_")

            if self.config.face_recognition.save_attempts:
                # write face to library
                folder = os.path.join(FACE_DIR, "train")
                os.makedirs(folder, exist_ok=True)
                new_file = os.path.join(
                    folder, f"{id}-{timestamp}-{sub_label}-{score}.webp"
                )
                shutil.move(current_file, new_file)

            return {
                "message": f"Successfully reprocessed face. Result: {sub_label} (score: {score:.2f})",
                "success": True,
                "face_name": sub_label,
                "score": score,
            }

        return None

    def _stock_attempt_ok(self, id: str, obj_data: dict[str, Any]) -> bool:
        """Frigate's own per-person attempt limits (unchanged)."""
        # don't overwrite sub label for objects that have a sub label
        # that is not a face
        if obj_data.get("sub_label") and id not in self.person_face_history:
            logger.debug(
                f"Not processing face due to existing sub label: {obj_data.get('sub_label')}."
            )
            return False

        # check if we have hit limits
        if (
            id in self.person_face_history
            and len(self.person_face_history[id]) >= MAX_FACES_ATTEMPTS_AFTER_REC
        ):
            # if we are at max attempts after rec and we have a rec
            if obj_data.get("sub_label"):
                logger.debug(
                    "Not processing due to hitting max attempts after true recognition."
                )
                return False

            # if we don't have a rec and are at max attempts
            if len(self.person_face_history[id]) >= MAX_FACE_ATTEMPTS:
                logger.debug("Not processing due to hitting max rec attempts.")
                return False

        return True

    def _bench_face_due(
        self, id: str, frame_time: float, face_area: int | None
    ) -> bool:
        """Bench dump sampling over the whole track: at most one crop per
        FRIGATE_FACE_DUMP_GAP seconds (default 0.5); after FRIGATE_FACE_DUMP_MAX
        crops (default 150) only a face 25% larger than the best so far."""
        n, last, best = self.bench_face_state.get(id, (0, 0.0, 0))
        if frame_time - last < float(setting("FRIGATE_FACE_DUMP_GAP", "0.5")):
            return False
        if n >= int(setting("FRIGATE_FACE_DUMP_MAX", "150")):
            # face size unknown yet (pre-check): decide once it is
            return face_area is None or face_area > 1.25 * best
        return True

    def _bench_face(
        self,
        camera: str,
        id: str,
        obj_data: dict[str, Any],
        start: float,
        face_frame: np.ndarray,
        face_box: list[int],
        face_score: float | None,
        res: tuple[str, float] | None,
    ) -> None:
        """Fork-only evaluation dump of the exact face crop (see bench_dump),
        with the geometry needed to check the face belongs to this person."""
        try:
            frame_time = obj_data.get("frame_time", start)
            others: list[dict[str, Any]] = []
            for other_id, (t, box) in self.bench_people.get(camera, {}).items():
                if other_id == id or abs(frame_time - t) > 1.0:
                    continue
                if (
                    box[0] < face_box[2]
                    and face_box[0] < box[2]
                    and box[1] < face_box[3]
                    and face_box[1] < box[3]
                ):
                    others.append(
                        {"id": other_id, "box": box, "dt": round(t - frame_time, 3)}
                    )
            get_dump(self.config, "faces", "person").save(
                camera,
                id,
                int(frame_time * 1000),
                face_frame,
                {
                    "t": round(float(frame_time), 3),
                    "face_box": [int(x) for x in face_box],
                    "face_score": round(float(face_score), 3)
                    if face_score is not None
                    else None,
                    "person_box": [int(x) for x in obj_data.get("box") or []],
                    "person_score": obj_data.get("score"),
                    "sub_label": obj_data.get("sub_label"),
                    "others": others,
                    "frigate": [res[0], round(float(res[1]), 3)] if res else None,
                    "attempt": self.bench_face_state.get(id, [0])[0] + 1,
                },
            )
        except Exception as e:  # the dump never stops face processing
            logger.debug(f"{camera}: face bench dump failed: {e}")

    def expire_object(self, object_id: str, camera: str) -> None:
        self.bench_people.get(camera, {}).pop(object_id, None)
        self.bench_face_state.pop(object_id, None)
        if object_id in self.person_face_history:
            self.person_face_history.pop(object_id)

            if object_id in self.camera_current_people.get(camera, []):
                self.camera_current_people[camera].remove(object_id)

    def weighted_average(
        self, results_list: list[tuple[str, float, int]], max_weight: int = 4000
    ) -> tuple[str | None, float]:
        """
        Calculates a robust weighted average, capping the area weight and giving more weight to higher scores.

        Args:
            results_list: A list of tuples, where each tuple contains (name, score, face_area).
            max_weight: The maximum weight to apply based on face area.

        Returns:
            A tuple containing the prominent name and its weighted average score, or (None, 0.0) if the list is empty.
        """
        if not results_list:
            return None, 0.0

        counts: dict[str, int] = {}
        weighted_scores: dict[str, float] = {}
        total_weights: dict[str, float] = {}

        for name, score, face_area in results_list:
            if name == "unknown":
                continue

            if name not in weighted_scores:
                counts[name] = 0
                weighted_scores[name] = 0.0
                total_weights[name] = 0.0

            # increase count
            counts[name] += 1

            # Capped weight based on face area
            weight: float = min(face_area, max_weight)

            # Score-based weighting (higher scores get more weight)
            weight *= (score - self.face_config.unknown_score) * 10
            weighted_scores[name] += score * weight
            total_weights[name] += weight

        if not weighted_scores:
            return None, 0.0

        best_name = max(weighted_scores, key=lambda k: weighted_scores[k])

        # If the number of faces for this person < min_faces, we are not confident it is a correct result
        if counts[best_name] < self.face_config.min_faces:
            return None, 0.0

        # If the best name has the same number of results as another name, we are not confident it is a correct result
        for name, count in counts.items():
            if name != best_name and counts[best_name] == count:
                return None, 0.0

        weighted_average = weighted_scores[best_name] / total_weights[best_name]

        return best_name, weighted_average

    def write_face_attempt(
        self,
        frame: np.ndarray,
        event_id: str,
        timestamp: float,
        sub_label: str,
        score: float,
    ) -> None:
        if self.config.face_recognition.save_attempts:
            # write face to library
            folder = os.path.join(FACE_DIR, "train")

            if "-" in sub_label:
                sub_label = sub_label.replace("-", "_")

            file = os.path.join(
                folder, f"{event_id}-{timestamp}-{sub_label}-{score}.webp"
            )
            os.makedirs(folder, exist_ok=True)
            cv2.imwrite(file, frame)

            files = sorted(
                filter(lambda f: f.endswith(".webp"), os.listdir(folder)),
                key=lambda f: os.path.getctime(os.path.join(folder, f)),
                reverse=True,
            )

            # delete oldest face image if maximum is reached
            if len(files) > self.config.face_recognition.save_attempts:
                Path(os.path.join(folder, files[-1])).unlink(missing_ok=True)
