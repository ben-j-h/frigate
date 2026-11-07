"""Evaluation dump for the ben-j-h fork (not part of any upstream proposal).

Saves the exact crops given to the face and plate recognition models, one JSON
line per pass, and announces each record on MQTT under a topic root outside
Frigate's topic_prefix (so Home Assistant, which only listens to frigate/...,
never sees it). Crops are kept as long as the camera keeps snapshots for the
object (snapshots.retain), checked hourly.

Switched by KEY=VALUE lines in /config/bench_dump.env (re-read every minute;
an environment variable of the same name wins, but Frigate's worker processes
start from a stripped environment, so the file is what works). Stock Frigate
ignores the file, so removing the patched files never leaves a config it
rejects:
  FRIGATE_BENCH_DUMP=1              enable the dump (faces and plates)
  FRIGATE_BENCH_MQTT_TOPIC=<root>   MQTT topic root (default frigate_bench,
                                    empty = no MQTT); records go to <root>/<kind>
  FRIGATE_FACE_RECOGNIZE=0          skip Frigate's own face recognizer
                                    (crops are still dumped)

Files: clips/bench/<kind>/<camera>/<event id>/<frame id>.png and
clips/bench/<kind>/reads-YYYY-MM-DD.jsonl (UTC date).
"""

import datetime
import json
import logging
import os
import shutil
import threading
import time
from typing import Any

import cv2
import numpy as np

from frigate.config import FrigateConfig
from frigate.const import CLIPS_DIR

logger = logging.getLogger(__name__)

BENCH_DIR = os.path.join(CLIPS_DIR, "bench")
PRUNE_INTERVAL = 3600


CONF_FILE = "/config/bench_dump.env"
CONF_TTL = 60
_conf: dict[str, str] = {}
_conf_read = 0.0


def setting(key: str, default: str) -> str:
    """Environment variable, else /config/bench_dump.env (cached CONF_TTL s)."""
    global _conf, _conf_read
    if key in os.environ:
        return os.environ[key]
    if time.time() - _conf_read > CONF_TTL:
        conf: dict[str, str] = {}
        try:
            with open(CONF_FILE) as f:
                for line in f:
                    line = line.split("#", 1)[0].strip()
                    k, sep, v = line.partition("=")
                    if sep:
                        conf[k.strip()] = v.strip().strip('"')
        except OSError:
            pass
        _conf, _conf_read = conf, time.time()
    return _conf.get(key, default)


def yuv_box_to_bgr(frame: np.ndarray, box: Any) -> np.ndarray | None:
    """BGR crop of box [x1, y1, x2, y2] from an I420 frame, converting only that
    region (x snapped to even, height to a multiple of 4, as I420 needs)."""
    H, W = frame.shape[0] * 2 // 3, frame.shape[1]
    x1, y1 = max(0, int(box[0])) // 2 * 2, max(0, int(box[1])) // 2 * 2
    x2, y2 = min(W, int(box[2])) // 2 * 2, min(H, int(box[3]))
    y2 = y1 + (y2 - y1) // 4 * 4
    if x2 - x1 < 2 or y2 - y1 < 4:
        return None
    u = frame[H : H + H // 4].reshape(H // 2, W // 2)
    v = frame[H + H // 4 : H + H // 2].reshape(H // 2, W // 2)
    h, w = y2 - y1, x2 - x1
    buf = np.concatenate(
        [
            frame[y1:y2, x1:x2].ravel(),
            u[y1 // 2 : y2 // 2, x1 // 2 : x2 // 2].ravel(),
            v[y1 // 2 : y2 // 2, x1 // 2 : x2 // 2].ravel(),
        ]
    ).reshape(h * 3 // 2, w)
    return cv2.cvtColor(buf, cv2.COLOR_YUV2BGR_I420)


def bench_enabled() -> bool:
    return setting("FRIGATE_BENCH_DUMP", "0") == "1"


def face_recognize_enabled() -> bool:
    return setting("FRIGATE_FACE_RECOGNIZE", "1") != "0"


_dumps: dict[str, "BenchDump"] = {}


def get_dump(config: FrigateConfig, kind: str, label: str) -> "BenchDump":
    """One dump (and one MQTT client) per kind per process: the real-time and
    post processors share it, so client ids never collide."""
    if kind not in _dumps:
        _dumps[kind] = BenchDump(config, kind, label)
    return _dumps[kind]


class BenchDump:
    def __init__(self, config: FrigateConfig, kind: str, label: str) -> None:
        self.config = config
        self.kind = kind
        self.label = label  # object whose snapshot retention applies
        self.root = os.path.join(BENCH_DIR, kind)
        self.topic_root = setting("FRIGATE_BENCH_MQTT_TOPIC", "frigate_bench")
        self.client: Any = None
        self.last_prune = 0.0
        self.lock = threading.Lock()

        if self.topic_root and config.mqtt.enabled:
            self._start_mqtt()

    def _start_mqtt(self) -> None:
        try:
            import paho.mqtt.client as mqtt
            from paho.mqtt.enums import CallbackAPIVersion

            m = self.config.mqtt
            client = mqtt.Client(
                callback_api_version=CallbackAPIVersion.VERSION2,
                client_id=f"{m.client_id}-bench-{self.kind}-{os.getpid()}",
            )
            if m.user is not None:
                client.username_pw_set(m.user, password=m.password)
            if m.tls_ca_certs is not None:
                client.tls_set(m.tls_ca_certs, m.tls_client_cert, m.tls_client_key)
                if m.tls_insecure is not None:
                    client.tls_insecure_set(m.tls_insecure)
            client.connect_async(m.host, m.port, 60)
            client.loop_start()
            self.client = client
        except Exception as e:
            logger.warning(f"Bench dump MQTT unavailable: {e}")
            self.client = None

    def _retain_days(self, camera: str) -> float:
        cam = self.config.cameras.get(camera)
        retain = cam.snapshots.retain if cam else self.config.snapshots.retain
        return retain.objects.get(self.label, retain.default)

    def save(
        self,
        camera: str,
        event_id: str,
        frame_id: int,
        image: np.ndarray,
        record: dict[str, Any],
        ext: str = "png",
    ) -> None:
        """Write the crop and its JSON line, publish the record, prune hourly."""
        try:
            rel = f"bench/{self.kind}/{camera}/{event_id}/{frame_id}.{ext}"
            os.makedirs(
                os.path.join(BENCH_DIR, self.kind, camera, event_id), exist_ok=True
            )
            params = [cv2.IMWRITE_JPEG_QUALITY, 92] if ext == "jpg" else []
            cv2.imwrite(os.path.join(CLIPS_DIR, rel), image, params)
            record = {
                "kind": self.kind,
                "camera": camera,
                "id": event_id,
                "frame_id": frame_id,
                "file": rel,
                **record,
            }
            line = json.dumps(record)
            day = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
            with self.lock:
                with open(os.path.join(self.root, f"reads-{day}.jsonl"), "a") as f:
                    f.write(line + "\n")
            if self.client is not None:
                self.client.publish(f"{self.topic_root}/{self.kind}", line, qos=0)
        except Exception as e:
            logger.debug(f"{camera}: bench dump ({self.kind}) failed: {e}")

        if time.time() - self.last_prune > PRUNE_INTERVAL:
            self.last_prune = time.time()
            threading.Thread(target=self._prune, daemon=True).start()

    def _prune(self) -> None:
        """Delete event folders older than the camera's snapshot retention, and
        JSON line files older than the longest retention."""
        now = time.time()
        longest = 0.0
        try:
            for camera in os.listdir(self.root):
                cam_dir = os.path.join(self.root, camera)
                if not os.path.isdir(cam_dir):
                    continue
                days = self._retain_days(camera)
                longest = max(longest, days)
                for event_id in os.listdir(cam_dir):
                    try:
                        start = float(event_id.split("-")[0])
                    except ValueError:
                        start = os.path.getmtime(os.path.join(cam_dir, event_id))
                    if now - start > days * 86400:
                        shutil.rmtree(
                            os.path.join(cam_dir, event_id), ignore_errors=True
                        )
            longest = longest or self.config.snapshots.retain.default
            for name in os.listdir(self.root):
                if name.startswith("reads-") and name.endswith(".jsonl"):
                    path = os.path.join(self.root, name)
                    if now - os.path.getmtime(path) > (longest + 1) * 86400:
                        os.remove(path)
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.debug(f"bench dump ({self.kind}) prune failed: {e}")
