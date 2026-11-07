from frigate.comms.inter_process import InterProcessRequestor
from frigate.config.classification import LPRRecognitionModelEnum
from frigate.embeddings.onnx.lpr_embedding import (
    FastPlateOCRRecognition,
    LicensePlateDetector,
    PaddleOCRClassification,
    PaddleOCRDetection,
    PaddleOCRRecognition,
)

from ...types import DataProcessorModelRunner


class LicensePlateModelRunner(DataProcessorModelRunner):
    def __init__(
        self,
        requestor: InterProcessRequestor,
        device: str = "CPU",
        model_size: str = "small",
        recognition_model: str = LPRRecognitionModelEnum.paddleocr,
    ):
        super().__init__(requestor, device, model_size)
        self.recognition_model_type = LPRRecognitionModelEnum(recognition_model)
        self.detection_model: PaddleOCRDetection | None = None
        self.classification_model: PaddleOCRClassification | None = None
        self.recognition_model: PaddleOCRRecognition | None = None
        self.plate_ocr_model: FastPlateOCRRecognition | None = None

        if self.recognition_model_type == LPRRecognitionModelEnum.fast_plate_ocr:
            # reads the whole plate crop, so PaddleOCR's text line models are not needed
            self.plate_ocr_model = FastPlateOCRRecognition(
                requestor=requestor, device=device
            )
        else:
            self.detection_model = PaddleOCRDetection(
                model_size=model_size, requestor=requestor, device=device
            )
            self.classification_model = PaddleOCRClassification(
                model_size=model_size, requestor=requestor, device=device
            )
            self.recognition_model = PaddleOCRRecognition(
                model_size=model_size, requestor=requestor, device=device
            )
        self.yolov9_detection_model = LicensePlateDetector(
            model_size=model_size, requestor=requestor, device=device
        )

        # Load all models once
        for model in (
            self.detection_model,
            self.classification_model,
            self.recognition_model,
            self.plate_ocr_model,
            self.yolov9_detection_model,
        ):
            if model is not None:
                model._load_model_and_utils()
