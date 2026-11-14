import os
import time
import numpy as np
from typing import List, Optional

from backend.config.ai_settings import ai_settings
from backend.core.exceptions import DetectorInferenceException, DetectorUnavailableException
from backend.core.logging import logger
from backend.schemas.detection import DetectionItem, DetectionResultContainer, NormalizedBoundingBox, TargetCategory, DetectorSource
from backend.services.ai.model_manager import model_manager


class DetectionService:
    """Runs only detectors with semantics suitable for SafeFrame targets.

    The bundled COCO YOLO model is intentionally not used: persons, vehicles,
    books, and phones are not face, plate, document, or screen detectors.
    YuNet is the only currently configured target detector.
    """
    supported_categories = {TargetCategory.FACES}

    def __init__(self) -> None:
        self.model_manager = model_manager
        self.conf_threshold = ai_settings.DETECTION_CONF_THRESHOLD

    def _requested_categories(self, target_categories: List[str]) -> set[TargetCategory]:
        try:
            requested = {TargetCategory(value) for value in target_categories}
        except ValueError as exc:
            raise DetectorUnavailableException(f"unknown target category: {exc}") from exc
        unsupported = requested - self.supported_categories
        if unsupported:
            names = ", ".join(sorted(item.value for item in unsupported))
            raise DetectorUnavailableException(f"no validated detector is configured for: {names}")
        return requested

    def validate_available(self, target_categories: List[str]) -> None:
        """Preflight used before a job is accepted."""
        requested = self._requested_categories(target_categories)
        if requested:
            handle = self.model_manager.get_model("secondary_face")
            if handle.get("model") is None or handle.get("status") != "LOADED":
                # Keep the resolved filesystem path out of public responses; the
                # full detail is logged by ModelManager.
                if handle.get("status") == "ERROR":
                    detail = "YuNet face detector failed to initialize; verify the model weights are valid."
                else:
                    weights_name = os.path.basename(handle.get("weights_path") or ai_settings.YUNET_MODEL_PATH)
                    detail = f"YuNet face detector weights '{weights_name}' are not available on the server."
                raise DetectorUnavailableException(detail)

    def detect_entities(self, frame: Optional[np.ndarray], target_categories: List[str], frame_index: int = 0,
                        custom_conf_threshold: Optional[float] = None) -> DetectionResultContainer:
        start = time.time()
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise DetectorInferenceException(f"frame {frame_index} is empty or unreadable")
        requested = self._requested_categories(target_categories)
        if not requested:
            return DetectionResultContainer(frame_index=frame_index, total_detections=0, detections=[],
                                            inference_time_ms=0.0, categories_detected=[], secondary_detector_used=False)

        self.validate_available(target_categories)
        handle = self.model_manager.get_model("secondary_face")
        detector = handle["model"]

        height, width = frame.shape[:2]
        try:
            detector.setInputSize((width, height))
            _, faces = detector.detect(frame)
        except Exception as exc:
            logger.error("YuNet inference failed on frame %s: %s", frame_index, exc)
            raise DetectorInferenceException(f"YuNet failed on frame {frame_index}: {exc}") from exc

        detections: List[DetectionItem] = []
        # faces=None is a genuine empty inference result. It must never cause
        # a fabricated detection or a false success signal.
        if faces is not None:
            for face in faces:
                fx, fy, fw, fh = (float(value) for value in face[:4])
                confidence = float(face[-1])
                if confidence < (custom_conf_threshold or self.conf_threshold):
                    continue
                x = max(0.0, min(100.0, fx / width * 100.0))
                y = max(0.0, min(100.0, fy / height * 100.0))
                detections.append(DetectionItem(
                    category=TargetCategory.FACES, label="FACE", confidence=round(confidence, 2),
                    bbox=NormalizedBoundingBox(x=round(x, 2), y=round(y, 2),
                        width=round(max(0.1, min(100.0 - x, fw / width * 100.0)), 2),
                        height=round(max(0.1, min(100.0 - y, fh / height * 100.0)), 2)),
                    source=DetectorSource.SECONDARY_FACE,
                ))
        elapsed = round((time.time() - start) * 1000, 2)
        return DetectionResultContainer(
            frame_index=frame_index, total_detections=len(detections), detections=detections,
            inference_time_ms=elapsed,
            categories_detected=[TargetCategory.FACES] if detections else [], secondary_detector_used=True,
        )


detection_service = DetectionService()
