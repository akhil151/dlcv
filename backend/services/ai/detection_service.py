import os
import time
import numpy as np
from typing import List, Optional, Sequence

import cv2

from backend.config.ai_settings import ai_settings
from backend.core.exceptions import DetectorInferenceException, DetectorUnavailableException
from backend.core.logging import logger
from backend.schemas.detection import (
    DetectionItem,
    DetectionResultContainer,
    DetectorSource,
    NormalizedBoundingBox,
    TargetCategory,
)
from backend.services.ai.document_detection_service import document_detection_service
from backend.services.ai.laptop_screen_detection_service import laptop_screen_detection_service
from backend.services.ai.phone_screen_detection_service import phone_screen_detection_service
from backend.services.ai.model_manager import model_manager


class DetectionService:
    """Runs only detectors with semantics suitable for SafeFrame targets.

    Three genuine, category-specific detectors are active:

    * FACES - YuNet ONNX (``cv2.FaceDetectorYN``).
    * LICENSE_PLATES - RT-DETRv2 (R18) ONNX trained on Open Images plates,
      run through OpenCV DNN.
    * ID_CARDS / PASSPORTS - DocAligner FastViT-SA24 corner-heatmap ONNX
      trained on MIDV identity documents. The graph is class-agnostic, so both
      document categories share it; every detected document is masked whole.
    * LAPTOP_SCREENS - RT-DETR R18 (COCO) ONNX for the device proposal plus a
      dedicated display-quad refinement stage. The COCO graph does not predict
      screen corners, so a detection is only reported as a refined screen when
      that stage passes its gates; otherwise the device box is masked and the
      region kind is reported as a fallback.

    The bundled COCO YOLO model is intentionally not used: it cannot run in this
    build and adds no screen semantics. Phone screens still have no validated
    detector and stay unsupported.
    """
    supported_categories = {
        TargetCategory.FACES,
        TargetCategory.LICENSE_PLATES,
        TargetCategory.ID_CARDS,
        TargetCategory.PASSPORTS,
        TargetCategory.LAPTOP_SCREENS,
        TargetCategory.PHONE_SCREENS,
    }
    document_categories = {TargetCategory.ID_CARDS, TargetCategory.PASSPORTS}

    def __init__(self) -> None:
        self.model_manager = model_manager
        self.document_service = document_detection_service
        self.laptop_service = laptop_screen_detection_service
        self.phone_service = phone_screen_detection_service
        self.conf_threshold = ai_settings.DETECTION_CONF_THRESHOLD
        self.plate_conf_threshold = ai_settings.PLATE_CONF_THRESHOLD
        self.plate_nms_threshold = ai_settings.PLATE_NMS_IOU_THRESHOLD
        self.plate_input_size = ai_settings.PLATE_DETECTOR_INPUT_SIZE
        self.plate_max_detections = ai_settings.PLATE_MAX_DETECTIONS

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
        """Preflight used before a job is accepted.

        Each requested category is checked against its own detector. A missing
        plate model therefore fails a plate job explicitly instead of silently
        processing only faces.
        """
        requested = self._requested_categories(target_categories)

        if TargetCategory.FACES in requested:
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

        if TargetCategory.LICENSE_PLATES in requested:
            handle = self.model_manager.ensure_plate_detector()
            if handle.get("model") is None or handle.get("status") != "LOADED":
                if handle.get("status") == "ERROR":
                    detail = ("License-plate detector failed to initialize; verify the ONNX "
                              "weights are valid.")
                else:
                    weights_name = os.path.basename(
                        handle.get("weights_path") or ai_settings.PLATE_MODEL_PATH)
                    detail = (f"License-plate detector weights '{weights_name}' are not "
                              "available on the server. Run "
                              "'python scripts/download_plate_model.py'.")
                raise DetectorUnavailableException(detail)

        if requested & self.document_categories:
            handle = self.model_manager.ensure_document_detector()
            if handle.get("model") is None or handle.get("status") != "LOADED":
                if handle.get("status") == "ERROR":
                    detail = ("Identity-document detector failed to initialize; verify the ONNX "
                              "weights are valid.")
                else:
                    weights_name = os.path.basename(
                        handle.get("weights_path") or ai_settings.DOCUMENT_MODEL_PATH)
                    detail = (f"Identity-document detector weights '{weights_name}' are not "
                              "available on the server. Run "
                              "'python scripts/download_document_model.py'.")
                raise DetectorUnavailableException(detail)

        if TargetCategory.LAPTOP_SCREENS in requested:
            handle = self.model_manager.ensure_laptop_detector()
            if handle.get("model") is None or handle.get("status") != "LOADED":
                if handle.get("status") == "ERROR":
                    detail = ("Laptop-screen detector failed to initialize; verify the ONNX "
                              "weights and that onnxruntime is installed.")
                else:
                    weights_name = os.path.basename(
                        handle.get("weights_path") or ai_settings.LAPTOP_MODEL_PATH)
                    detail = (f"Laptop-screen detector weights '{weights_name}' are not "
                              "available on the server. Run "
                              "'python scripts/download_laptop_screen_model.py'.")
                raise DetectorUnavailableException(detail)

        if TargetCategory.PHONE_SCREENS in requested:
            handle = self.model_manager.ensure_phone_detector()
            if handle.get("model") is None or handle.get("status") != "LOADED":
                if handle.get("status") == "ERROR":
                    detail = ("Phone-screen detector failed to initialize; verify the ONNX "
                              "weights and that onnxruntime is installed.")
                else:
                    weights_name = os.path.basename(
                        handle.get("weights_path") or ai_settings.PHONE_MODEL_PATH)
                    detail = (f"Phone-screen detector weights '{weights_name}' are not "
                              "available on the server. Run "
                              "'python scripts/download_phone_screen_model.py'.")
                raise DetectorUnavailableException(detail)

    def detect_entities(self, frame: Optional[np.ndarray], target_categories: List[str], frame_index: int = 0,
                        custom_conf_threshold: Optional[float] = None) -> DetectionResultContainer:
        start = time.time()
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise DetectorInferenceException(f"frame {frame_index} is empty or unreadable")
        requested = self._requested_categories(target_categories)
        if not requested:
            return DetectionResultContainer(frame_index=frame_index, total_detections=0, detections=[],
                                            inference_time_ms=0.0, categories_detected=[],
                                            secondary_detector_used=False, plate_detector_used=False,
                                            document_detector_used=False)

        self.validate_available(target_categories)

        detections: List[DetectionItem] = []
        # Every detector consumes the SAME decoded frame; no extra decode or copy
        # of the full frame is performed for additional categories.
        if TargetCategory.FACES in requested:
            detections.extend(self._detect_faces(frame, frame_index, custom_conf_threshold))
        if TargetCategory.LICENSE_PLATES in requested:
            detections.extend(self._detect_plates(frame, frame_index))
        if requested & self.document_categories:
            # One class-agnostic document detector serves both document
            # categories. It runs exactly once, no matter whether the request
            # asks for ID cards, passports, or both.
            detections.extend(self._detect_documents(frame, frame_index))
        laptop_inference_ms = 0.0
        if TargetCategory.LAPTOP_SCREENS in requested:
            detections.extend(self._detect_laptop_screens(frame, frame_index))
            laptop_inference_ms = float(self.laptop_service.last_inference_ms)
        phone_inference_ms = 0.0
        if TargetCategory.PHONE_SCREENS in requested:
            detections.extend(self._detect_phone_screens(frame, frame_index))
            phone_inference_ms = float(self.phone_service.last_inference_ms)

        categories_detected: List[TargetCategory] = []
        for category in (TargetCategory.FACES, TargetCategory.LICENSE_PLATES,
                         TargetCategory.ID_CARDS, TargetCategory.LAPTOP_SCREENS,
                         TargetCategory.PHONE_SCREENS):
            if any(detection.category == category for detection in detections):
                categories_detected.append(category)

        elapsed = round((time.time() - start) * 1000, 2)
        return DetectionResultContainer(
            frame_index=frame_index, total_detections=len(detections), detections=detections,
            inference_time_ms=elapsed, categories_detected=categories_detected,
            secondary_detector_used=TargetCategory.FACES in requested,
            plate_detector_used=TargetCategory.LICENSE_PLATES in requested,
            document_detector_used=bool(requested & self.document_categories),
            laptop_detector_used=TargetCategory.LAPTOP_SCREENS in requested,
            laptop_inference_ms=laptop_inference_ms,
            phone_detector_used=TargetCategory.PHONE_SCREENS in requested,
            phone_inference_ms=phone_inference_ms,
        )

    # ------------------------------------------------------------------- faces

    def _detect_faces(self, frame: np.ndarray, frame_index: int,
                      custom_conf_threshold: Optional[float]) -> List[DetectionItem]:
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
        return detections

    # ------------------------------------------------------------------ plates

    def _detect_plates(self, frame: np.ndarray, frame_index: int) -> List[DetectionItem]:
        """Runs the RT-DETR plate graph and decodes its two outputs.

        Input contract (from the pinned model card): RGB, /255, stretched to
        640x640 (no crop, no mean/std). Outputs are ``logits`` [1,300,1] and
        ``pred_boxes`` [1,300,4] as normalized cx,cy,w,h. RT-DETR is NMS-free,
        but the queries can overlap, so a light NMS avoids masking the same
        plate twice.
        """
        handle = self.model_manager.ensure_plate_detector()
        net = handle.get("model")
        if net is None or handle.get("status") != "LOADED":
            raise DetectorUnavailableException(
                handle.get("detail") or "license-plate detector is not available")

        height, width = frame.shape[:2]
        size = self.plate_input_size
        # swapRB=True converts BGR -> RGB inside blobFromImage, avoiding a
        # separate full-frame colour conversion copy.
        blob = cv2.dnn.blobFromImage(
            frame, scalefactor=1.0 / 255.0, size=(size, size), swapRB=True, crop=False)
        try:
            net.setInput(blob)
            output_names = handle.get("output_names") or net.getUnconnectedOutLayersNames()
            outputs = net.forward(list(output_names))
        except Exception as exc:
            logger.error("Plate detector inference failed on frame %s: %s", frame_index, exc)
            raise DetectorInferenceException(
                f"license-plate detector failed on frame {frame_index}: {exc}") from exc

        if len(outputs) < 2:
            raise DetectorInferenceException(
                f"license-plate detector returned {len(outputs)} outputs on frame {frame_index}")

        by_name = dict(zip(output_names, outputs))
        logits = by_name.get("logits", outputs[0]).reshape(-1)
        boxes = by_name.get("pred_boxes", outputs[1]).reshape(-1, 4)
        if logits.shape[0] != boxes.shape[0] or logits.size == 0:
            raise DetectorInferenceException(
                f"license-plate detector output shapes disagree on frame {frame_index}")

        # Single-class sigmoid scores. The graph emits raw logits.
        scores = 1.0 / (1.0 + np.exp(-logits.astype(np.float32)))
        keep = np.where(scores >= self.plate_conf_threshold)[0]
        if keep.size == 0:
            return []

        pixel_boxes: List[List[int]] = []
        confidences: List[float] = []
        for index in keep:
            cx, cy, bw, bh = (float(value) for value in boxes[index])
            x1 = max(0.0, (cx - bw / 2.0) * width)
            y1 = max(0.0, (cy - bh / 2.0) * height)
            x2 = min(float(width), (cx + bw / 2.0) * width)
            y2 = min(float(height), (cy + bh / 2.0) * height)
            if x2 - x1 < 2.0 or y2 - y1 < 2.0:
                continue
            pixel_boxes.append([int(round(x1)), int(round(y1)),
                                int(round(x2 - x1)), int(round(y2 - y1))])
            confidences.append(float(scores[index]))

        if not pixel_boxes:
            return []

        selected = cv2.dnn.NMSBoxes(
            pixel_boxes, confidences, self.plate_conf_threshold, self.plate_nms_threshold)
        indices = np.array(selected).flatten() if len(selected) else []
        detections: List[DetectionItem] = []
        for position in indices[: self.plate_max_detections]:
            x, y, w, h = pixel_boxes[int(position)]
            confidence = min(1.0, max(0.0, confidences[int(position)]))
            nx = max(0.0, min(100.0, x / width * 100.0))
            ny = max(0.0, min(100.0, y / height * 100.0))
            nw = max(0.1, min(100.0 - nx, w / width * 100.0))
            nh = max(0.1, min(100.0 - ny, h / height * 100.0))
            detections.append(DetectionItem(
                category=TargetCategory.LICENSE_PLATES, label="PLATE",
                confidence=round(confidence, 3),
                bbox=NormalizedBoundingBox(
                    x=round(nx, 2), y=round(ny, 2),
                    width=round(nw, 2), height=round(nh, 2)),
                source=DetectorSource.RTDETR_PLATE,
            ))
        return detections

    # ---------------------------------------------------------------- documents

    def _detect_documents(self, frame: np.ndarray, frame_index: int) -> List[DetectionItem]:
        """Runs the class-agnostic identity-document detector.

        Detections are attributed to ``ID_CARDS`` (the canonical document
        bucket). The graph does not distinguish an ID card from a passport, so
        no per-type label is claimed; the requested category only decides
        whether this detector runs.
        """
        results = self.document_service.detect(frame, frame_index=frame_index)
        detections: List[DetectionItem] = []
        for result in results:
            detections.append(DetectionItem(
                category=TargetCategory.ID_CARDS,
                label="DOCUMENT",
                confidence=min(1.0, max(0.0, float(result.confidence))),
                bbox=result.bbox,
                source=DetectorSource.DOCUMENT_CORNER,
            ))
        return detections

    # ---------------------------------------------------------- laptop screens

    def _detect_laptop_screens(self, frame: np.ndarray, frame_index: int) -> List[DetectionItem]:
        """Runs the laptop device detector plus display-quad refinement.

        A detection whose refinement failed is still emitted (masking the whole
        device box) and carries ``region_kind="laptop_fallback"`` so the
        summary can report that the screen boundary was not separated.
        """
        results = self.laptop_service.detect(frame, frame_index=frame_index)
        detections: List[DetectionItem] = []
        for result in results:
            detections.append(DetectionItem(
                category=TargetCategory.LAPTOP_SCREENS,
                label="LAPTOP SCREEN",
                confidence=min(1.0, max(0.0, float(result.confidence))),
                bbox=result.bbox,
                source=DetectorSource.RTDETR_LAPTOP,
                quad=[[float(x), float(y)] for x, y in result.quad],
                region_kind=result.region_kind,
            ))
        return detections

    # ----------------------------------------------------------- phone screens

    def _detect_phone_screens(self, frame: np.ndarray, frame_index: int) -> List[DetectionItem]:
        """Runs the mobile-phone detector plus display-quad refinement.

        A detection whose refinement failed is still emitted (masking the whole
        device box) and carries ``region_kind="phone_fallback"`` so the
        summary can report that the screen boundary was not separated.
        """
        results = self.phone_service.detect(frame, frame_index=frame_index)
        detections: List[DetectionItem] = []
        for result in results:
            detections.append(DetectionItem(
                category=TargetCategory.PHONE_SCREENS,
                label="PHONE SCREEN",
                confidence=min(1.0, max(0.0, float(result.confidence))),
                bbox=result.bbox,
                source=DetectorSource.RTDETR_PHONE,
                quad=[[float(x), float(y)] for x, y in result.quad],
                region_kind=result.region_kind,
            ))
        return detections


detection_service = DetectionService()
