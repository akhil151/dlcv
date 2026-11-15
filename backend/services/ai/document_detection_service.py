"""Identity-document detection (ID cards and passports).

The detector is DocAligner's FastViT-SA24 corner-heatmap regressor: a real,
pretrained ONNX graph trained on MIDV-500/MIDV-2020 identity-document captures.
It predicts four document-corner heatmaps; this module decodes them into a
document quadrilateral and its bounding box.

Design notes
------------
* The graph is class-agnostic. It locates a document; it does not label an ID
  card vs a passport. Both UI categories therefore use this one validated
  detector, and every detection is redacted in full.
* A document whose fourth corner lies outside the frame (partially visible)
  still yields three confident corners. The fourth is reconstructed as the
  parallelogram vertex implied by the other three. This is geometry on real
  heatmap corners, not a fabricated detection, and it is required so a document
  leaving the frame cannot leave sensitive pixels uncovered.
* DocAligner returns one document per inference. Overlapping half-strips are
  additionally scanned so two or more documents in one frame can be located;
  detections are merged by mask IoU so the same document is not masked twice.
* Every detection is clamped to the frame and filtered by area and aspect ratio,
  which rejects full-frame hallucinations and banner-shaped non-documents.
"""
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from backend.config.ai_settings import ai_settings
from backend.core.exceptions import DetectorInferenceException, DetectorUnavailableException
from backend.core.logging import logger
from backend.schemas.detection import NormalizedBoundingBox
from backend.services.ai.model_manager import model_manager

Point = Tuple[float, float]


@dataclass
class DocumentDetection:
    """One detected identity document in pixel and normalized coordinates."""

    bbox: NormalizedBoundingBox
    confidence: float
    quad: List[Point] = field(default_factory=list)


class DocumentDetectionService:
    """Decodes DocAligner corner heatmaps into document boxes."""

    def __init__(self, manager: Any = None) -> None:
        self.model_manager = manager or model_manager
        self.input_size = ai_settings.DOCUMENT_DETECTOR_INPUT_SIZE
        self.heatmap_size = ai_settings.DOCUMENT_HEATMAP_OUTPUT_SIZE
        self.heatmap_threshold = ai_settings.DOCUMENT_HEATMAP_THRESHOLD
        self.conf_threshold = ai_settings.DOCUMENT_CONF_THRESHOLD
        self.min_area_ratio = ai_settings.DOCUMENT_MIN_AREA_RATIO
        self.max_area_ratio = ai_settings.DOCUMENT_MAX_AREA_RATIO
        self.min_aspect = ai_settings.DOCUMENT_MIN_ASPECT_RATIO
        self.max_aspect = ai_settings.DOCUMENT_MAX_ASPECT_RATIO
        self.multi_scan = ai_settings.DOCUMENT_MULTI_SCAN_ENABLED
        self.strip_overlap = ai_settings.DOCUMENT_SCAN_STRIP_OVERLAP
        self.max_docs = ai_settings.DOCUMENT_MAX_DOCS_PER_FRAME
        self.merge_iou = ai_settings.DOCUMENT_MERGE_MASK_IOU

    # ------------------------------------------------------------------ public

    def detect(self, frame: np.ndarray, frame_index: int = 0) -> List[DocumentDetection]:
        """Returns every identity document found in ``frame``.

        Raises DetectorUnavailableException when the validated graph cannot be
        loaded and DetectorInferenceException when inference itself fails. It
        never returns a fabricated detection.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise DetectorInferenceException(
                f"document detector received an empty frame at index {frame_index}")

        handle = self.model_manager.ensure_document_detector()
        net = handle.get("model")
        if net is None or handle.get("status") != "LOADED":
            raise DetectorUnavailableException(
                handle.get("detail") or "identity-document detector is not available")

        height, width = frame.shape[:2]
        quads: List[Tuple[np.ndarray, float]] = []

        first = self._infer_quad(net, frame, frame_index)
        if first is not None:
            quads.append(first)

        if self.multi_scan and len(quads) < self.max_docs:
            for x0, y0, x1, y1 in self._scan_regions(width, height):
                if len(quads) >= self.max_docs:
                    break
                crop = frame[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                found = self._infer_quad(net, crop, frame_index)
                if found is None:
                    continue
                quad = found[0] + np.array([x0, y0], dtype=np.float32)
                if self._is_duplicate(quad, quads, (height, width)):
                    continue
                quads.append((quad, found[1]))

        detections: List[DocumentDetection] = []
        for quad, confidence in quads:
            box = self._normalize(quad, width, height)
            if box is None:
                continue
            detections.append(DocumentDetection(
                bbox=box,
                confidence=round(float(confidence), 3),
                quad=[(float(x), float(y)) for x, y in quad],
            ))
        return detections

    # ------------------------------------------------------------------ decode

    def _infer_quad(self, net: Any, image: np.ndarray,
                    frame_index: int) -> Optional[Tuple[np.ndarray, float]]:
        height, width = image.shape[:2]
        try:
            blob = cv2.dnn.blobFromImage(
                image, scalefactor=1.0 / 255.0,
                size=(self.input_size, self.input_size), swapRB=False, crop=False)
            net.setInput(blob)
            heatmaps = net.forward()
        except Exception as exc:
            logger.error("Document detector inference failed on frame %s: %s", frame_index, exc)
            raise DetectorInferenceException(
                f"identity-document detector failed on frame {frame_index}: {exc}") from exc

        heatmaps = np.asarray(heatmaps)
        if heatmaps.ndim != 4 or heatmaps.shape[1] != 4:
            raise DetectorInferenceException(
                f"identity-document detector returned heatmaps with shape "
                f"{heatmaps.shape} on frame {frame_index}")

        points: List[Optional[np.ndarray]] = []
        peaks: List[float] = []
        for channel in range(4):
            heatmap = heatmaps[0, channel]
            resized = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
            peaks.append(float(heatmap.max()))
            points.append(self._peak_centroid(resized))

        available = [peak for point, peak in zip(points, peaks) if point is not None]
        confidence = min(available) if available else 0.0
        if confidence < self.conf_threshold:
            return None

        quad = self._complete_quad(points)
        if quad is None:
            return None
        if not self._plausible(quad, width, height):
            return None
        return quad, confidence

    def _peak_centroid(self, heatmap: np.ndarray) -> Optional[np.ndarray]:
        """Centroid of the largest above-threshold corner blob."""
        mask = (heatmap >= self.heatmap_threshold).astype(np.uint8)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        if count <= 1:
            return None
        index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return centroids[index].astype(np.float32)

    @staticmethod
    def _complete_quad(points: Sequence[Optional[np.ndarray]]) -> Optional[np.ndarray]:
        """Returns four corners, reconstructing one missing corner.

        Corner channel order is top-left, top-right, bottom-right, bottom-left.
        For a parallelogram the missing vertex is the sum of its two neighbours
        minus the opposite vertex. More than one missing corner means the frame
        does not show enough of the document to place it reliably, so the
        detection is dropped (Other frames may still catch it).
        """
        corners = list(points)
        missing = [index for index, point in enumerate(corners) if point is None]
        if len(missing) > 1:
            return None
        if missing:
            index = missing[0]
            if index == 0:
                corners[0] = corners[1] + corners[3] - corners[2]
            elif index == 1:
                corners[1] = corners[0] + corners[2] - corners[3]
            elif index == 2:
                corners[2] = corners[1] + corners[3] - corners[0]
            else:
                corners[3] = corners[0] + corners[2] - corners[1]
        quad = np.asarray(corners, dtype=np.float32)
        if not np.all(np.isfinite(quad)):
            return None
        return quad

    def _plausible(self, quad: np.ndarray, width: int, height: int) -> bool:
        """Area/aspect sanity on the detected quadrilateral's bounding box."""
        x, y, w, h = cv2.boundingRect(np.round(quad).astype(np.int32))
        if w < 8 or h < 8:
            return False
        area_ratio = (w * h) / float(max(1, width * height))
        if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
            return False
        aspect = w / float(max(1, h))
        if aspect < self.min_aspect or aspect > self.max_aspect:
            return False
        return cv2.isContourConvex(np.round(quad).astype(np.int32))

    def _scan_regions(self, width: int, height: int) -> List[Tuple[int, int, int, int]]:
        """Overlapping half-frame strips used for additional documents.

        A side-by-side pair is fully contained in the left or right strip; a
        stacked pair is fully contained in the top or bottom strip. The overlap
        keeps a document straddling the centre line whole in at least one strip.
        """
        overlap = max(0.0, min(0.48, self.strip_overlap))
        half_w = min(width, max(1, int(round(width * (0.5 + overlap)))))
        half_h = min(height, max(1, int(round(height * (0.5 + overlap)))))
        right_x = max(0, width - half_w)
        bottom_y = max(0, height - half_h)
        regions = [
            (0, 0, half_w, height),
            (right_x, 0, width, height),
            (0, 0, width, half_h),
            (0, bottom_y, width, height),
        ]
        seen = set()
        unique = []
        for region in regions:
            if region in seen:
                continue
            seen.add(region)
            unique.append(region)
        return unique

    def _is_duplicate(self, quad: np.ndarray,
                      existing: Sequence[Tuple[np.ndarray, float]],
                      shape: Tuple[int, int]) -> bool:
        mask = np.zeros(shape, np.uint8)
        cv2.fillPoly(mask, [np.round(quad).astype(np.int32)], 1)
        for other, _ in existing:
            other_mask = np.zeros(shape, np.uint8)
            cv2.fillPoly(other_mask, [np.round(other).astype(np.int32)], 1)
            intersection = np.logical_and(mask, other_mask).sum()
            union = np.logical_or(mask, other_mask).sum()
            if union and intersection / union >= self.merge_iou:
                return True
        return False

    @staticmethod
    def _normalize(quad: np.ndarray, width: int, height: int) -> Optional[NormalizedBoundingBox]:
        x, y, w, h = cv2.boundingRect(np.round(quad).astype(np.int32))
        x1 = max(0.0, min(float(width), float(x)))
        y1 = max(0.0, min(float(height), float(y)))
        x2 = max(0.0, min(float(width), float(x + w)))
        y2 = max(0.0, min(float(height), float(y + h)))
        if x2 - x1 < 8.0 or y2 - y1 < 8.0:
            return None
        nx = max(0.0, min(100.0, x1 / width * 100.0))
        ny = max(0.0, min(100.0, y1 / height * 100.0))
        nw = max(0.1, min(100.0 - nx, (x2 - x1) / width * 100.0))
        nh = max(0.1, min(100.0 - ny, (y2 - y1) / height * 100.0))
        return NormalizedBoundingBox(
            x=round(nx, 2), y=round(ny, 2),
            width=round(nw, 2), height=round(nh, 2),
        )


document_detection_service = DocumentDetectionService()
