"""Laptop-screen detection and display-region estimation (Phase 4).

Model
-----
Device proposal comes from RT-DETR R18 trained on COCO (Apache-2.0 source
weights, ``PekingU/rtdetr_r18vd``), exported to ONNX and executed with
onnxruntime. The graph predicts a bounding box for the COCO ``laptop`` class.
It does **not** predict the display quadrilateral.

Screen region
-------------
A dedicated refinement stage estimates the visible display inside the device
box from real edge/line evidence:

* edge map -> quad candidates from contour approximation and from pairs of
  detected near-horizontal/near-vertical lines (perspective-distorted screens
  are quadrilaterals, not rectangles);
* each candidate is scored on perimeter edge support and border luminance
  contrast, and gated on geometry: it must cover a large part of the device
  box, start at its top edge, stay above the keyboard deck, and have a
  display-like aspect ratio.

A candidate is only accepted when every gate passes. Otherwise the region kind
is reported as ``laptop_fallback`` and the whole device box is masked: the
pipeline fails safe (nothing on the display stays visible) and the summary
reports that the screen boundary could not be separated, instead of pretending
a display quad was found. No accuracy is claimed for the bounding-box-only
model beyond what ``scripts/evaluate_laptop_screen_detection.py`` measures.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.config.ai_settings import ai_settings
from backend.core.exceptions import DetectorInferenceException, DetectorUnavailableException
from backend.core.logging import logger
from backend.schemas.detection import NormalizedBoundingBox
from backend.services.ai.model_manager import model_manager

Point = Tuple[float, float]

REGION_SCREEN_QUAD = "screen_quad"
REGION_LAPTOP_FALLBACK = "laptop_fallback"


@dataclass
class LaptopScreenDetection:
    """One laptop in a frame, with the display region that must be masked.

    ``quad`` always has four corners in frame pixel coordinates. ``region_kind``
    says whether those corners are the refined display boundary
    (``screen_quad``) or the conservative whole-device fallback
    (``laptop_fallback``).
    """

    bbox: NormalizedBoundingBox
    confidence: float
    quad: List[Point] = field(default_factory=list)
    region_kind: str = REGION_LAPTOP_FALLBACK
    evidence: Dict[str, float] = field(default_factory=dict)


def order_quad(points: Any) -> np.ndarray:
    """Orders four corners as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    total = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(total)],
        pts[np.argmin(diff)],
        pts[np.argmax(total)],
        pts[np.argmax(diff)],
    ], dtype=np.float32)


def quad_metrics(quad: np.ndarray) -> Tuple[float, float, float]:
    """Returns (mean width, mean height, polygon area) of a quad."""
    top_left, top_right, bottom_right, bottom_left = quad
    width = (np.linalg.norm(top_right - top_left)
             + np.linalg.norm(bottom_right - bottom_left)) / 2.0
    height = (np.linalg.norm(bottom_left - top_left)
              + np.linalg.norm(bottom_right - top_right)) / 2.0
    return float(width), float(height), float(cv2.contourArea(quad.astype(np.int32)))


def edge_support(edge_map: np.ndarray, quad: np.ndarray, samples: int = 120) -> float:
    """Fraction of perimeter samples that land on a (dilated) image edge."""
    total = hits = 0
    height, width = edge_map.shape[:2]
    for index in range(4):
        start = quad[index]
        end = quad[(index + 1) % 4]
        for t in np.linspace(0.0, 1.0, samples, endpoint=False):
            x, y = start + (end - start) * t
            xi, yi = int(round(float(x))), int(round(float(y)))
            if 0 <= yi < height and 0 <= xi < width:
                total += 1
                hits += int(edge_map[yi, xi] > 0)
    return hits / total if total else 0.0


def border_contrast(gray: np.ndarray, quad: np.ndarray, offset: int = 4) -> float:
    """Mean luminance step across the quad's sides (0..1).

    A real display boundary separates screen content from a bezel; an arbitrary
    rectangle drawn over a flat surface does not.
    """
    height, width = gray.shape[:2]
    values: List[float] = []

    def sample(point: np.ndarray) -> Optional[float]:
        xi, yi = int(round(float(point[0]))), int(round(float(point[1])))
        if 0 <= yi < height and 0 <= xi < width:
            return float(gray[yi, xi])
        return None

    for index in range(4):
        start = quad[index]
        end = quad[(index + 1) % 4]
        direction = end - start
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-3:
            continue
        normal /= norm
        for t in (0.2, 0.5, 0.8):
            point = start + direction * t
            inside = sample(point - normal * offset)
            outside = sample(point + normal * offset)
            if inside is not None and outside is not None:
                values.append(abs(inside - outside) / 255.0)
    return float(np.mean(values)) if values else 0.0


class LaptopScreenDetectionService:
    """Runs the RT-DETR laptop proposal and estimates the display region."""

    def __init__(self, manager: Any = None) -> None:
        self.model_manager = manager or model_manager
        self.input_size = ai_settings.LAPTOP_DETECTOR_INPUT_SIZE
        self.class_index = ai_settings.LAPTOP_CLASS_INDEX
        self.conf_threshold = ai_settings.LAPTOP_CONF_THRESHOLD
        self.nms_threshold = ai_settings.LAPTOP_NMS_IOU_THRESHOLD
        self.max_detections = ai_settings.LAPTOP_MAX_DETECTIONS
        self.min_area_ratio = ai_settings.LAPTOP_MIN_AREA_RATIO
        self.max_area_ratio = ai_settings.LAPTOP_MAX_AREA_RATIO
        self.refine_scale = ai_settings.LAPTOP_SCREEN_REFINE_SCALE
        self.min_coverage = ai_settings.LAPTOP_SCREEN_MIN_COVERAGE
        self.max_coverage = ai_settings.LAPTOP_SCREEN_MAX_COVERAGE
        self.min_aspect = ai_settings.LAPTOP_SCREEN_MIN_ASPECT
        self.max_aspect = ai_settings.LAPTOP_SCREEN_MAX_ASPECT
        self.top_tolerance = ai_settings.LAPTOP_SCREEN_TOP_TOLERANCE
        self.min_bottom = ai_settings.LAPTOP_SCREEN_MIN_BOTTOM
        self.max_bottom = ai_settings.LAPTOP_SCREEN_MAX_BOTTOM
        self.min_edge_support = ai_settings.LAPTOP_SCREEN_MIN_EDGE_SUPPORT
        self.min_border_contrast = ai_settings.LAPTOP_SCREEN_MIN_BORDER_CONTRAST
        self.min_score = ai_settings.LAPTOP_SCREEN_MIN_SCORE
        self.last_inference_ms: float = 0.0
        self.last_refine_ms: float = 0.0

    # ------------------------------------------------------------------ public

    def detect(self, frame: np.ndarray, frame_index: int = 0) -> List[LaptopScreenDetection]:
        """Returns every laptop found in ``frame`` with its redaction region.

        Raises DetectorUnavailableException when the pinned graph cannot be
        loaded and DetectorInferenceException when inference fails. It never
        returns a fabricated detection or a fabricated display boundary.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise DetectorInferenceException(
                f"laptop-screen detector received an empty frame at index {frame_index}")

        handle = self.model_manager.ensure_laptop_detector()
        session = handle.get("model")
        if session is None or handle.get("status") != "LOADED":
            raise DetectorUnavailableException(
                handle.get("detail") or "laptop-screen detector is not available")

        height, width = frame.shape[:2]
        boxes, confidences = self._propose_laptops(session, handle, frame, frame_index)

        detections: List[LaptopScreenDetection] = []
        refine_ms = 0.0
        for box, confidence in zip(boxes, confidences):
            import time as _time

            started = _time.time()
            quad, evidence = self._estimate_screen_quad(frame, box)
            refine_ms += (_time.time() - started) * 1000.0
            if quad is None:
                quad = np.array([
                    [box[0], box[1]], [box[2], box[1]],
                    [box[2], box[3]], [box[0], box[3]],
                ], dtype=np.float32)
                region_kind = REGION_LAPTOP_FALLBACK
                evidence = {"score": 0.0}
            else:
                region_kind = REGION_SCREEN_QUAD

            x, y, w, h = cv2.boundingRect(np.round(quad).astype(np.int32))
            box_normalized = self._normalize_box(x, y, w, h, width, height)
            if box_normalized is None:
                continue
            detections.append(LaptopScreenDetection(
                bbox=box_normalized,
                confidence=round(float(confidence), 3),
                quad=[(float(px), float(py)) for px, py in quad],
                region_kind=region_kind,
                evidence={key: round(float(value), 3) for key, value in evidence.items()},
            ))
        self.last_refine_ms = round(refine_ms, 2)
        return detections

    # ------------------------------------------------------------------ decode

    def _propose_laptops(self, session: Any, handle: Dict[str, Any],
                         frame: np.ndarray, frame_index: int) -> Tuple[List[List[int]], List[float]]:
        """Runs the COCO graph and returns NMS-filtered laptop boxes."""
        height, width = frame.shape[:2]
        size = self.input_size
        # RT-DETR contract (from the model's preprocessor config): RGB, /255,
        # ImageNet mean/std, stretched to a square. BGR->RGB happens inside
        # cvtColor so no extra full-frame copy is made.
        rgb = cv2.cvtColor(cv2.resize(frame, (size, size),
                                      interpolation=cv2.INTER_LINEAR),
                           cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        blob = ((rgb - mean) / std).transpose(2, 0, 1)[np.newaxis, ...]

        import time as _time

        started = _time.time()
        try:
            outputs = session.run(None, {session.get_inputs()[0].name: blob})
        except Exception as exc:
            logger.error("Laptop detector inference failed on frame %s: %s", frame_index, exc)
            raise DetectorInferenceException(
                f"laptop-screen detector failed on frame {frame_index}: {exc}") from exc
        self.last_inference_ms = round((_time.time() - started) * 1000.0, 2)

        names = handle.get("output_names") or tuple(
            output.name for output in session.get_outputs())
        by_name = dict(zip(names, outputs))
        if "logits" not in by_name or "pred_boxes" not in by_name:
            raise DetectorInferenceException(
                f"laptop detector outputs {list(names)} are not logits/pred_boxes")
        logits = np.asarray(by_name["logits"]).reshape(-1, np.asarray(by_name["logits"]).shape[-1])
        boxes = np.asarray(by_name["pred_boxes"]).reshape(-1, 4)
        if logits.shape[0] != boxes.shape[0] or logits.shape[0] == 0:
            raise DetectorInferenceException(
                f"laptop detector output shapes disagree on frame {frame_index}")
        if logits.shape[1] <= self.class_index:
            raise DetectorInferenceException(
                f"laptop detector has {logits.shape[1]} classes, "
                f"cannot read class index {self.class_index}")

        # RT-DETR emits raw logits per class; the COCO scores are sigmoids.
        scores = 1.0 / (1.0 + np.exp(-logits.astype(np.float32)))
        laptop_scores = scores[:, self.class_index]
        keep = np.where(laptop_scores >= self.conf_threshold)[0]
        if keep.size == 0:
            return [], []

        pixel_boxes: List[List[int]] = []
        confidences: List[float] = []
        for index in keep:
            cx, cy, bw, bh = (float(value) for value in boxes[index])
            x1 = max(0.0, (cx - bw / 2.0) * width)
            y1 = max(0.0, (cy - bh / 2.0) * height)
            x2 = min(float(width), (cx + bw / 2.0) * width)
            y2 = min(float(height), (cy + bh / 2.0) * height)
            if x2 - x1 < 16.0 or y2 - y1 < 16.0:
                continue
            area_ratio = ((x2 - x1) * (y2 - y1)) / float(max(1, width * height))
            if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
                continue
            pixel_boxes.append([int(round(x1)), int(round(y1)),
                                int(round(x2 - x1)), int(round(y2 - y1))])
            confidences.append(float(laptop_scores[index]))

        if not pixel_boxes:
            return [], []

        selected = cv2.dnn.NMSBoxes(
            pixel_boxes, confidences, self.conf_threshold, self.nms_threshold)
        indices = np.array(selected).flatten() if len(selected) else []
        boxes_out: List[List[int]] = []
        confidences_out: List[float] = []
        for position in indices[: self.max_detections]:
            x, y, w, h = pixel_boxes[int(position)]
            boxes_out.append([float(x), float(y), float(x + w), float(y + h)])
            confidences_out.append(min(1.0, max(0.0, confidences[int(position)])))
        return boxes_out, confidences_out

    def _normalize_box(self, x: int, y: int, w: int, h: int,
                       width: int, height: int) -> Optional[NormalizedBoundingBox]:
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

    # --------------------------------------------------------------- refinement

    def _estimate_screen_quad(self, frame: np.ndarray, box: Sequence[float]) -> Tuple[
            Optional[np.ndarray], Dict[str, float]]:
        """Estimates the display quadrilateral inside a device box.

        Returns ``(quad_pixels, evidence)`` where quad is None when no candidate
        passes every gate. The caller then masks the device box (fail safe).
        """
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = (float(value) for value in box)
        box_w, box_h = x2 - x1, y2 - y1
        if box_w < 24.0 or box_h < 24.0:
            return None, {}

        margin_x, margin_y = box_w * 0.08, box_h * 0.08
        crop_x1 = max(0, int(x1 - margin_x))
        crop_y1 = max(0, int(y1 - margin_y))
        crop_x2 = min(width, int(x2 + margin_x))
        crop_y2 = min(height, int(y2 + margin_y))
        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            return None, {}

        scale = float(self.refine_scale) / max(crop.shape[:2])
        if scale < 1.0:
            crop = cv2.resize(crop, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        median = float(np.median(blurred))
        low = int(max(20, 0.66 * median))
        high = int(min(255, max(low + 30, 1.33 * median)))
        edges = cv2.Canny(blurred, low, high)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8))

        crop_h, crop_w = gray.shape
        candidates = self._contour_candidates(edges, crop_h * crop_w * 0.06)
        horizontal, vertical = self._line_segments(
            edges, min(crop_w, crop_h) * 0.22, max(crop_w, crop_h) * 0.03)
        candidates += self._quads_from_segments(horizontal, vertical)

        box_area_crop = max(1.0, box_w * scale * box_h * scale)
        box_top_crop = (y1 - crop_y1) * scale
        box_height_crop = box_h * scale

        best: Optional[Tuple[float, Dict[str, float], np.ndarray]] = None
        for candidate in candidates:
            quad = order_quad(candidate)
            if not cv2.isContourConvex(quad.astype(np.int32)):
                continue
            quad_w, quad_h, area = quad_metrics(quad)
            if quad_w <= 1.0 or quad_h <= 1.0:
                continue
            aspect = quad_w / quad_h
            coverage = area / box_area_crop
            if not (self.min_coverage <= coverage <= self.max_coverage):
                continue
            if not (self.min_aspect <= aspect <= self.max_aspect):
                continue
            if np.any(quad < -2) or np.any(quad[:, 0] > crop_w + 2) \
                    or np.any(quad[:, 1] > crop_h + 2):
                continue
            top = float(quad[:, 1].min())
            bottom = float(quad[:, 1].max())
            # A display starts at the top of the device box and ends above the
            # keyboard deck. These two gates reject inner windows and quads
            # that swallow the keyboard.
            if top > box_top_crop + self.top_tolerance * box_height_crop:
                continue
            if not (box_top_crop + self.min_bottom * box_height_crop <= bottom
                    <= box_top_crop + self.max_bottom * box_height_crop):
                continue
            # The bottom edge is what separates the display from the keyboard
            # deck. A quad whose bottom is only an inner panel border (a window,
            # a dialog) would leave real display content below it visible, so it
            # is only accepted when a long horizontal line corroborates it.
            if not self._bottom_edge_corroborated(horizontal, quad):
                continue
            support = edge_support(dilated, quad)
            contrast = border_contrast(gray, quad)
            if support < self.min_edge_support or contrast < self.min_border_contrast:
                continue
            aspect_fit = max(0.0, 1.0 - abs(aspect - 1.55) / 1.2)
            score = 0.52 * support + 0.28 * min(1.0, contrast * 8.0) + 0.20 * aspect_fit
            if score < self.min_score:
                continue
            evidence = {
                "score": score,
                "edge_support": support,
                "border_contrast": contrast,
                "aspect": aspect,
                "coverage": coverage,
            }
            if best is None or score > best[0]:
                best = (score, evidence, quad)

        if best is None:
            return None, {}
        quad = best[2] / scale + np.array([crop_x1, crop_y1], dtype=np.float32)
        return quad, best[1]

    @staticmethod
    def _contour_candidates(edges: np.ndarray, min_area: float) -> List[np.ndarray]:
        """Quad candidates from approximated contours."""
        candidates: List[np.ndarray] = []
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            for epsilon in (0.01, 0.02, 0.035, 0.05):
                approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    candidates.append(approx.reshape(4, 2).astype(np.float32))
        return candidates

    @staticmethod
    def _line_segments(edges: np.ndarray, min_len: float, max_gap: float) -> Tuple[
            List[Tuple[Tuple[int, int], Tuple[int, int]]],
            List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
        """Detects near-horizontal and near-vertical line segments."""
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180.0, threshold=60,
            minLineLength=int(max(8, min_len)), maxLineGap=int(max(2, max_gap)))
        horizontal: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        vertical: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        if lines is None:
            return horizontal, vertical
        for x_start, y_start, x_end, y_end in lines.reshape(-1, 4):
            angle = abs(np.degrees(np.arctan2(y_end - y_start, x_end - x_start)))
            if angle > 90.0:
                angle = 180.0 - angle
            segment = ((int(x_start), int(y_start)), (int(x_end), int(y_end)))
            if angle <= 25.0:
                horizontal.append(segment)
            elif angle >= 65.0:
                vertical.append(segment)

        def merge(segments: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                  offset_of: Any) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
            # Near-collinear segments are collapsed to their longest member so
            # one physical edge does not multiply candidate pairs.
            best: Dict[int, Tuple[float, Tuple[Tuple[int, int], Tuple[int, int]]]] = {}
            for start, end in segments:
                middle = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
                length = float(np.hypot(end[0] - start[0], end[1] - start[1]))
                key = int(round(offset_of(middle) / 12.0))
                if key not in best or length > best[key][0]:
                    best[key] = (length, (start, end))
            return [value[1] for value in best.values()]

        return merge(horizontal, lambda middle: middle[1]), \
            merge(vertical, lambda middle: middle[0])

    @staticmethod
    def _quads_from_segments(
            horizontal: List[Tuple[Tuple[int, int], Tuple[int, int]]],
            vertical: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> List[np.ndarray]:
        """Quad candidates from intersecting horizontal/vertical line pairs."""
        if len(horizontal) < 2 or len(vertical) < 2:
            return []
        candidates: List[np.ndarray] = []
        for i in range(len(horizontal)):
            for j in range(i + 1, len(horizontal)):
                for k in range(len(vertical)):
                    for l in range(k + 1, len(vertical)):
                        quad = LaptopScreenDetectionService._intersect_quad(
                            horizontal[i], horizontal[j], vertical[k], vertical[l])
                        if quad is not None:
                            candidates.append(quad)
        return candidates

    @staticmethod
    def _bottom_edge_corroborated(
            horizontal: List[Tuple[Tuple[int, int], Tuple[int, int]]],
            quad: np.ndarray) -> bool:
        """True when a long detected line supports the quad's bottom edge.

        Projects each near-horizontal segment onto the quad's bottom edge and
        measures how much of that edge it spans. This rejects quads whose bottom
        is an inner panel border rather than the real display/keyboard boundary.
        """
        bottom_left, bottom_right = quad[3], quad[2]
        edge = bottom_right - bottom_left
        edge_length = float(np.linalg.norm(edge))
        if edge_length < 8.0:
            return False
        direction = edge / edge_length
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)
        covered = 0.0
        for (x1, y1), (x2, y2) in horizontal:
            start = np.array([x1, y1], dtype=np.float32)
            end = np.array([x2, y2], dtype=np.float32)
            if (abs(float(np.dot(start - bottom_left, normal))) > 8.0
                    or abs(float(np.dot(end - bottom_left, normal))) > 8.0):
                continue
            t_start = float(np.dot(start - bottom_left, direction))
            t_end = float(np.dot(end - bottom_left, direction))
            covered = max(covered, abs(t_end - t_start))
        return covered >= ai_settings.LAPTOP_SCREEN_MIN_BOTTOM_SPAN * edge_length

    @staticmethod
    def _intersect_quad(top: Tuple[Tuple[int, int], Tuple[int, int]],
                        bottom: Tuple[Tuple[int, int], Tuple[int, int]],
                        left: Tuple[Tuple[int, int], Tuple[int, int]],
                        right: Tuple[Tuple[int, int], Tuple[int, int]]) -> Optional[np.ndarray]:
        corners = [
            LaptopScreenDetectionService._line_intersection(top, left),
            LaptopScreenDetectionService._line_intersection(top, right),
            LaptopScreenDetectionService._line_intersection(bottom, right),
            LaptopScreenDetectionService._line_intersection(bottom, left),
        ]
        if any(corner is None for corner in corners):
            return None
        quad = np.array(corners, dtype=np.float32)
        if not np.all(np.isfinite(quad)):
            return None
        return quad

    @staticmethod
    def _line_intersection(first: Tuple[Tuple[int, int], Tuple[int, int]],
                           second: Tuple[Tuple[int, int], Tuple[int, int]]) -> Optional[np.ndarray]:
        (x1, y1), (x2, y2) = first
        (x3, y3), (x4, y4) = second
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denominator) < 1e-6:
            return None
        px = ((x1 * y2 - y1 * x2) * (x3 - x4)
              - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
        py = ((x1 * y2 - y1 * x2) * (y3 - y4)
              - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
        return np.array([px, py], dtype=np.float32)


laptop_screen_detection_service = LaptopScreenDetectionService()
