"""Phase 10 - ID card and passport detection, tracking and redaction.

Deterministic tests use an injected fake corner-heatmap network to pin down the
decoder, partial-document completion, multi-document scanning, geometry,
tracking state and API wiring. Tests that need the real DocAligner weights /
FFmpeg / cached MIDV frames are skipped (never reported as passed) when those
dependencies are absent.

Measured detector accuracy on real, ground-truth-annotated MIDV frames lives in
scripts/evaluate_document_detection.py; nothing here claims accuracy from
synthetic heuristics.
"""
import asyncio
import math
import os
import subprocess
import tempfile
import unittest
import uuid

import cv2
import numpy as np

from backend.config.ai_settings import ai_settings
from backend.core.exceptions import DetectorInferenceException, DetectorUnavailableException
from backend.schemas.detection import (
    DetectionItem,
    DetectorSource,
    NormalizedBoundingBox,
    TargetCategory,
)
from backend.schemas.redact import StartRedactionRequest
from backend.schemas.tracking import TrackStateEnum, TrackedObjectItem
from backend.services.ai import model_manager
from backend.services.ai.blur_service import BlurService
from backend.services.ai.detection_service import DetectionService
from backend.services.ai.document_detection_service import DocumentDetectionService
from backend.services.ai.tracking_service import TrackingService
from backend.services.ai.video_processor_service import video_processor_service
from backend.services.redact_service import RedactService
from backend.utils import ffmpeg as ffmpeg_util
from backend.utils.storage import storage_manager
from backend.workers.task_manager import task_manager

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MIDV_DIR = os.path.join(PROJECT_ROOT, ".cache", "docmodels", "midv_val")
MODES = ("gaussian", "pixelate", "blackout")


def _corner_heatmaps(quad):
    """Builds a (1, 4, 128, 128) heatmap stack from four normalized corners.

    ``quad`` entries may be None, which leaves that corner channel empty (the
    partially-visible-document case).
    """
    heatmaps = np.zeros((1, 4, 128, 128), dtype=np.float32)
    if quad:
        for index, point in enumerate(quad):
            if point is None:
                continue
            x = int(round(point[0] * 127))
            y = int(round(point[1] * 127))
            x = max(0, min(127, x))
            y = max(0, min(127, y))
            heatmaps[0, index, max(0, y - 1):min(128, y + 2),
                                  max(0, x - 1):min(128, x + 2)] = 0.9
    return heatmaps


class FakeDocumentNet:
    """Deterministic stand-in for the DocAligner corner-heatmap graph."""

    def __init__(self, quads):
        self.quads = list(quads)
        self.calls = 0
        self.inputs = []
        self.raise_on_forward = False

    def setInput(self, blob):
        self.inputs.append(blob)

    def forward(self):
        if self.raise_on_forward:
            raise RuntimeError("heatmap boom")
        quad = self.quads[self.calls] if self.calls < len(self.quads) else None
        self.calls += 1
        return _corner_heatmaps(quad)


class FakeDocumentManager:
    def __init__(self, net=None, status="LOADED", detail="ok"):
        self.net = net
        self.status = status
        self.detail = detail
        self.calls = 0

    def ensure_document_detector(self):
        self.calls += 1
        return {
            "model": self.net,
            "status": self.status,
            "weights_path": ai_settings.DOCUMENT_MODEL_PATH,
            "detail": self.detail,
        }


class FakeFaceNet:
    def setInputSize(self, size):
        self.size = size

    def detect(self, frame):
        return None, np.array(
            [[10, 10, 20, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.91]])


class FakePlateNet:
    def setInput(self, blob):
        pass

    def getUnconnectedOutLayersNames(self):
        return ("logits", "pred_boxes")

    def forward(self, names=None):
        import math as _math
        logits = np.full((1, 300, 1), _math.log(0.01 / 0.99), dtype=np.float32)
        logits[0, 0, 0] = _math.log(0.5 / 0.5)
        boxes = np.zeros((1, 300, 4), dtype=np.float32)
        boxes[0, 0] = (0.7, 0.5, 0.1, 0.05)
        return [logits, boxes]


def document_detection(x, y, w, h, conf=0.9):
    return DetectionItem(
        category=TargetCategory.ID_CARDS, label="DOCUMENT", confidence=conf,
        bbox=NormalizedBoundingBox(x=x, y=y, width=w, height=h),
        source=DetectorSource.DOCUMENT_CORNER,
    )


def document_track(bbox, track_id=1, state=TrackStateEnum.TRACKED,
                   frames_since_update=0, confirmed=True):
    return TrackedObjectItem(
        track_id=track_id, category=TargetCategory.ID_CARDS,
        label=f"DOC #{track_id}", confidence=0.9, bbox=bbox,
        track_state=state, age=5, frames_since_update=frames_since_update,
        is_confirmed=confirmed,
    )


def noise_frame(height, width, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(1, 255, (height, width, 3), dtype=np.uint8)


class DocumentDecodingTests(unittest.TestCase):
    """Corner-heatmap decoding must be exact and fail closed."""

    def _service(self, quads):
        service = DocumentDetectionService()
        manager = FakeDocumentManager(FakeDocumentNet(quads))
        service.model_manager = manager
        return service, manager

    def test_single_document_is_decoded_to_its_box_and_confidence(self):
        quad = [(0.30, 0.20), (0.70, 0.22), (0.68, 0.62), (0.32, 0.60)]
        service, _ = self._service([quad])
        detections = service.detect(np.zeros((500, 800, 3), np.uint8))
        self.assertEqual(len(detections), 1)
        detection = detections[0]
        self.assertAlmostEqual(detection.confidence, 0.9, places=2)
        self.assertAlmostEqual(detection.bbox.x, 30.0, delta=1.5)
        self.assertAlmostEqual(detection.bbox.y, 20.0, delta=1.5)
        self.assertAlmostEqual(detection.bbox.width, 40.0, delta=3.5)
        self.assertAlmostEqual(detection.bbox.height, 40.0, delta=3.5)

    def test_multiple_documents_are_found_by_strip_scanning(self):
        # The full-frame pass sees the left document; the right strip sees the
        # second one. Two distinct boxes must survive deduplication.
        left = [(0.05, 0.20), (0.30, 0.21), (0.31, 0.60), (0.06, 0.59)]
        right_in_strip = [(0.35, 0.25), (0.75, 0.25), (0.75, 0.60), (0.35, 0.60)]
        service, _ = self._service([left, None, right_in_strip])
        detections = service.detect(np.zeros((600, 1000, 3), np.uint8))
        self.assertEqual(len(detections), 2)
        boxes = sorted(detections, key=lambda d: d.bbox.x)
        self.assertLess(boxes[0].bbox.x + boxes[0].bbox.width, boxes[1].bbox.x + 1e-6)

    def test_duplicate_strip_detection_is_merged(self):
        quad = [(0.30, 0.20), (0.70, 0.22), (0.68, 0.62), (0.32, 0.60)]
        service, _ = self._service([quad, quad, None, None, None])
        detections = service.detect(np.zeros((500, 800, 3), np.uint8))
        self.assertEqual(len(detections), 1)

    def test_missing_corner_is_reconstructed_from_three_corners(self):
        # Bottom-right corner is outside the frame; the parallelogram vertex is
        # recovered so a partially visible document is still covered.
        partial = [(0.30, 0.20), (0.70, 0.20), None, (0.30, 0.60)]
        service, _ = self._service([partial])
        detections = service.detect(np.zeros((500, 800, 3), np.uint8))
        self.assertEqual(len(detections), 1)
        box = detections[0].bbox
        self.assertGreater(box.width, 35.0)
        self.assertGreater(box.height, 35.0)

    def test_two_missing_corners_are_not_guessed(self):
        service, _ = self._service([[(0.30, 0.20), None, None, (0.30, 0.60)]])
        self.assertEqual(service.detect(np.zeros((500, 800, 3), np.uint8)), [])

    def test_low_corner_activation_is_rejected(self):
        service, _ = self._service([None])
        self.assertEqual(service.detect(np.zeros((500, 800, 3), np.uint8)), [])

    def test_implausibly_small_document_is_rejected(self):
        tiny = [(0.500, 0.500), (0.510, 0.500), (0.510, 0.510), (0.500, 0.510)]
        service, _ = self._service([tiny])
        self.assertEqual(service.detect(np.zeros((1080, 1920, 3), np.uint8)), [])

    def test_full_frame_hallucination_is_rejected(self):
        whole = [(0.01, 0.01), (0.99, 0.01), (0.99, 0.99), (0.01, 0.99)]
        service, _ = self._service([whole])
        self.assertEqual(service.detect(np.zeros((600, 600, 3), np.uint8)), [])

    def test_unavailable_detector_raises_a_clear_error(self):
        service = DocumentDetectionService()
        service.model_manager = FakeDocumentManager(
            net=None, status="UNAVAILABLE",
            detail="Identity-document detector weights are not available")
        with self.assertRaises(DetectorUnavailableException) as raised:
            service.detect(np.zeros((100, 100, 3), np.uint8))
        self.assertIn("Identity-document", str(raised.exception))

    def test_inference_failure_raises_instead_of_returning_empty(self):
        net = FakeDocumentNet([])
        net.raise_on_forward = True
        service = DocumentDetectionService()
        service.model_manager = FakeDocumentManager(net)
        with self.assertRaises(DetectorInferenceException):
            service.detect(np.zeros((100, 100, 3), np.uint8))

    def test_empty_frame_is_rejected(self):
        service = DocumentDetectionService()
        with self.assertRaises(DetectorInferenceException):
            service.detect(np.zeros((0, 0, 3), np.uint8))

    def test_detection_service_runs_one_document_model_for_both_categories(self):
        for categories in (["id_cards"], ["passports"], ["id_cards", "passports"]):
            with self.subTest(categories=categories):
                service = DetectionService()
                manager = FakeDocumentManager(FakeDocumentNet(
                    [[(0.2, 0.2), (0.6, 0.2), (0.6, 0.6), (0.2, 0.6)]]))
                # Replace the document service rather than mutating the shared
                # singleton, so a test cannot leak a fake net into other tests.
                service.document_service = DocumentDetectionService()
                service.document_service.model_manager = manager
                result = service.detect_entities(
                    np.zeros((400, 400, 3), np.uint8), categories)
                self.assertTrue(result.document_detector_used)
                self.assertEqual(result.total_detections, 1)
                self.assertEqual(result.detections[0].category, TargetCategory.ID_CARDS)
                self.assertEqual(result.detections[0].label, "DOCUMENT")
                self.assertEqual(manager.calls, 1)

    def test_document_detector_is_not_initialized_for_face_only_jobs(self):
        service = DetectionService()
        manager = FakeDocumentManager(FakeDocumentNet([]))
        service.model_manager = manager
        service.model_manager.get_model = lambda key: {
            "model": FakeFaceNet(), "status": "LOADED",
            "weights_path": "face.onnx", "detail": "ok"}
        service.detect_entities(np.zeros((100, 100, 3), np.uint8), ["faces"])
        self.assertEqual(manager.calls, 0)


@unittest.skipUnless(os.path.isdir(MIDV_DIR),
                     "cached MIDV validation frames are not available")
class CachedDocumentImageTests(unittest.TestCase):
    """Real detector, real document pixels: rotation and small/distant scale."""

    @staticmethod
    def _box_iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        iy = max(0.0, min(ay2, by2) - max(ay1, by1))
        intersection = ix * iy
        union = ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1)
                 - intersection)
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _place(document, scale, angle, width=1280, height=960):
        resized = cv2.resize(document, (int(document.shape[1] * scale),
                                        int(document.shape[0] * scale)))
        matrix = cv2.getRotationMatrix2D(
            (resized.shape[1] / 2, resized.shape[0] / 2), angle, 1.0)
        cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
        out_w = int(resized.shape[0] * sin + resized.shape[1] * cos)
        out_h = int(resized.shape[0] * cos + resized.shape[1] * sin)
        matrix[0, 2] += out_w / 2 - resized.shape[1] / 2
        matrix[1, 2] += out_h / 2 - resized.shape[0] / 2
        rotated = cv2.warpAffine(resized, matrix, (out_w, out_h),
                                 borderValue=(70, 70, 70))
        scene = np.full((height, width, 3), 60, np.uint8)
        x = (width - out_w) // 2
        y = (height - out_h) // 2
        scene[y:y + out_h, x:x + out_w] = rotated
        return scene, (x, y, x + out_w, y + out_h)

    def test_rotated_and_small_documents_are_localised(self):
        service = DocumentDetectionService()
        document = _cached_midv_frames(1)[0]
        for scale, angle in ((0.55, 30.0), (0.35, -25.0), (0.12, 0.0)):
            with self.subTest(scale=scale, angle=angle):
                frame, expected = self._place(document, scale, angle)
                detections = service.detect(frame)
                height, width = frame.shape[:2]
                best = 0.0
                for detection in detections:
                    box = detection.bbox
                    predicted = (box.x / 100 * width, box.y / 100 * height,
                                 (box.x + box.width) / 100 * width,
                                 (box.y + box.height) / 100 * height)
                    best = max(best, self._box_iou(predicted, expected))
                self.assertGreaterEqual(
                    best, 0.45,
                    f"scale={scale} angle={angle}: no detection covered the "
                    f"placed document (best bbox IoU {best:.3f})")

    def test_two_documents_in_one_frame_are_found_independently(self):
        service = DocumentDetectionService()
        first, second = _cached_midv_frames(2)
        width, height = 1280, 960
        scene = np.full((height, width, 3), 60, np.uint8)

        def fit(image, max_width, max_height):
            scale = min(max_width / image.shape[1], max_height / image.shape[0])
            return cv2.resize(image, (int(image.shape[1] * scale),
                                      int(image.shape[0] * scale)))

        left = fit(first, 512, 800)
        right = fit(second, 512, 800)
        left_x, left_y = 30, height - 30 - left.shape[0]
        right_x, right_y = width - 30 - right.shape[1], height - 30 - right.shape[0]
        scene[left_y:left_y + left.shape[0], left_x:left_x + left.shape[1]] = left
        scene[right_y:right_y + right.shape[0], right_x:right_x + right.shape[1]] = right
        expected = [
            (left_x, left_y, left_x + left.shape[1], left_y + left.shape[0]),
            (right_x, right_y, right_x + right.shape[1], right_y + right.shape[0]),
        ]

        detections = service.detect(scene)
        predicted = []
        for detection in detections:
            box = detection.bbox
            predicted.append((box.x / 100 * width, box.y / 100 * height,
                              (box.x + box.width) / 100 * width,
                              (box.y + box.height) / 100 * height))
        matched = sum(
            1 for target in expected
            if any(self._box_iou(box, target) >= 0.45 for box in predicted)
        )
        self.assertGreaterEqual(
            matched, 2,
            f"found {len(detections)} detection(s), but only {matched} of the 2 "
            "placed documents were covered")


class DocumentRedactionTests(unittest.TestCase):
    """Documents need a full rectangular cover, proportional padding, and a
    stronger mosaic than faces."""

    def setUp(self):
        self.svc = BlurService()

    def test_padding_is_proportional_and_uses_document_ratios(self):
        bbox = NormalizedBoundingBox(x=30.0, y=35.0, width=40.0, height=25.0)
        plan = self.svc._convert_normalized_bbox_to_roi(
            bbox, 2000, 1000, TargetCategory.ID_CARDS, False)
        core_w = math.ceil((bbox.x + bbox.width) / 100 * 2000) - math.floor(bbox.x / 100 * 2000)
        core_h = math.ceil((bbox.y + bbox.height) / 100 * 1000) - math.floor(bbox.y / 100 * 1000)
        expected_x = max(ai_settings.DOCUMENT_MIN_PADDING_PX,
                         min(ai_settings.DOCUMENT_MAX_PADDING_PX,
                             round(core_w * ai_settings.DOCUMENT_PADDING_X_RATIO)))
        expected_y = max(ai_settings.DOCUMENT_MIN_PADDING_PX,
                         min(ai_settings.DOCUMENT_MAX_PADDING_PX,
                             round(core_h * ai_settings.DOCUMENT_PADDING_Y_RATIO)))
        self.assertEqual(plan.core_x1 - plan.x1, expected_x)
        self.assertEqual(plan.x2 - plan.core_x2, expected_x)
        self.assertEqual(plan.core_y1 - plan.y1, expected_y)
        self.assertEqual(plan.y2 - plan.core_y2, expected_y)

    def test_whole_document_roi_is_filtered_in_every_mode(self):
        bbox = NormalizedBoundingBox(x=30.0, y=35.0, width=40.0, height=25.0)
        for mode in MODES:
            with self.subTest(mode=mode):
                frame = noise_frame(600, 1000, seed=3)
                original = frame.copy()
                out, report = self.svc.render_blur(
                    frame, [document_track(bbox)], [], blur_style=mode,
                    blur_intensity=80, frame_index=1)
                self.assertEqual(report.total_blurred_regions, 1)
                plan = report.roi_plans[0]
                roi = out[plan.y1:plan.y2, plan.x1:plan.x2]
                if mode == "blackout":
                    self.assertTrue(np.all(roi == 0))
                else:
                    changed = (roi != original[plan.y1:plan.y2,
                                             plan.x1:plan.x2]).any(axis=2)
                    self.assertGreaterEqual(float(changed.mean()), 0.95)
                if mode in ("gaussian", "blackout"):
                    # A document is a flat rectangle: its mask covers the
                    # padding corners too, so no fragment survives there.
                    # (Pixelation may coincidentally keep a block's own pixel.)
                    self.assertFalse(np.array_equal(
                        out[plan.y1, plan.x1], original[plan.y1, plan.x1]))

    def test_pixelation_uses_the_document_block_cap(self):
        bbox = NormalizedBoundingBox(x=20.0, y=20.0, width=50.0, height=35.0)
        frame = noise_frame(800, 1200, seed=5)
        out, report = self.svc.render_blur(
            frame, [document_track(bbox)], [], blur_style="pixelate",
            blur_intensity=80, frame_index=1)
        plan = report.roi_plans[0]
        roi = out[plan.y1:plan.y2, plan.x1:plan.x2]
        block = self.svc._pixelate_block(
            plan.x2 - plan.x1, plan.y2 - plan.y1, 80,
            max_blocks=ai_settings.DOCUMENT_PIXELATE_MAX_BLOCKS)
        small_w = max(1, (plan.x2 - plan.x1) // block)
        self.assertLessEqual(small_w, ai_settings.DOCUMENT_PIXELATE_MAX_BLOCKS + 1)
        self.assertFalse(np.array_equal(
            roi, frame[plan.y1:plan.y2, plan.x1:plan.x2]))

    def test_multiple_documents_are_redacted_independently(self):
        boxes = [
            NormalizedBoundingBox(x=5.0, y=30.0, width=30.0, height=20.0),
            NormalizedBoundingBox(x=60.0, y=30.0, width=30.0, height=20.0),
        ]
        frame = noise_frame(400, 1000, seed=9)
        out, report = self.svc.render_blur(
            frame, [document_track(b, i + 1) for i, b in enumerate(boxes)], [],
            blur_style="blackout", frame_index=1)
        self.assertEqual(report.total_blurred_regions, 2)
        plan_a, plan_b = sorted(report.roi_plans, key=lambda p: p.x1)
        gap = out[plan_a.y1:plan_a.y2, plan_a.x2:plan_b.x1]
        original_gap = frame[plan_a.y1:plan_a.y2, plan_a.x2:plan_b.x1]
        self.assertTrue(np.array_equal(gap, original_gap))

    def test_face_and_document_regions_are_not_merged(self):
        face = DetectionItem(
            category=TargetCategory.FACES, label="FACE", confidence=0.9,
            bbox=NormalizedBoundingBox(x=10.0, y=10.0, width=30.0, height=30.0))
        document = document_detection(12.0, 12.0, 28.0, 25.0)
        frame = noise_frame(300, 300, seed=11)
        _, report = self.svc.render_blur(
            frame, [], [], blur_style="blackout", frame_index=1,
            detections=[face, document])
        self.assertEqual(report.total_blurred_regions, 2)
        self.assertEqual({plan.category for plan in report.roi_plans},
                         {TargetCategory.FACES, TargetCategory.ID_CARDS})

    def test_mask_geometry_survives_rotation_like_wide_boxes(self):
        # A rotated document produces a large axis-aligned box; the rectangular
        # mask must cover every corner of that box (no exposed document corner).
        boxes = [
            NormalizedBoundingBox(x=10.0, y=10.0, width=70.0, height=70.0),
            NormalizedBoundingBox(x=55.0, y=20.0, width=30.0, height=60.0),
        ]
        frame = noise_frame(800, 800, seed=13)
        out, report = self.svc.render_blur(
            frame, [document_track(b, i + 1) for i, b in enumerate(boxes)], [],
            blur_style="blackout", frame_index=1)
        for plan in report.roi_plans:
            corners_are_black = all(np.all(out[y, x] == 0) for x, y in (
                (plan.x1, plan.y1), (plan.x2 - 1, plan.y1),
                (plan.x1, plan.y2 - 1), (plan.x2 - 1, plan.y2 - 1)))
            self.assertTrue(corners_are_black)


class DocumentTrackingTests(unittest.TestCase):
    def setUp(self):
        self.tracker = TrackingService()

    def test_document_detections_are_tracked_at_their_own_threshold(self):
        result = self.tracker.update_tracks([
            document_detection(20.0, 30.0, 30.0, 20.0, conf=0.31)])
        self.assertEqual(len(result.tracks), 1)
        self.assertEqual(result.tracks[0].category, TargetCategory.ID_CARDS)
        self.assertEqual(result.tracks[0].label, "DOC #1")

    def test_jittering_document_does_not_jump_between_frames(self):
        rng = np.random.default_rng(17)
        previous = None
        jumps = []
        for _ in range(20):
            detection = document_detection(
                40.0 + float(rng.normal(0, 0.15)), 30.0 + float(rng.normal(0, 0.15)),
                30.0, 20.0)
            track = self.tracker.update_tracks([detection]).tracks[0]
            current = (track.bbox.x, track.bbox.y)
            if previous is not None:
                jumps.append(math.dist(previous, current))
            previous = current
        self.assertLess(max(jumps), 1.0, "document box jittered between frames")

    def test_moving_document_track_covers_every_detection(self):
        for index in range(15):
            bbox = NormalizedBoundingBox(x=10.0 + index * 2.0, y=30.0,
                                         width=30.0, height=20.0)
            track = self.tracker.update_tracks([document_detection(
                bbox.x, bbox.y, bbox.width, bbox.height)]).tracks[0]
            self.assertLessEqual(track.bbox.x, bbox.x + 1e-6)
            self.assertGreaterEqual(track.bbox.x + track.bbox.width,
                                    bbox.x + bbox.width - 1e-6)

    def test_multiple_documents_get_independent_tracks(self):
        detections = [
            document_detection(5.0, 30.0, 30.0, 20.0),
            document_detection(60.0, 30.0, 30.0, 20.0),
        ]
        tracks = self.tracker.update_tracks(detections).tracks
        self.assertEqual(len(tracks), 2)
        self.assertEqual(len({track.track_id for track in tracks}), 2)

    def test_document_hold_is_short_and_does_not_move(self):
        for _ in range(3):
            self.tracker.update_tracks([document_detection(20.0, 30.0, 30.0, 20.0)])
        lost = self.tracker.update_tracks([]).tracks
        self.assertEqual(len(lost), 1)
        before = lost[0].bbox.model_dump()
        blur = BlurService()
        frame = noise_frame(480, 640, seed=15)
        _, report = blur.render_blur(frame, lost, [], blur_style="blackout", frame_index=1)
        self.assertEqual(len(report.roi_plans), 1)

        for _ in range(ai_settings.DOCUMENT_REDACTION_HOLD_FRAMES + 1):
            tracks = self.tracker.update_tracks([]).tracks
        held = [t for t in tracks if t.track_state == TrackStateEnum.LOST]
        if held:
            self.assertEqual(held[0].bbox.model_dump(), before)
        _, stale = blur.render_blur(frame, tracks, [], blur_style="blackout", frame_index=2)
        self.assertEqual(len(stale.roi_plans), 0,
                         "a stale document box kept masking after the hold window")


class DocumentApiTests(unittest.TestCase):
    def setUp(self):
        task_manager.tasks.clear()

    def test_request_accepts_id_cards_and_passports(self):
        request = StartRedactionRequest(
            video_id="vid_x", target_categories=["id_cards", "passports"])
        self.assertEqual({c.value for c in request.target_categories},
                         {"id_cards", "passports"})

    def test_face_only_request_stays_backward_compatible(self):
        request = StartRedactionRequest(video_id="vid_x")
        self.assertEqual([c.value for c in request.target_categories], ["faces"])

    def test_missing_document_weights_fail_before_a_task_is_created(self):
        original = model_manager.ensure_document_detector
        model_manager.ensure_document_detector = lambda: {
            "model": None, "status": "UNAVAILABLE",
            "weights_path": ai_settings.DOCUMENT_MODEL_PATH, "detail": "missing"}
        try:
            with self.assertRaises(DetectorUnavailableException) as raised:
                DetectionService().validate_available(["id_cards"])
            self.assertIn("Identity-document", str(raised.exception))
        finally:
            model_manager.ensure_document_detector = original

    def test_summary_reports_document_regions_together(self):
        task_manager.create_task("task_docs", "vid_x", "fixture.mp4",
                                 ["id_cards", "passports"], "blackout", 80)
        task_manager.get_task("task_docs").update(
            status="COMPLETED", output_verified=True,
            redaction_counts={"faces": 2, "id_cards": 7},
            metrics={"processed_frames": 10, "frames_with_detections": 8,
                     "elapsed_seconds": 4},
            verification={"frame_count": 10, "width": 640, "height": 480})
        summary = RedactService().get_task_summary("task_docs")
        self.assertEqual(summary.documents_redacted, 7)
        # No per-type count is claimed by a class-agnostic detector.
        self.assertEqual(summary.id_cards_found, 0)
        self.assertEqual(summary.faces_blurred, 2)
        self.assertEqual(summary.total_objects_count, 9)

    def test_combined_categories_run_face_plate_and_document_detectors(self):
        service = DetectionService()
        manager = FakeDocumentManager(FakeDocumentNet(
            [[(0.2, 0.2), (0.6, 0.2), (0.6, 0.6), (0.2, 0.6)]]))
        service.model_manager = manager
        service.model_manager.get_model = lambda key: {
            "model": FakeFaceNet(), "status": "LOADED",
            "weights_path": "face.onnx", "detail": "ok"}
        service.document_service = DocumentDetectionService()
        service.document_service.model_manager = manager
        service.model_manager.ensure_plate_detector = lambda: {
            "model": FakePlateNet(), "status": "LOADED",
            "weights_path": "plate.onnx", "detail": "ok"}
        result = service.detect_entities(
            np.zeros((400, 400, 3), np.uint8), ["faces", "license_plates", "passports"])
        categories = {d.category for d in result.detections}
        self.assertEqual(categories, {TargetCategory.FACES,
                                      TargetCategory.LICENSE_PLATES,
                                      TargetCategory.ID_CARDS})
        self.assertTrue(result.secondary_detector_used)
        self.assertTrue(result.plate_detector_used)
        self.assertTrue(result.document_detector_used)
        self.assertGreaterEqual(manager.calls, 1)


def _cached_midv_frames(limit=3):
    """Returns document crops cut from the cached MIDV validation frames.

    Cropping to the annotated mask keeps the document large in the generated
    clip, which is what a document-centric test clip should exercise.
    """
    if not os.path.isdir(MIDV_DIR):
        return []
    names = sorted(name for name in os.listdir(MIDV_DIR) if name.endswith("_img.jpg"))
    frames = []
    for name in names:
        if len(frames) >= limit:
            break
        image = cv2.imread(os.path.join(MIDV_DIR, name))
        mask = cv2.imread(os.path.join(MIDV_DIR, name.replace("_img", "_mask")),
                          cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue
        ys, xs = np.where(mask > 127)
        if len(xs) == 0:
            continue
        margin_x = int((xs.max() - xs.min()) * 0.06)
        margin_y = int((ys.max() - ys.min()) * 0.06)
        x1 = max(0, int(xs.min()) - margin_x)
        y1 = max(0, int(ys.min()) - margin_y)
        x2 = min(image.shape[1], int(xs.max()) + margin_x)
        y2 = min(image.shape[0], int(ys.max()) + margin_y)
        frames.append(image[y1:y2, x1:x2])
    return frames


def _write_document_clip(path, frames, count=12, width=960, height=540):
    """Builds a real-photo clip from the cached MIDV frames."""
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10,
                             (width, height))
    if not writer.isOpened():
        return False
    try:
        for index in range(count):
            canvas = np.full((height, width, 3), 48, np.uint8)
            source = frames[index % len(frames)]
            scale = min((width * 0.7) / source.shape[1],
                        (height * 0.7) / source.shape[0])
            resized = cv2.resize(source, None, fx=scale, fy=scale)
            y0 = (height - resized.shape[0]) // 2
            x0 = (width - resized.shape[1]) // 2
            canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
            writer.write(canvas)
    finally:
        writer.release()
    return True


@unittest.skipUnless(os.path.isfile(model_manager.resolve_document_path()),
                     "NOT RUN: identity-document ONNX weights are missing "
                     "(run scripts/download_document_model.py)")
class DocumentPipelineIntegrationTests(unittest.TestCase):
    """Real detector + FFmpeg on a real-photo clip."""

    def setUp(self):
        task_manager.tasks.clear()
        if not ffmpeg_util.validate_ffmpeg().get("available"):
            self.skipTest("NOT RUN: FFmpeg is not available")
        frames = _cached_midv_frames(3)
        if len(frames) < 2:
            self.skipTest("NOT RUN: cached MIDV validation frames are unavailable")
        self.temp_dir = tempfile.mkdtemp(prefix="safeframe_documents_")
        raw = os.path.join(self.temp_dir, "raw.mp4")
        if not _write_document_clip(raw, frames):
            self.skipTest("NOT RUN: could not build the document clip")
        self.clip = os.path.join(self.temp_dir, "document_clip.mp4")
        ffmpeg = ffmpeg_util.resolve_ffmpeg_path()
        build = subprocess.run(
            [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", raw,
             "-f", "lavfi", "-i", "sine=frequency=440:duration=1.2",
             "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", self.clip],
            capture_output=True, timeout=120)
        if build.returncode != 0:
            self.skipTest("NOT RUN: could not encode the document clip")
        self.video_id = f"vid_test_docs_{uuid.uuid4().hex[:8]}"
        self.staged = os.path.join(storage_manager.base_dir, "raw_videos",
                                   f"{self.video_id}_document_clip.mp4")
        with open(self.clip, "rb") as source, open(self.staged, "wb") as target:
            target.write(source.read())

    def tearDown(self):
        try:
            os.remove(self.staged)
        except OSError:
            pass
        for task_id in getattr(self, "task_ids", []):
            try:
                os.remove(os.path.join(storage_manager.base_dir, "redacted_videos",
                                       f"{task_id}_protected.mp4"))
            except OSError:
                pass
        for name in os.listdir(self.temp_dir):
            try:
                os.remove(os.path.join(self.temp_dir, name))
            except OSError:
                pass
        try:
            os.rmdir(self.temp_dir)
        except OSError:
            pass

    def _run(self, categories, style):
        task_id = f"task_test_docs_{style}_{uuid.uuid4().hex[:8]}"
        self.task_ids = getattr(self, "task_ids", []) + [task_id]
        task_manager.create_task(task_id, self.video_id, "document_clip.mp4",
                                 categories, style, 80)
        output = os.path.join(storage_manager.base_dir, "redacted_videos",
                              f"{task_id}_protected.mp4")
        asyncio.run(video_processor_service.process_video_pipeline(
            task_id, self.video_id, categories, style, 80))
        return task_manager.get_task(task_id), output

    def test_id_card_pipeline_produces_a_verified_output(self):
        task, _ = self._run(["id_cards"], "gaussian")
        self.assertEqual(task["status"], "COMPLETED", task.get("failure_reason"))
        self.assertTrue(task["output_verified"])
        self.assertEqual(task["metrics"]["output_frame_count"], 12)
        self.assertGreater(task["metrics"]["masks_total"], 0)
        self.assertGreater(task["redaction_counts"].get("id_cards", 0), 0)
        self.assertEqual(task["redaction_counts"].get("faces", 0), 0)

    def test_passport_category_runs_the_document_detector(self):
        task, _ = self._run(["passports"], "blackout")
        self.assertEqual(task["status"], "COMPLETED", task.get("failure_reason"))
        self.assertGreater(task["redaction_counts"].get("id_cards", 0), 0)

    def test_all_three_modes_redact_documents_and_verify(self):
        for style in MODES:
            with self.subTest(style=style):
                task, _ = self._run(["id_cards"], style)
                self.assertEqual(task["status"], "COMPLETED", task.get("failure_reason"))
                self.assertTrue(task["output_verified"])
                self.assertGreater(task["metrics"]["masks_total"], 0)

    def test_combined_face_plate_and_document_processing(self):
        task, _ = self._run(["faces", "license_plates", "passports"], "blackout")
        self.assertEqual(task["status"], "COMPLETED", task.get("failure_reason"))
        self.assertTrue(task["output_verified"])
        self.assertGreater(task["redaction_counts"].get("id_cards", 0), 0,
                           "the document detector stopped working in a combined job")
        self.assertGreater(task["metrics"]["new_tracks_total"], 0)

    def test_face_only_regression_still_completes_without_documents(self):
        task, _ = self._run(["faces"], "gaussian")
        self.assertEqual(task["status"], "COMPLETED", task.get("failure_reason"))
        self.assertTrue(task["output_verified"])
        self.assertEqual(task["redaction_counts"].get("id_cards", 0), 0)

    def test_audio_is_preserved_and_resolution_matches(self):
        task, _ = self._run(["id_cards"], "pixelate")
        self.assertEqual(task["status"], "COMPLETED", task.get("failure_reason"))
        self.assertGreaterEqual(task["metrics"]["input_audio_streams"] or 0, 1)
        self.assertTrue(task["metrics"]["audio_preserved"])
        self.assertEqual((task["metrics"]["output_width"], task["metrics"]["output_height"]),
                         (960, 540))

    @staticmethod
    def _iou(a, b):
        ax1, ay1 = a.x, a.y
        ax2, ay2 = a.x + a.width, a.y + a.height
        bx1, by1 = b.x, b.y
        bx2, by2 = b.x + b.width, b.y + b.height
        ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        iy = max(0.0, min(ay2, by2) - max(ay1, by1))
        intersection = ix * iy
        union = a.width * a.height + b.width * b.height - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _content_correlation(original, protected):
        """Normalized correlation of the original and protected document crops.

        A redacted document may still look like a document-shaped rectangle (a
        mosaic block or a grey patch), so detection alone is not a content
        leak. A survivor is a region that is still *recognizable*: its protected
        pixels correlate with the original document content.
        """
        a = cv2.resize(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), (64, 64)).astype(np.float32)
        b = cv2.resize(cv2.cvtColor(protected, cv2.COLOR_BGR2GRAY), (64, 64)).astype(np.float32)
        a = (a - a.mean()) / (a.std() + 1e-6)
        b = (b - b.mean()) / (b.std() + 1e-6)
        return float((a * b).mean())

    def test_document_regions_are_actually_redacted_in_every_mode(self):
        service = DocumentDetectionService()
        for style in MODES:
            with self.subTest(style=style):
                task, output = self._run(["id_cards"], style)
                self.assertEqual(task["status"], "COMPLETED", task.get("failure_reason"))

                input_cap = cv2.VideoCapture(self.clip)
                output_cap = cv2.VideoCapture(output)
                survivors = 0
                documents_seen = 0
                frames = 0
                correlations = []
                try:
                    while True:
                        ok_in, frame_in = input_cap.read()
                        ok_out, frame_out = output_cap.read()
                        if not ok_in or not ok_out:
                            break
                        frames += 1
                        original = service.detect(frame_in, frames)
                        protected = service.detect(frame_out, frames)
                        height, width = frame_in.shape[:2]
                        for document in original:
                            documents_seen += 1
                            for other in protected:
                                if self._iou(document.bbox, other.bbox) < 0.3:
                                    continue
                                x1 = int(min(document.bbox.x, other.bbox.x) / 100 * width)
                                y1 = int(min(document.bbox.y, other.bbox.y) / 100 * height)
                                x2 = int(max(document.bbox.x + document.bbox.width,
                                             other.bbox.x + other.bbox.width) / 100 * width)
                                y2 = int(max(document.bbox.y + document.bbox.height,
                                             other.bbox.y + other.bbox.height) / 100 * height)
                                x1, y1 = max(0, x1), max(0, y1)
                                x2, y2 = min(width, x2), min(height, y2)
                                if x2 <= x1 or y2 <= y1:
                                    continue
                                correlation = self._content_correlation(
                                    frame_in[y1:y2, x1:x2], frame_out[y1:y2, x1:x2])
                                correlations.append(round(correlation, 3))
                                if correlation >= 0.5:
                                    survivors += 1
                finally:
                    input_cap.release()
                    output_cap.release()
                self.assertGreater(frames, 0)
                self.assertGreater(documents_seen, 0,
                                   "the clip contains no detectable document to redact")
                self.assertEqual(
                    survivors, 0,
                    f"{style}: {survivors} of {documents_seen} document instances were "
                    f"still recognizable in the protected output (correlations "
                    f"{correlations[:6]})")


if __name__ == "__main__":
    unittest.main()
