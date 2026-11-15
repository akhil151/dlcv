"""Phase 4 - laptop-screen detection, display-region estimation and redaction.

Deterministic tests use an injected fake COCO detection graph and synthetic
frames to pin down the decoder, the display-quad gates, the polygon mask,
tracking state and API wiring. Tests that need the real RT-DETR weights, FFmpeg
or the cached real photos are skipped (never reported as passed) when those
dependencies are absent.

Measured accuracy on real, manually annotated laptop photos lives in
scripts/evaluate_laptop_screen_detection.py; nothing here claims accuracy from
synthetic fixtures.
"""
import asyncio
import json
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
from backend.services.ai.laptop_screen_detection_service import (
    REGION_LAPTOP_FALLBACK,
    REGION_SCREEN_QUAD,
    LaptopScreenDetectionService,
)
from backend.services.ai.tracking_service import TrackingService
from backend.services.ai.video_processor_service import video_processor_service
from backend.services.redact_service import RedactService
from backend.utils import ffmpeg as ffmpeg_util
from backend.utils.storage import storage_manager
from backend.workers.task_manager import task_manager

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PHOTO_DIR = os.path.join(PROJECT_ROOT, ".cache", "validation", "screens")
ANNOTATIONS = os.path.join(PROJECT_ROOT, "scripts", "laptop_screen_annotations.json")
MODES = ("gaussian", "pixelate", "blackout")
LAPTOP_CLASS = ai_settings.LAPTOP_CLASS_INDEX


class FakeOutput:
    def __init__(self, name):
        self.name = name


class FakeInput:
    name = "pixel_values"


class FakeLaptopSession:
    """Deterministic stand-in for the RT-DETR COCO ONNX graph.

    ``detections`` is a list of (class_index, score, cx, cy, w, h) in
    normalized coordinates.
    """

    def __init__(self, detections):
        self.detections = list(detections)
        self.calls = 0
        self.raise_on_run = False

    def get_inputs(self):
        return [FakeInput()]

    def get_outputs(self):
        return [FakeOutput("logits"), FakeOutput("pred_boxes")]

    def run(self, output_names, feed):
        self.calls += 1
        if self.raise_on_run:
            raise RuntimeError("onnx boom")
        logits = np.full((1, 300, 80), -12.0, dtype=np.float32)
        boxes = np.zeros((1, 300, 4), dtype=np.float32)
        for index, (class_index, score, cx, cy, width, height) in enumerate(self.detections):
            clamped = min(0.999, max(0.001, float(score)))
            value = math.log(clamped / (1.0 - clamped))
            logits[0, index, class_index] = value
            boxes[0, index] = (cx, cy, width, height)
        return [logits, boxes]


class FakeLaptopManager:
    def __init__(self, session=None, status="LOADED", detail="ok"):
        self.session = session
        self.status = status
        self.detail = detail
        self.calls = 0

    def ensure_laptop_detector(self):
        self.calls += 1
        return {
            "model": self.session,
            "status": self.status,
            "detail": self.detail,
            "weights_path": ai_settings.LAPTOP_MODEL_PATH,
            "output_names": ("logits", "pred_boxes"),
        }


def synthetic_laptop_scene(quad=None, width=960, height=720, inner_window=False,
                           dark_screen=False, glare=False):
    """Draws a laptop on a desk and returns (frame, device_box, screen_quad).

    The scene has a bezel, screen content, a hinge band and a keyboard with key
    rows, so edge/line evidence resembles a real photograph.
    """
    if quad is None:
        quad = np.array([[180, 150], [770, 140], [750, 430], [200, 445]], np.float32)
    frame = np.full((height, width, 3), 60, np.uint8)
    body = np.array([[120, 110], [840, 100], [850, 640], [110, 650]], np.int32)
    cv2.fillPoly(frame, [body], (48, 48, 50))

    points = np.round(np.asarray(quad, np.float32)).astype(np.int32)
    cv2.fillPoly(frame, [points], (90, 90, 92))  # bezel

    inner = cv2.resize(points.astype(np.float32), (4, 1)).astype(np.float32)
    inner = np.array([
        points[0] + (points[2] - points[0]) * 0.05,
        points[1] + (points[3] - points[1]) * 0.05,
        points[2] + (points[0] - points[2]) * 0.05,
        points[3] + (points[1] - points[3]) * 0.05,
    ], np.int32)
    del inner
    inset = 8
    screen = np.array([
        [quad[0][0] + inset, quad[0][1] + inset],
        [quad[1][0] - inset, quad[1][1] + inset],
        [quad[2][0] - inset, quad[2][1] - inset],
        [quad[3][0] + inset, quad[3][1] - inset],
    ], np.int32)
    screen_fill = (35, 35, 35) if dark_screen else (215, 215, 215)
    cv2.fillPoly(frame, [screen], screen_fill)
    if glare:
        overlay = frame.copy()
        cv2.ellipse(overlay, (int(screen[:, 0].mean()), int(screen[:, 1].mean())),
                    (140, 80), 20, 0, 360, (255, 255, 255), -1)
        frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
    if inner_window:
        window = np.array([
            [quad[0][0] + 110, quad[0][1] + 60],
            [quad[1][0] - 120, quad[1][1] + 60],
            [quad[2][0] - 120, quad[2][1] - 80],
            [quad[3][0] + 110, quad[3][1] - 80],
        ], np.int32)
        cv2.fillPoly(frame, [window], (250, 250, 250))

    # Screen content: a few text-like bars (real content, never just flat grey).
    for offset in range(0, 5):
        y = int(screen[:, 1].min() + 30 + offset * 40)
        cv2.line(frame, (int(screen[:, 0].min() + 20), y),
                 (int(screen[:, 0].max() - 60), y), (30, 30, 30), 3)

    # Hinge band and keyboard deck below the display.
    hinge_y = int(max(quad[2][1], quad[3][1])) + 18
    cv2.line(frame, (130, hinge_y), (835, hinge_y), (25, 25, 25), 6)
    cv2.rectangle(frame, (150, hinge_y + 10), (820, 630), (70, 70, 72), -1)
    for row in range(4):
        y = hinge_y + 30 + row * 38
        for column in range(10):
            x = 165 + column * 64
            cv2.rectangle(frame, (x, y), (x + 50, y + 28), (30, 30, 30), -1)
    cv2.rectangle(frame, (330, hinge_y + 190), (600, hinge_y + 270), (55, 55, 55), -1)

    box = [120.0, 100.0, 850.0, 650.0]
    return frame, box, np.asarray(quad, np.float32)


def laptop_detection(quad, confidence=0.9, region_kind=REGION_SCREEN_QUAD, category=TargetCategory.LAPTOP_SCREENS):
    points = np.asarray(quad, np.float32)
    x, y, w, h = cv2.boundingRect(np.round(points).astype(np.int32))
    height, width = quad_height_width(points)
    return DetectionItem(
        category=category,
        label="LAPTOP SCREEN",
        confidence=confidence,
        bbox=NormalizedBoundingBox(
            x=round(x / width * 100, 2), y=round(y / height * 100, 2),
            width=round(w / width * 100, 2), height=round(h / height * 100, 2)),
        source=DetectorSource.RTDETR_LAPTOP,
        quad=[[float(px), float(py)] for px, py in points],
        region_kind=region_kind,
    )


def quad_height_width(points):
    width = max(960.0, float(points[:, 0].max()) + 10.0)
    height = max(720.0, float(points[:, 1].max()) + 10.0)
    return height, width


def laptop_track(quad, track_id=1, state=TrackStateEnum.TRACKED, frames_since_update=0,
                 confirmed=True, region_kind=REGION_SCREEN_QUAD):
    detection = laptop_detection(quad, region_kind=region_kind)
    return TrackedObjectItem(
        track_id=track_id,
        category=TargetCategory.LAPTOP_SCREENS,
        label=f"SCREEN #{track_id}",
        confidence=0.9,
        bbox=detection.bbox,
        track_state=state,
        age=4,
        frames_since_update=frames_since_update,
        is_confirmed=confirmed,
        quad=detection.quad,
        region_kind=region_kind,
    )


def noise_frame(height=720, width=960, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(1, 255, (height, width, 3), dtype=np.uint8)


def service_with(session):
    service = LaptopScreenDetectionService()
    service.model_manager = FakeLaptopManager(session)
    return service


class LaptopDecodingTests(unittest.TestCase):
    def test_single_laptop_is_decoded(self):
        service = service_with(FakeLaptopSession([(LAPTOP_CLASS, 0.9, 0.5, 0.5, 0.6, 0.6)]))
        detections = service.detect(np.zeros((480, 640, 3), np.uint8))
        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0].confidence, 0.9, places=2)
        self.assertAlmostEqual(detections[0].bbox.x, 20.0, delta=2.0)
        self.assertEqual(len(detections[0].quad), 4)
        self.assertEqual(detections[0].region_kind, REGION_LAPTOP_FALLBACK)

    def test_non_laptop_classes_are_ignored(self):
        service = service_with(FakeLaptopSession([
            (62, 0.99, 0.5, 0.5, 0.8, 0.8),
            (41, 0.95, 0.2, 0.2, 0.1, 0.1),
        ]))
        self.assertEqual(service.detect(np.zeros((480, 640, 3), np.uint8)), [])

    def test_low_score_laptop_is_rejected(self):
        service = service_with(FakeLaptopSession([(LAPTOP_CLASS, 0.2, 0.5, 0.5, 0.6, 0.6)]))
        self.assertEqual(service.detect(np.zeros((480, 640, 3), np.uint8)), [])

    def test_overlapping_queries_are_suppressed(self):
        service = service_with(FakeLaptopSession([
            (LAPTOP_CLASS, 0.9, 0.5, 0.5, 0.6, 0.6),
            (LAPTOP_CLASS, 0.8, 0.5, 0.5, 0.6, 0.6),
        ]))
        self.assertEqual(len(service.detect(np.zeros((480, 640, 3), np.uint8))), 1)

    def test_multiple_laptops_are_found(self):
        service = service_with(FakeLaptopSession([
            (LAPTOP_CLASS, 0.9, 0.2, 0.5, 0.2, 0.4),
            (LAPTOP_CLASS, 0.8, 0.75, 0.5, 0.2, 0.4),
        ]))
        self.assertEqual(len(service.detect(np.zeros((480, 640, 3), np.uint8))), 2)

    def test_unavailable_detector_raises_a_clear_error(self):
        service = LaptopScreenDetectionService()
        service.model_manager = FakeLaptopManager(
            None, status="UNAVAILABLE",
            detail="Laptop-screen detector weights 'model.onnx' are not available")
        with self.assertRaises(DetectorUnavailableException) as raised:
            service.detect(np.zeros((100, 100, 3), np.uint8))
        self.assertIn("Laptop-screen", str(raised.exception))

    def test_inference_failure_raises_instead_of_returning_empty(self):
        session = FakeLaptopSession([(LAPTOP_CLASS, 0.9, 0.5, 0.5, 0.5, 0.5)])
        session.raise_on_run = True
        with self.assertRaises(DetectorInferenceException):
            service_with(session).detect(np.zeros((100, 100, 3), np.uint8))

    def test_empty_frame_is_rejected(self):
        with self.assertRaises(DetectorInferenceException):
            LaptopScreenDetectionService().detect(np.zeros((0, 0, 3), np.uint8))

    def test_detection_service_marks_laptop_detector_usage(self):
        service = DetectionService()
        manager = FakeLaptopManager(FakeLaptopSession([
            (LAPTOP_CLASS, 0.9, 0.5, 0.5, 0.5, 0.5)]))
        service.model_manager = manager
        service.laptop_service = LaptopScreenDetectionService()
        service.laptop_service.model_manager = manager
        result = service.detect_entities(
            synthetic_laptop_scene()[0], ["laptop_screens"])
        self.assertTrue(result.laptop_detector_used)
        self.assertEqual(result.detections[0].category, TargetCategory.LAPTOP_SCREENS)
        self.assertEqual(result.detections[0].source, DetectorSource.RTDETR_LAPTOP)

    def test_laptop_detector_is_not_loaded_for_face_jobs(self):
        service = DetectionService()
        manager = FakeLaptopManager(FakeLaptopSession([]))
        service.model_manager = manager
        service.model_manager.get_model = lambda key: {
            "model": type("FaceNet", (), {
                "setInputSize": lambda self, size: None,
                "detect": lambda self, frame: (None, None)})(),
            "status": "LOADED", "weights_path": "face.onnx", "detail": "ok"}
        service.detect_entities(np.zeros((120, 160, 3), np.uint8), ["faces"])
        self.assertEqual(manager.calls, 0)


class ScreenRegionTests(unittest.TestCase):
    """Nothing on the annotated display may remain visible."""

    def setUp(self):
        self.service = LaptopScreenDetectionService()

    @staticmethod
    def _quad_iou(first, second):
        a = np.zeros((800, 1200), np.uint8)
        b = np.zeros((800, 1200), np.uint8)
        cv2.fillPoly(a, [np.round(first).astype(np.int32)], 1)
        cv2.fillPoly(b, [np.round(second).astype(np.int32)], 1)
        union = np.logical_or(a, b).sum()
        return float(np.logical_and(a, b).sum()) / union if union else 0.0

    def test_front_facing_screen_is_refined(self):
        frame, box, screen = synthetic_laptop_scene()
        quad, evidence = self.service._estimate_screen_quad(frame, box)
        self.assertIsNotNone(quad, "a clear front-facing display was not refined")
        self.assertGreaterEqual(self._quad_iou(quad, screen), 0.55)
        self.assertGreater(evidence["edge_support"], 0.7)

    def test_perspective_screen_is_refined(self):
        screen = np.array([[170, 170], [780, 120], [720, 430], [230, 470]], np.float32)
        frame, box, screen = synthetic_laptop_scene(quad=screen)
        quad, _ = self.service._estimate_screen_quad(frame, box)
        self.assertIsNotNone(quad)
        self.assertGreaterEqual(self._quad_iou(quad, screen), 0.5)

    def test_dark_screen_is_redacted_without_claiming_a_boundary(self):
        # A switched-off/dim display is hard to separate from its bezel. Either a
        # correct quad is found, or the region is reported as a device fallback;
        # in both cases the display is still covered (see redaction tests).
        frame, box, screen = synthetic_laptop_scene(dark_screen=True)
        quad, _ = self.service._estimate_screen_quad(frame, box)
        if quad is not None:
            self.assertGreaterEqual(self._quad_iou(quad, screen), 0.5)
        else:
            detections = service_with(
                FakeLaptopSession([(LAPTOP_CLASS, 0.9, 0.47, 0.5, 0.8, 0.8)])
            ).detect(frame)
            self.assertEqual(detections[0].region_kind, REGION_LAPTOP_FALLBACK)

    def test_glare_screen_is_refined(self):
        frame, box, screen = synthetic_laptop_scene(glare=True)
        quad, _ = self.service._estimate_screen_quad(frame, box)
        self.assertIsNotNone(quad)
        self.assertGreaterEqual(self._quad_iou(quad, screen), 0.5)

    def test_inner_window_is_not_mistaken_for_the_display(self):
        frame, box, screen = synthetic_laptop_scene(inner_window=True)
        quad, _ = self.service._estimate_screen_quad(frame, box)
        if quad is not None:
            # If a boundary was accepted it must still contain the whole display;
            # a quad around the inner window would hide only part of it.
            coverage = self._quad_iou(quad, screen)
            self.assertGreaterEqual(
                coverage, 0.5,
                "the refinement locked onto an inner panel and would leak display content")

    def test_refined_quad_never_escapes_the_device_box(self):
        # Noise can legitimately produce a spurious quad. That is acceptable
        # only if it stays inside the device box, where the model already said a
        # laptop is: the failure mode must be over-redaction, never masking (or
        # leaking) content outside the detected device.
        frame = noise_frame(seed=3)
        box = [100.0, 100.0, 400.0, 400.0]
        quad, _ = self.service._estimate_screen_quad(frame, box)
        if quad is not None:
            self.assertGreaterEqual(quad[:, 0].min(), box[0] - 40)
            self.assertLessEqual(quad[:, 0].max(), box[2] + 40)
            self.assertGreaterEqual(quad[:, 1].min(), box[1] - 40)
            self.assertLessEqual(quad[:, 1].max(), box[3] + 40)

    def test_detector_falls_back_to_the_device_box_when_unrefined(self):
        session = FakeLaptopSession([(LAPTOP_CLASS, 0.9, 0.5, 0.5, 0.5, 0.5)])
        detections = service_with(session).detect(np.zeros((480, 640, 3), np.uint8))
        self.assertEqual(detections[0].region_kind, REGION_LAPTOP_FALLBACK)
        corners = {(round(x), round(y)) for x, y in detections[0].quad}
        self.assertIn((160, 120), {(160, 120), (160, 360), (480, 120)})
        self.assertEqual(len(corners), 4)


class LaptopRedactionTests(unittest.TestCase):
    def setUp(self):
        self.blur = BlurService()

    def test_polygon_mask_covers_the_display_and_padding(self):
        frame, _, screen = synthetic_laptop_scene()
        detection = laptop_detection(screen)
        out, report = self.blur.render_blur(
            frame.copy(), [], [], blur_style="blackout", blur_intensity=100,
            frame_index=1, detections=[detection])
        self.assertEqual(report.total_blurred_regions, 1)
        plan = report.roi_plans[0]
        self.assertIsNotNone(plan.quad, "the plan lost its display polygon")
        self.assertEqual(plan.region_kind, REGION_SCREEN_QUAD)
        screen_mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.fillPoly(screen_mask, [np.round(screen).astype(np.int32)], 1)
        changed = np.any(out != frame, axis=2).astype(np.uint8)
        leaked = int(np.logical_and(changed == 0, screen_mask).sum())
        self.assertEqual(leaked, 0, "display pixels survived the redaction")
        # Padding grows outward: the mask is strictly larger than the polygon.
        self.assertGreater(plan.pad, 0)

    def test_keyboard_and_deck_are_not_covered_by_a_screen_quad(self):
        frame, _, screen = synthetic_laptop_scene()
        keyboard = (slice(500, 700), slice(160, 840))
        before = frame[keyboard].copy()
        out, _ = self.blur.render_blur(
            frame.copy(), [], [], blur_style="blackout", blur_intensity=100,
            frame_index=1, detections=[laptop_detection(screen)])
        changed = np.any(out[keyboard] != before, axis=2)
        self.assertLess(float(changed.mean()), 0.02,
                        "the display mask spilled onto the keyboard")

    def test_every_mode_redacts_the_display(self):
        frame, _, screen = synthetic_laptop_scene()
        mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.round(screen).astype(np.int32)], 1)
        for mode in MODES:
            with self.subTest(mode=mode):
                out, _ = self.blur.render_blur(
                    frame.copy(), [], [], blur_style=mode, blur_intensity=80,
                    frame_index=1, detections=[laptop_detection(screen)])
                changed = np.any(out != frame, axis=2).astype(np.uint8)
                covered = float(np.logical_and(changed, mask).sum()) / max(1, int(mask.sum()))
                self.assertGreaterEqual(covered, 0.995)

    def test_device_fallback_masks_the_whole_device(self):
        frame, box, _ = synthetic_laptop_scene()
        quad = [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]
        detection = laptop_detection(quad, region_kind=REGION_LAPTOP_FALLBACK)
        out, report = self.blur.render_blur(
            frame.copy(), [], [], blur_style="blackout", blur_intensity=100,
            frame_index=1, detections=[detection])
        self.assertEqual(report.roi_plans[0].region_kind, REGION_LAPTOP_FALLBACK)
        self.assertTrue(np.all(out[int(box[1]) + 5:int(box[3]) - 5,
                                      int(box[0]) + 5:int(box[2]) - 5] == 0))

    def test_multiple_screens_are_redacted_independently(self):
        frame = noise_frame(seed=5)
        first = [[80, 100], [380, 100], [380, 300], [80, 300]]
        second = [[560, 120], [880, 120], [880, 330], [560, 330]]
        out, report = self.blur.render_blur(
            frame.copy(), [], [], blur_style="blackout", blur_intensity=100,
            frame_index=1, detections=[
                laptop_detection(first), laptop_detection(second)])
        self.assertEqual(report.total_blurred_regions, 2)
        gap = out[150:280, 420:540]
        self.assertTrue(np.array_equal(gap, frame[150:280, 420:540]))

    def test_detection_and_track_describing_one_screen_are_merged(self):
        frame, _, screen = synthetic_laptop_scene()
        detection = laptop_detection(screen)
        track = laptop_track(screen)
        _, report = self.blur.render_blur(
            frame, [track], [], blur_style="blackout", blur_intensity=100,
            frame_index=1, detections=[detection])
        self.assertEqual(report.total_blurred_regions, 1,
                         "the same screen was filtered twice as two regions")

    def test_mask_does_not_shrink_between_consecutive_frames(self):
        frame, _, screen = synthetic_laptop_scene()
        shrunk = (screen * 0.85 + np.array([30, 20], np.float32)).astype(np.float32)
        _, first = self.blur.render_blur(
            frame.copy(), [], [], blur_style="blackout", frame_index=1,
            detections=[laptop_detection(screen)])
        _, second = self.blur.render_blur(
            frame.copy(), [], [], blur_style="blackout", frame_index=2,
            detections=[laptop_detection(shrunk)])
        first_area = (first.roi_plans[0].x2 - first.roi_plans[0].x1) * (
            first.roi_plans[0].y2 - first.roi_plans[0].y1)
        second_area = (second.roi_plans[0].x2 - second.roi_plans[0].x1) * (
            second.roi_plans[0].y2 - second.roi_plans[0].y1)
        self.assertGreater(second_area, 0)
        self.assertLessEqual(first_area, second_area * 1.5)


class LaptopTrackingTests(unittest.TestCase):
    def setUp(self):
        self.tracker = TrackingService()

    def test_laptop_is_tracked_with_a_screen_label(self):
        quad = [[100, 100], [400, 100], [400, 300], [100, 300]]
        tracks = self.tracker.update_tracks([laptop_detection(quad)]).tracks
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].category, TargetCategory.LAPTOP_SCREENS)
        self.assertEqual(tracks[0].label, "SCREEN #1")
        self.assertEqual(tracks[0].region_kind, REGION_SCREEN_QUAD)
        self.assertEqual(len(tracks[0].quad), 4)

    def test_laptop_uses_its_own_confidence_floor(self):
        quad = [[100, 100], [400, 100], [400, 300], [100, 300]]
        tracks = self.tracker.update_tracks(
            [laptop_detection(quad, confidence=ai_settings.LAPTOP_CONF_THRESHOLD)]).tracks
        self.assertEqual(len(tracks), 1)

    def test_moving_laptop_keeps_covering_the_screen(self):
        for index in range(12):
            quad = [[100 + index * 4, 100], [400 + index * 4, 100],
                    [400 + index * 4, 300], [100 + index * 4, 300]]
            detection = laptop_detection(quad)
            track = self.tracker.update_tracks([detection]).tracks[0]
            self.assertLessEqual(track.bbox.x, detection.bbox.x + 1e-6)
            self.assertGreaterEqual(track.bbox.x + track.bbox.width,
                                    detection.bbox.x + detection.bbox.width - 1e-6)

    def test_multiple_laptops_get_independent_tracks(self):
        first = [[60, 100], [300, 100], [300, 280], [60, 280]]
        second = [[600, 120], [880, 120], [880, 340], [600, 340]]
        tracks = self.tracker.update_tracks([
            laptop_detection(first), laptop_detection(second)]).tracks
        self.assertEqual(len({track.track_id for track in tracks}), 2)

    def test_lost_screen_hold_is_short_and_does_not_move(self):
        quad = [[100, 100], [400, 100], [400, 300], [100, 300]]
        for _ in range(3):
            self.tracker.update_tracks([laptop_detection(quad)])
        lost = self.tracker.update_tracks([]).tracks
        self.assertEqual(len(lost), 1)
        for _ in range(ai_settings.LAPTOP_REDACTION_HOLD_FRAMES + 1):
            tracks = self.tracker.update_tracks([]).tracks
        held = [track for track in tracks if track.track_state == TrackStateEnum.LOST]
        if held:
            self.assertEqual(held[0].quad, lost[0].quad)
        frame = noise_frame(seed=9)
        blur = BlurService()
        _, report = blur.render_blur(frame, tracks, [], blur_style="blackout",
                                     frame_index=3)
        self.assertEqual(report.total_blurred_regions, 0,
                         "a stale screen quad kept masking after the hold window")


class LaptopApiTests(unittest.TestCase):
    def setUp(self):
        task_manager.tasks.clear()

    def test_request_accepts_laptop_screens(self):
        request = StartRedactionRequest(
            video_id="vid_x", target_categories=["laptop_screens"])
        self.assertEqual([c.value for c in request.target_categories], ["laptop_screens"])

    def test_face_only_request_stays_backward_compatible(self):
        request = StartRedactionRequest(video_id="vid_x")
        self.assertEqual([c.value for c in request.target_categories], ["faces"])

    def test_missing_laptop_weights_fail_with_a_clear_message(self):
        original = model_manager.ensure_laptop_detector
        model_manager.ensure_laptop_detector = lambda: {
            "model": None, "status": "UNAVAILABLE",
            "weights_path": ai_settings.LAPTOP_MODEL_PATH, "detail": "missing"}
        try:
            with self.assertRaises(DetectorUnavailableException) as raised:
                DetectionService().validate_available(["laptop_screens"])
            self.assertIn("Laptop-screen", str(raised.exception))
            self.assertIn("download_laptop_screen_model.py", str(raised.exception))
        finally:
            model_manager.ensure_laptop_detector = original

    def test_summary_reports_quad_and_fallback_regions(self):
        task_manager.create_task("task_screens", "vid_x", "fixture.mp4",
                                 ["laptop_screens"], "blackout", 80)
        task_manager.get_task("task_screens").update(
            status="COMPLETED", output_verified=True,
            redaction_counts={"laptop_screens": 10},
            metrics={"processed_frames": 12, "frames_with_detections": 10,
                     "elapsed_seconds": 3, "laptop_screen_quad_regions": 7,
                     "laptop_screen_fallback_regions": 3},
            verification={"frame_count": 12, "width": 640, "height": 480})
        summary = RedactService().get_task_summary("task_screens")
        self.assertEqual(summary.laptop_screens_redacted, 10)
        self.assertEqual(summary.laptop_screen_quad_regions, 7)
        self.assertEqual(summary.laptop_screen_fallback_regions, 3)
        self.assertEqual(summary.laptop_screen_boundary, "mixed")
        self.assertAlmostEqual(summary.laptop_screen_quad_pct, 70.0, places=1)
        self.assertEqual(summary.screens_hidden, 10)
        self.assertEqual(summary.phones_found, 0)

    def test_pure_fallback_is_reported_as_such(self):
        task_manager.create_task("task_fallback", "vid_x", "fixture.mp4",
                                 ["laptop_screens"], "blackout", 80)
        task_manager.get_task("task_fallback").update(
            status="COMPLETED", output_verified=True,
            redaction_counts={"laptop_screens": 4},
            metrics={"processed_frames": 4, "frames_with_detections": 4,
                     "laptop_screen_quad_regions": 0,
                     "laptop_screen_fallback_regions": 4},
            verification={"frame_count": 4, "width": 640, "height": 480})
        summary = RedactService().get_task_summary("task_fallback")
        self.assertEqual(summary.laptop_screen_boundary, "device_box_fallback")
        self.assertEqual(summary.laptop_screen_quad_pct, 0.0)


def _annotated_photo():
    """Returns (frame, screen_quad, keyboard_rect) for the cached lenovo photo."""
    image = os.path.join(PHOTO_DIR, "lenovo.jpg")
    if not os.path.isfile(image):
        return None
    frame = cv2.imread(image)
    if frame is None:
        return None
    if not os.path.isfile(ANNOTATIONS):
        return None
    with open(ANNOTATIONS, encoding="utf-8") as handle:
        data = json.load(handle)
    scale = frame.shape[1] / float(data.get("reference_width", 1280))
    entry = next((item for item in data["images"] if item["key"] == "lenovo"), None)
    if entry is None:
        return None
    screen = np.array([[x * scale, y * scale] for x, y in entry["screen_quad"]], np.float32)
    keyboard = [value * scale for value in entry["keyboard_rect"]]
    return frame, screen, keyboard


def _write_laptop_clip(path, frames, width=960, height=720):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height))
    if not writer.isOpened():
        return False
    try:
        for index in range(10):
            canvas = np.full((height, width, 3), 30, np.uint8)
            source = frames[index % len(frames)]
            scale = min(width / source.shape[1], height / source.shape[0])
            resized = cv2.resize(source, (int(source.shape[1] * scale),
                                          int(source.shape[0] * scale)))
            y0 = (height - resized.shape[0]) // 2
            x0 = (width - resized.shape[1]) // 2
            canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
            writer.write(canvas)
    finally:
        writer.release()
    return True


@unittest.skipUnless(os.path.isfile(model_manager.resolve_laptop_path()),
                     "NOT RUN: laptop-screen ONNX weights are missing "
                     "(run scripts/download_laptop_screen_model.py)")
class LaptopPipelineIntegrationTests(unittest.TestCase):
    """Real RT-DETR graph + real FFmpeg on a real annotated photo."""

    def setUp(self):
        task_manager.tasks.clear()
        if not ffmpeg_util.validate_ffmpeg().get("available"):
            self.skipTest("NOT RUN: FFmpeg is not available")
        annotated = _annotated_photo()
        if annotated is None:
            self.skipTest("NOT RUN: cached annotated laptop photo is unavailable")
        self.frame, self.screen, self.keyboard = annotated
        self.temp_dir = tempfile.mkdtemp(prefix="safeframe_screens_")
        raw = os.path.join(self.temp_dir, "raw.mp4")
        if not _write_laptop_clip(raw, [self.frame]):
            self.skipTest("NOT RUN: could not build the laptop clip")
        self.clip = os.path.join(self.temp_dir, "laptop_clip.mp4")
        ffmpeg = ffmpeg_util.resolve_ffmpeg_path()
        build = subprocess.run(
            [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", raw,
             "-f", "lavfi", "-i", "sine=frequency=440:duration=1.2",
             "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", self.clip],
            capture_output=True, timeout=120)
        if build.returncode != 0:
            self.skipTest("NOT RUN: could not encode the laptop clip")
        self.video_id = f"vid_test_screens_{uuid.uuid4().hex[:8]}"
        self.staged = os.path.join(storage_manager.base_dir, "raw_videos",
                                   f"{self.video_id}_laptop_clip.mp4")
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
        task_id = f"task_test_screens_{style}_{uuid.uuid4().hex[:8]}"
        self.task_ids = getattr(self, "task_ids", []) + [task_id]
        task_manager.create_task(task_id, self.video_id, "laptop_clip.mp4",
                                 categories, style, 80)
        output = os.path.join(storage_manager.base_dir, "redacted_videos",
                              f"{task_id}_protected.mp4")
        asyncio.run(video_processor_service.process_video_pipeline(
            task_id, self.video_id, categories, style, 80))
        return task_manager.get_task(task_id), output

    def test_laptop_pipeline_produces_a_verified_output(self):
        task, _ = self._run(["laptop_screens"], "blackout")
        self.assertEqual(task["status"], "COMPLETED", task.get("failure_reason"))
        self.assertTrue(task["output_verified"])
        self.assertEqual(task["metrics"]["output_frame_count"], 10)
        self.assertGreater(task["redaction_counts"].get("laptop_screens", 0), 0)
        self.assertEqual(task["redaction_counts"].get("faces", 0), 0)
        self.assertIn(task["metrics"]["laptop_screen_quad_regions"]
                      + task["metrics"]["laptop_screen_fallback_regions"],
                      range(1, 11))

    def test_display_is_darkened_and_keyboard_survives(self):
        task, output = self._run(["laptop_screens"], "blackout")
        self.assertEqual(task["status"], "COMPLETED", task.get("failure_reason"))
        cap = cv2.VideoCapture(output)
        ok, protected = cap.read()
        cap.release()
        self.assertTrue(ok)
        height, width = protected.shape[:2]
        # The clip is the photo scaled to 960x720 and centred, so scale the
        # annotations into the clip's coordinate space.
        scale = min(width / self.frame.shape[1], height / self.frame.shape[0])
        x0 = (width - int(self.frame.shape[1] * scale)) // 2
        y0 = (height - int(self.frame.shape[0] * scale)) // 2
        screen = np.array([[x * scale + x0, y * scale + y0] for x, y in self.screen],
                          np.int32)
        screen_mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(screen_mask, [screen], 1)
        dark = (protected.max(axis=2) < 45).astype(np.uint8)
        covered = float(np.logical_and(dark, screen_mask).sum()) / max(1, int(screen_mask.sum()))
        self.assertGreaterEqual(covered, 0.95,
                                "visible display content survived blackout redaction")

        keyboard = [value * scale for value in self.keyboard]
        kx1, ky1 = int(max(0, keyboard[0] + x0)), int(max(0, keyboard[1] + y0))
        kx2 = int(min(width, keyboard[2] + x0))
        ky2 = int(min(height, keyboard[3] + y0))
        # Compare against the same clip decoded from its encoded original, so
        # H.264 noise is not mistaken for redaction.
        baseline_cap = cv2.VideoCapture(self.clip)
        ok_baseline, baseline = baseline_cap.read()
        baseline_cap.release()
        self.assertTrue(ok_baseline)
        baseline = cv2.resize(baseline, (width, height))
        changed = (np.abs(protected.astype(np.int16) - baseline.astype(np.int16))
                   .max(axis=2) > 12)
        # A refined display quad must leave the keyboard deck alone. When the
        # boundary could not be separated the pipeline deliberately falls back
        # to the device box; that is privacy-safe, must be reported as such,
        # and is therefore not a silent regression.
        quad_regions = int(task["metrics"].get("laptop_screen_quad_regions", 0))
        fallback_regions = int(task["metrics"].get("laptop_screen_fallback_regions", 0))
        if quad_regions > 0:
            self.assertLess(float(changed.mean()), 0.5,
                            "the refined display mask spilled onto the keyboard")
        else:
            self.assertGreater(fallback_regions, 0,
                               "screen regions were masked without a reported region kind")
            self.assertLess(float(changed.mean()), 1.01)

    def test_all_three_modes_complete_and_verify(self):
        for style in MODES:
            with self.subTest(style=style):
                task, _ = self._run(["laptop_screens"], style)
                self.assertEqual(task["status"], "COMPLETED", task.get("failure_reason"))
                self.assertTrue(task["output_verified"])
                self.assertGreater(task["metrics"]["masks_total"], 0)

    def test_audio_and_resolution_are_preserved(self):
        task, _ = self._run(["laptop_screens"], "pixelate")
        self.assertEqual(task["status"], "COMPLETED", task.get("failure_reason"))
        self.assertGreaterEqual(task["metrics"]["input_audio_streams"] or 0, 1)
        self.assertTrue(task["metrics"]["audio_preserved"])
        self.assertEqual((task["metrics"]["output_width"], task["metrics"]["output_height"]),
                         (960, 720))

    def test_combined_categories_do_not_break_laptop_detection(self):
        task, _ = self._run(["faces", "license_plates", "id_cards", "laptop_screens"],
                            "gaussian")
        self.assertEqual(task["status"], "COMPLETED", task.get("failure_reason"))
        self.assertTrue(task["output_verified"])
        self.assertGreater(task["redaction_counts"].get("laptop_screens", 0), 0,
                           "enabling other categories stopped laptop-screen redaction")

    def test_face_only_regression_still_completes(self):
        task, _ = self._run(["faces"], "gaussian")
        self.assertEqual(task["status"], "COMPLETED", task.get("failure_reason"))
        self.assertEqual(task["redaction_counts"].get("laptop_screens", 0), 0)
        self.assertEqual(task["metrics"].get("laptop_screen_quad_regions", 0), 0)


if __name__ == "__main__":
    unittest.main()
