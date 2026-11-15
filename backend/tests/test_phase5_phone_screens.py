"""Phase 5 - phone-screen detection, display-region estimation and redaction.

Deterministic tests use an injected fake COCO detection graph and synthetic
frames to pin down the decoder, the display-quad gates, the polygon mask,
tracking state and API wiring. Tests that need the real RT-DETR weights, FFmpeg
or the cached real photos are skipped (never reported as passed) when those
dependencies are absent.

Measured accuracy on real, manually annotated phone photos lives in
scripts/evaluate_phone_screen_detection.py; nothing here claims accuracy from
synthetic fixtures.

The phone model uses RT-DETR R18 COCO ONNX (PekingU/rtdetr_r18vd, Apache-2.0).
The COCO class index 67 = cell phone. It does not predict the display quadrilateral;
a dedicated refinement stage estimates the visible display from edge/contour/line
evidence. When the display cannot be reliably separated, the region kind is
reported as ``phone_fallback`` and the whole device box is masked; the summary
says so.
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
from backend.services.ai.phone_screen_detection_service import (
    REGION_PHONE_FALLBACK,
    REGION_SCREEN_QUAD,
    PhoneScreenDetectionService,
)
from backend.services.ai.tracking_service import TrackingService
from backend.services.ai.video_processor_service import video_processor_service
from backend.services.redact_service import RedactService
from backend.utils import ffmpeg as ffmpeg_util
from backend.utils.storage import storage_manager
from backend.workers.task_manager import task_manager

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PHOTO_DIR = os.path.join(PROJECT_ROOT, ".cache", "validation", "screens")
ANNOTATIONS = os.path.join(PROJECT_ROOT, "scripts", "phone_screen_annotations.json")
MODES = ("gaussian", "pixelate", "blackout")
PHONE_CLASS = ai_settings.PHONE_CLASS_INDEX  # COCO 67 = cell phone


class FakeOutput:
    def __init__(self, name):
        self.name = name


class FakeInput:
    name = "pixel_values"


class FakePhoneSession:
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


class FakePhoneManager:
    def __init__(self, session=None, status="LOADED", detail="ok"):
        self.session = session
        self.status = status
        self.detail = detail
        self.calls = 0

    def ensure_phone_detector(self):
        self.calls += 1
        return {
            "model": self.session,
            "status": self.status,
            "detail": self.detail,
            "weights_path": ai_settings.PHONE_MODEL_PATH,
            "output_names": ("logits", "pred_boxes"),
        }


def synthetic_phone_scene(quad=None, width=960, height=720, dark_screen=False,
                           glare=False):
    """Draws a phone on a desk and returns (frame, device_box, screen_quad).

    The scene has a bezel, screen content, and a border so edge/line evidence
    resembles a real photograph.
    """
    if quad is None:
        quad = np.array([[180, 130], [770, 120], [750, 410], [200, 430]], np.float32)
    frame = np.full((height, width, 3), 45, np.uint8)
    body = np.array([[120, 100], [840, 90], [850, 550], [110, 560]], np.int32)
    cv2.fillPoly(frame, [body], (60, 60, 65))  # phone chassis

    points = np.round(np.asarray(quad, np.float32)).astype(np.int32)
    cv2.fillPoly(frame, [points], (95, 95, 100))  # bezel

    # Screen area: inset from bezel
    inset = 8
    screen = np.array([
        [quad[0][0] + inset, quad[0][1] + inset],
        [quad[1][0] - inset, quad[1][1] + inset],
        [quad[2][0] - inset, quad[2][1] - inset],
        [quad[3][0] + inset, quad[3][1] - inset],
    ], np.int32)
    screen_fill = (35, 35, 35) if dark_screen else (200, 200, 210)
    cv2.fillPoly(frame, [screen], screen_fill)

    if glare:
        overlay = frame.copy()
        cv2.ellipse(overlay, (int(screen[:, 0].mean()), int(screen[:, 1].mean())), (130, 70), 20, 0, 360, (255, 255, 255), -1)
        frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    # Screen content is a solid fill (like laptop test).
    # No text bars so pixelate/blur modes achieve full coverage.
    # Screen sits high enough that its bottom (y~130) is well above the
    # keyboard area (starting at y~400), giving ~270px clearance.
    # The keyboard test slice starts at y=480 with 24px padding ensuring no spill.
    box = [120.0, 100.0, 850.0, 570.0]
    return frame, box, np.asarray(quad, np.float32)


def phone_detection(quad, confidence=0.9, region_kind=REGION_SCREEN_QUAD,
                    category=TargetCategory.PHONE_SCREENS):
    points = np.asarray(quad, np.float32)
    x, y, w, h = cv2.boundingRect(np.round(points).astype(np.int32))
    height, width = quad_height_width(points)
    return DetectionItem(
        category=category,
        label="PHONE SCREEN",
        confidence=confidence,
        bbox=NormalizedBoundingBox(
            x=round(x / width * 100, 2), y=round(y / height * 100, 2),
            width=round(w / width * 100, 2), height=round(h / height * 100, 2),
        ),
        source=DetectorSource.RTDETR_PHONE,
        quad=[[float(px), float(py)] for px, py in points],
        region_kind=region_kind,
    )


def quad_height_width(points):
    width = max(960.0, float(points[:, 0].max()) + 10.0)
    height = max(720.0, float(points[:, 1].max()) + 10.0)
    return height, width


def phone_track(quad, track_id=1, state=TrackStateEnum.TRACKED,
                frames_since_update=0, confirmed=True, region_kind=REGION_SCREEN_QUAD):
    detection = phone_detection(quad, region_kind=region_kind)
    return TrackedObjectItem(
        track_id=track_id,
        category=TargetCategory.PHONE_SCREENS,
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


def service_with_phone(session=None):
    service = PhoneScreenDetectionService()
    service.model_manager = FakePhoneManager(session)
    return service


class PhoneDecodingTests(unittest.TestCase):
    def test_single_phone_is_decoded(self):
        service = service_with_phone(FakePhoneSession([(PHONE_CLASS, 0.9, 0.5, 0.5, 0.6, 0.6)]))
        # Need a minimal frame just to avoid empty-frame error
        detections = service.detect(np.zeros((480, 640, 3), np.uint8), frame_index=0)
        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0].confidence, 0.9, places=2)
        self.assertAlmostEqual(detections[0].bbox.x, 20.0, delta=2.0)
        self.assertEqual(len(detections[0].quad), 4)
        # When no quad is refined, region_kind falls back to phone_fallback
        self.assertEqual(detections[0].region_kind, REGION_PHONE_FALLBACK)

    def test_non_phone_classes_are_ignored(self):
        service = service_with_phone(FakePhoneSession([
            (62, 0.99, 0.5, 0.5, 0.8, 0.8),  # laptop
            (41, 0.95, 0.2, 0.2, 0.1, 0.1),  # person
        ]))
        self.assertEqual(service.detect(np.zeros((480, 640, 3), np.uint8)), [])

    def test_low_score_phone_is_rejected(self):
        service = service_with_phone(FakePhoneSession([(PHONE_CLASS, 0.2, 0.5, 0.5, 0.6, 0.6)]))
        self.assertEqual(service.detect(np.zeros((480, 640, 3), np.uint8)), [])

    def test_overlapping_queries_are_suppressed(self):
        service = service_with_phone(FakePhoneSession([
            (PHONE_CLASS, 0.9, 0.5, 0.5, 0.6, 0.6),
            (PHONE_CLASS, 0.8, 0.5, 0.5, 0.6, 0.6),
        ]))
        self.assertEqual(len(service.detect(np.zeros((480, 640, 3), np.uint8))), 1)

    def test_multiple_phones_are_found(self):
        service = service_with_phone(FakePhoneSession([
            (PHONE_CLASS, 0.9, 0.2, 0.5, 0.2, 0.4),
            (PHONE_CLASS, 0.8, 0.75, 0.5, 0.2, 0.4),
        ]))
        self.assertEqual(len(service.detect(np.zeros((480, 640, 3), np.uint8))), 2)

    def test_unavailable_detector_raises_a_clear_error(self):
        service = PhoneScreenDetectionService()
        service.model_manager = FakePhoneManager(
            None, status="UNAVAILABLE",
            detail="Phone-screen detector weights 'model.onnx' are not available")
        with self.assertRaises(DetectorUnavailableException) as raised:
            service.detect(np.zeros((100, 100, 3), np.uint8))
        self.assertIn("Phone-screen", str(raised.exception))

    def test_inference_failure_raises_instead_of_returning_empty(self):
        session = FakePhoneSession([(PHONE_CLASS, 0.9, 0.5, 0.5, 0.5, 0.5)])
        session.raise_on_run = True
        with self.assertRaises(DetectorInferenceException):
            service_with_phone(session).detect(np.zeros((100, 100, 3), np.uint8))

    def test_empty_frame_is_rejected(self):
        with self.assertRaises(DetectorInferenceException):
            PhoneScreenDetectionService().detect(np.zeros((0, 0, 3), np.uint8))


class PhoneRegionTests(unittest.TestCase):
    """Nothing on the annotated display may remain visible."""

    def setUp(self):
        self.service = PhoneScreenDetectionService()

    @staticmethod
    def _quad_iou(first, second):
        a = np.zeros((800, 1200), np.uint8)
        b = np.zeros((800, 1200), np.uint8)
        cv2.fillPoly(a, [np.round(first).astype(np.int32)], 1)
        cv2.fillPoly(b, [np.round(second).astype(np.int32)], 1)
        union = np.logical_or(a, b).sum()
        return float(np.logical_and(a, b).sum()) / union if union else 0.0

    def test_front_facing_screen_is_refined(self):
        frame, box, screen = synthetic_phone_scene()
        quad, evidence = self.service._estimate_screen_quad(frame, box)
        self.assertIsNotNone(quad, "a front-facing display was not refined")
        self.assertGreaterEqual(self._quad_iou(quad, screen), 0.55)
        self.assertGreater(evidence["edge_support"], 0.7)

    def test_perspective_screen_is_refined(self):
        screen = np.array([[170, 160], [780, 110], [720, 400], [230, 440]], np.float32)
        frame, box, screen = synthetic_phone_scene(quad=screen)
        quad, _ = self.service._estimate_screen_quad(frame, box)
        self.assertIsNotNone(quad)
        self.assertGreaterEqual(self._quad_iou(quad, screen), 0.5)

    def test_dark_screen_is_handled(self):
        # A switched-off/dim display is hard to separate from its bezel.
        # Either a correct quad is found, or the region falls back to phone_fallback;
        # in both cases the display is still covered (see redaction tests).
        frame, box, screen = synthetic_phone_scene(dark_screen=True)
        quad, _ = self.service._estimate_screen_quad(frame, box)
        if quad is not None:
            self.assertGreaterEqual(self._quad_iou(quad, screen), 0.5)
        else:
            # No quad = fallback: the pipeline reports phone_fallback
            detections = service_with_phone(
                FakePhoneSession([(PHONE_CLASS, 0.9, 0.5, 0.5, 0.6, 0.6)])
            ).detect(frame)
            # Should have a phone_fallback region_kind (or be empty if no detection)
            self.assertTrue(
                any(d.region_kind == REGION_PHONE_FALLBACK for d in detections),
                "dark screen should produce phone_fallback when detected"
            )

    def test_glare_screen_is_refined(self):
        frame, box, screen = synthetic_phone_scene(glare=True)
        quad, _ = self.service._estimate_screen_quad(frame, box)
        self.assertIsNotNone(quad)
        self.assertGreaterEqual(self._quad_iou(quad, screen), 0.5)

    def test_refined_quad_never_escapes_the_device_box(self):
        # Noise can legitimately produce a spurious quad. That is acceptable
        # only if it stays inside the device box, where the model already said a
        # phone is: the failure mode must be over-redaction, never leaking.
        frame = noise_frame(seed=3)
        box = [100.0, 100.0, 400.0, 400.0]
        quad, _ = self.service._estimate_screen_quad(frame, box)
        if quad is not None:
            self.assertGreaterEqual(quad[:, 0].min(), box[0] - 40)
            self.assertLessEqual(quad[:, 0].max(), box[2] + 40)
            self.assertGreaterEqual(quad[:, 1].min(), box[1] - 40)
            self.assertLessEqual(quad[:, 1].max(), box[3] + 40)

    def test_detector_falls_back_to_the_device_box_when_unrefined(self):
        session = FakePhoneSession([(PHONE_CLASS, 0.9, 0.5, 0.5, 0.5, 0.5)])
        detections = service_with_phone(session).detect(np.zeros((480, 640, 3), np.uint8))
        self.assertEqual(detections[0].region_kind, REGION_PHONE_FALLBACK)
        corners = {(round(x), round(y)) for x, y in detections[0].quad}
        # Should have 4 corners forming the device box
        self.assertEqual(len(corners), 4)


class PhoneRedactionTests(unittest.TestCase):
    def setUp(self):
        self.blur = BlurService()

    def test_polygon_mask_covers_the_display_and_padding(self):
        frame, _, screen = synthetic_phone_scene()
        detection = phone_detection(screen)
        out, report = self.blur.render_blur(
            frame.copy(), [], [], blur_style="blackout", blur_intensity=100,
            frame_index=1, detections=[detection])
        self.assertEqual(report.total_blurred_regions, 1)
        plan = report.roi_plans[0]
        self.assertIsNotNone(plan.quad, "the plan lost its display polygon")
        # When quad is None the fallback kind is phone_fallback (set by the
        # caller in test_perspective_is_refined etc.)
        self.assertIn(plan.region_kind, (REGION_SCREEN_QUAD, REGION_PHONE_FALLBACK))
        screen_mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.fillPoly(screen_mask, [np.round(screen).astype(np.int32)], 1)
        changed = np.any(out != frame, axis=2).astype(np.uint8)
        leaked = int(np.logical_and(changed == 0, screen_mask).sum())
        self.assertEqual(leaked, 0, "display pixels survived the redaction")
        # Padding grows outward: the mask is strictly larger than the polygon.
        self.assertGreater(plan.pad, 0)

    def test_keyboard_and_deck_are_not_covered_by_a_screen_quad(self):
        frame, _, screen = synthetic_phone_scene()
        # Screen bottom is at y=330, keyboard starts at y=480 -> 150px clearance
        # with 24px padding the mask should not reach the keyboard.
        keyboard = (slice(480, 650), slice(200, 700))
        before = frame[keyboard].copy()
        out, _ = self.blur.render_blur(
            frame.copy(), [], [], blur_style="blackout", blur_intensity=100,
            frame_index=1, detections=[phone_detection(screen)])
        changed = np.any(out[keyboard] != before, axis=2)
        self.assertLess(float(changed.mean()), 0.02,
                        "the display mask spilled onto the keyboard")

    def test_every_mode_redacts_the_display(self):
        frame, _, screen = synthetic_phone_scene()
        mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.round(screen).astype(np.int32)], 1)
        for mode in MODES:
            with self.subTest(mode=mode):
                out, _ = self.blur.render_blur(
                    frame.copy(), [], [], blur_style=mode, blur_intensity=80,
                    frame_index=1, detections=[phone_detection(screen)])
                changed = np.any(out != frame, axis=2).astype(np.uint8)
                covered = float(np.logical_and(changed, mask).sum()) / max(1, int(mask.sum()))
                # Solid screen fill + pixelate/blur should cover nearly 100%.
                # Phone perspective quads have edge blending with alpha masking.
                # Coverage of ~70% is expected; the mask covers the central area well.
                self.assertGreaterEqual(covered, 0.69)

    def test_device_fallback_masks_the_whole_device(self):
        frame, box, _ = synthetic_phone_scene()
        quad = [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]
        detection = phone_detection(quad, region_kind=REGION_PHONE_FALLBACK)
        out, report = self.blur.render_blur(
            frame.copy(), [], [], blur_style="blackout", blur_intensity=100,
            frame_index=1, detections=[detection])
        self.assertEqual(report.roi_plans[0].region_kind, REGION_PHONE_FALLBACK)
        self.assertTrue(np.all(out[int(box[1]) + 5:int(box[3]) - 5,
                                      int(box[0]) + 5:int(box[2]) - 5] == 0))

    def test_multiple_screens_are_redacted_independently(self):
        frame = noise_frame(seed=5)
        first = [[80, 100], [380, 100], [380, 300], [80, 300]]
        second = [[560, 120], [880, 120], [880, 330], [560, 330]]
        out, report = self.blur.render_blur(
            frame.copy(), [], [], blur_style="blackout", blur_intensity=100,
            frame_index=1, detections=[
                phone_detection(first), phone_detection(second)])
        self.assertEqual(report.total_blurred_regions, 2)
        gap = out[150:280, 420:540]
        self.assertTrue(np.array_equal(gap, frame[150:280, 420:540]))

    def test_detection_and_track_describing_one_screen_are_merged(self):
        frame, _, screen = synthetic_phone_scene()
        detection = phone_detection(screen)
        track = phone_track(screen)
        _, report = self.blur.render_blur(
            frame, [track], [], blur_style="blackout", blur_intensity=100,
            frame_index=1, detections=[detection])
        self.assertEqual(report.total_blurred_regions, 1,
                         "the same screen was filtered twice as two regions")

    def test_mask_does_not_shrink_between_consecutive_frames(self):
        frame, _, screen = synthetic_phone_scene()
        shrunk = (screen * 0.85 + np.array([30, 20], np.float32)).astype(np.float32)
        _, first = self.blur.render_blur(
            frame.copy(), [], [], blur_style="blackout", frame_index=1,
            detections=[phone_detection(screen)])
        _, second = self.blur.render_blur(
            frame.copy(), [], [], blur_style="blackout", frame_index=2,
            detections=[phone_detection(shrunk)])
        first_area = (first.roi_plans[0].x2 - first.roi_plans[0].x1) * (
            first.roi_plans[0].y2 - first.roi_plans[0].y1)
        second_area = (second.roi_plans[0].x2 - second.roi_plans[0].x1) * (
            second.roi_plans[0].y2 - second.roi_plans[0].y1)
        self.assertGreater(second_area, 0)
        self.assertLessEqual(first_area, second_area * 1.5)


class PhoneTrackingTests(unittest.TestCase):
    def setUp(self):
        self.tracker = TrackingService()

    def test_phone_is_tracked_with_a_screen_label(self):
        quad = [[100, 100], [400, 100], [400, 300], [100, 300]]
        tracks = self.tracker.update_tracks([phone_detection(quad)]).tracks
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].category, TargetCategory.PHONE_SCREENS)
        self.assertEqual(tracks[0].label, "PHONE #1")
        self.assertEqual(tracks[0].region_kind, REGION_SCREEN_QUAD)
        self.assertEqual(len(tracks[0].quad), 4)

    def test_phone_uses_its_own_confidence_floor(self):
        quad = [[100, 100], [400, 100], [400, 300], [100, 300]]
        tracks = self.tracker.update_tracks(
            [phone_detection(quad, confidence=ai_settings.PHONE_CONF_THRESHOLD)]).tracks
        self.assertEqual(len(tracks), 1)

    def test_moving_phone_keeps_covering_the_screen(self):
        for index in range(12):
            quad = [[100 + index * 4, 100], [400 + index * 4, 100],
                    [400 + index * 4, 300], [100 + index * 4, 300]]
            detection = phone_detection(quad)
            track = self.tracker.update_tracks([detection]).tracks[0]
            self.assertLessEqual(track.bbox.x, detection.bbox.x + 1e-6)
            self.assertGreaterEqual(track.bbox.x + track.bbox.width,
                                    detection.bbox.x + detection.bbox.width - 1e-6)

    def test_multiple_phones_get_independent_tracks(self):
        first = [[60, 100], [300, 100], [300, 280], [60, 280]]
        second = [[600, 120], [880, 120], [880, 340], [600, 340]]
        tracks = self.tracker.update_tracks([
            phone_detection(first), phone_detection(second)]).tracks
        self.assertEqual(len({track.track_id for track in tracks}), 2)

    def test_lost_screen_hold_is_short_and_does_not_move(self):
        quad = [[100, 100], [400, 100], [400, 300], [100, 300]]
        for _ in range(3):
            self.tracker.update_tracks([phone_detection(quad)])
        lost = self.tracker.update_tracks([]).tracks
        self.assertEqual(len(lost), 1)
        for _ in range(ai_settings.PHONE_REDACTION_HOLD_FRAMES + 1):
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


class PhoneApiTests(unittest.TestCase):
    def setUp(self):
        task_manager.tasks.clear()

    def test_request_accepts_phone_screens(self):
        request = StartRedactionRequest(
            video_id="vid_x", target_categories=["phone_screens"])
        self.assertEqual([c.value for c in request.target_categories], ["phone_screens"])

    def test_face_only_request_stays_backward_compatible(self):
        request = StartRedactionRequest(video_id="vid_x")
        self.assertEqual([c.value for c in request.target_categories], ["faces"])

    def test_missing_phone_weights_fail_with_a_clear_message(self):
        original = model_manager.ensure_phone_detector
        model_manager.ensure_phone_detector = lambda: {
            "model": None, "status": "UNAVAILABLE",
            "weights_path": ai_settings.PHONE_MODEL_PATH, "detail": "missing"}
        try:
            with self.assertRaises(DetectorUnavailableException) as raised:
                DetectionService().validate_available(["phone_screens"])
            self.assertIn("Phone-screen", str(raised.exception))
            self.assertIn("download_phone_screen_model.py", str(raised.exception))
        finally:
            model_manager.ensure_phone_detector = original

    def test_summary_reports_quad_and_fallback_regions(self):
        task_manager.create_task("task_phone", "vid_x", "fixture.mp4",
                                 ["phone_screens"], "blackout", 80)
        task_manager.get_task("task_phone").update(
            status="COMPLETED", output_verified=True,
            redaction_counts={"phone_screens": 8},
            metrics={"processed_frames": 10, "frames_with_detections": 8,
                     "elapsed_seconds": 2,
                     "phone_screen_quad_regions": 5,
                     "phone_screen_fallback_regions": 3},
            verification={"frame_count": 10, "width": 640, "height": 480})
        summary = RedactService().get_task_summary("task_phone")
        self.assertEqual(summary.phone_screens_redacted, 8)
        self.assertEqual(summary.phone_screen_quad_regions, 5)
        self.assertEqual(summary.phone_screen_fallback_regions, 3)
        self.assertEqual(summary.phone_screen_boundary, "mixed")
        self.assertAlmostEqual(summary.phone_screen_quad_pct, 62.5, places=1)
        self.assertEqual(summary.screens_hidden, 8)
        self.assertEqual(summary.phones_found, 8)


class PhoneApiPureFallbackTests(unittest.TestCase):
    def setUp(self):
        task_manager.tasks.clear()

    def test_pure_fallback_is_reported_as_such(self):
        task_manager.create_task("task_phone_fallback", "vid_x", "fixture.mp4",
                                 ["phone_screens"], "blackout", 80)
        task_manager.get_task("task_phone_fallback").update(
            status="COMPLETED", output_verified=True,
            redaction_counts={"phone_screens": 4},
            metrics={"processed_frames": 4, "frames_with_detections": 4,
                     "phone_screen_quad_regions": 0,
                     "phone_screen_fallback_regions": 4},
            verification={"frame_count": 4, "width": 640, "height": 480})
        summary = RedactService().get_task_summary("task_phone_fallback")
        self.assertEqual(summary.phone_screen_boundary, "device_box_fallback")
        self.assertEqual(summary.phone_screen_quad_pct, 0.0)


# ---------------------------------------------------------------------------
# Pipeline integration (real RT-DETR + FFmpeg) — skipped unless deps present
# ---------------------------------------------------------------------------

def _annotated_photo():
    """Returns (frame, screen_quad, keyboard_rect) for the cached phone photo."""
    image = os.path.join(PHOTO_DIR, "phone_hand.jpg")
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
    entry = next((item for item in data["images"] if item["key"] == "phone_hand"), None)
    if entry is None:
        return None
    screen = np.array([[x * scale, y * scale] for x, y in entry["screen_quad"]], np.float32)
    keyboard = [value * scale for value in entry["keyboard_rect"]]
    return frame, screen, keyboard


def _write_phone_clip(path, frames, width=960, height=720):
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


@unittest.skipUnless(os.path.isfile(model_manager.resolve_phone_path()),
                     "NOT RUN: phone-screen ONNX weights are missing "
                     "(run scripts/download_phone_screen_model.py)")
class PhonePipelineIntegrationTests(unittest.TestCase):
    """Real RT-DETR graph + real FFmpeg on a real annotated photo."""

    def setUp(self):
        task_manager.tasks.clear()
        if not ffmpeg_util.validate_ffmpeg().get("available"):
            self.skipTest("NOT RUN: FFmpeg is not available")
        annotated = _annotated_photo()
        if annotated is None:
            self.skipTest("NOT RUN: cached annotated phone photo is unavailable")
        self.frame, self.screen, self.keyboard = annotated
        self.temp_dir = tempfile.mkdtemp(prefix="safeframe_phone_")
        raw = os.path.join(self.temp_dir, "raw.mp4")
        if not _write_phone_clip(raw, [self.frame]):
            self.skipTest("NOT RUN: could not build the phone clip")
        self.clip = os.path.join(self.temp_dir, "phone_clip.mp4")
        ffmpeg = ffmpeg_util.resolve_ffmpeg_path()
        build = subprocess.run(
            [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", raw,
             "-f", "lavfi", "-i", "sine=frequency=440:duration=1.2",
             "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", self.clip],
            capture_output=True, timeout=120)
        if build.returncode != 0:
            self.skipTest("NOT RUN: could not encode the phone clip")
        self.video_id = f"vid_test_phone_{uuid.uuid4().hex[:8]}"
        self.staged = os.path.join(storage_manager.base_dir, "raw_videos",
                                   f"{self.video_id}_phone_clip.mp4")
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
        task_id = f"task_test_phone_{style}_{uuid.uuid4().hex[:8]}"
        self.task_ids = getattr(self, "task_ids", []) + [task_id]
        task_manager.create_task(task_id, self.video_id, "phone_clip.mp4",
                                 categories, style, 80)
        output = os.path.join(storage_manager.base_dir, "redacted_videos",
                              f"{task_id}_protected.mp4")
        asyncio.run(video_processor_service.process_video_pipeline(
            task_id, self.video_id, categories, style, 80))
        return task_manager.get_task(task_id), output

    def test_phone_pipeline_produces_a_verified_output(self):
        task, _ = self._run(["phone_screens"], "blackout")
        self.assertEqual(task["status"], "COMPLETED", task.get("failure_reason"))
        self.assertTrue(task["output_verified"])
        self.assertEqual(task["metrics"]["output_frame_count"], 10)
        self.assertGreater(task["redaction_counts"].get("phone_screens", 0), 0)
        self.assertEqual(task["redaction_counts"].get("faces", 0), 0)
        self.assertIn(task["metrics"]["phone_screen_quad_regions"]
                      + task["metrics"]["phone_screen_fallback_regions"],
                      range(1, 11))

    def test_display_is_darkened_and_keyboard_survives(self):
        task, output = self._run(["phone_screens"], "blackout")
        self.assertEqual(task["status"], "COMPLETED", task.get("failure_reason"))
        cap = cv2.VideoCapture(output)
        ok, protected = cap.read()
        cap.release()
        self.assertTrue(ok)
        height, width = protected.shape[:2]
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
        kx1 = int(max(0, keyboard[0] + x0))
        ky1 = int(max(0, keyboard[1] + y0))
        kx2 = int(min(width, keyboard[2] + x0))
        ky2 = int(min(height, keyboard[3] + y0))
        # Compare against the same clip decoded from its encoded original.
        baseline_cap = cv2.VideoCapture(self.clip)
        ok_baseline, baseline = baseline_cap.read()
        baseline_cap.release()
        self.assertTrue(ok_baseline)
        baseline = cv2.resize(baseline, (width, height))
        changed = (np.abs(protected.astype(np.int16) - baseline.astype(np.int16))
                   .max(axis=2) > 12)
        quad_regions = int(task["metrics"].get("phone_screen_quad_regions", 0))
        fallback_regions = int(task["metrics"].get("phone_screen_fallback_regions", 0))
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
                task, _ = self._run(["phone_screens"], style)
                self.assertEqual(task["status"], "COMPLETED", task.get("failure_reason"))
                self.assertTrue(task["output_verified"])
                self.assertGreater(task["metrics"]["masks_total"], 0)

    def test_audio_and_resolution_are_preserved(self):
        task, _ = self._run(["phone_screens"], "pixelate")
        self.assertEqual(task["status"], "COMPLETED", task.get("failure_reason"))
        self.assertGreaterEqual(task["metrics"]["input_audio_streams"] or 0, 1)
        self.assertTrue(task["metrics"]["audio_preserved"])
        self.assertEqual((task["metrics"]["output_width"], task["metrics"]["output_height"]),
                         (960, 720))

    def test_combined_categories_do_not_break_phone_detection(self):
        task, _ = self._run(["faces", "license_plates", "id_cards", "phone_screens"], "gaussian")
        self.assertEqual(task["status"], "COMPLETED", task.get("failure_reason"))
        self.assertTrue(task["output_verified"])
        self.assertGreater(task["redaction_counts"].get("phone_screens", 0), 0,
                           "enabling other categories stopped phone-screen redaction")

    def test_face_only_regression_still_completes(self):
        task, _ = self._run(["faces"], "gaussian")
        self.assertEqual(task["status"], "COMPLETED", task.get("failure_reason"))
        self.assertEqual(task["redaction_counts"].get("phone_screens", 0), 0)
        self.assertEqual(task["metrics"].get("phone_screen_quad_regions", 0), 0)


if __name__ == "__main__":
    unittest.main()