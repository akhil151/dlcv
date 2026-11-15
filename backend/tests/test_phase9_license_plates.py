"""Phase 9 - license-plate detection, tracking and redaction.

Deterministic tests use injected fake detectors to pin down decoding, NMS,
geometry, tracking state and API wiring. Tests that need the real RT-DETR
weights / FFmpeg / fixture photos are skipped (never reported as passed) when
those dependencies are absent.

Measured detector accuracy and the reproducible procedure live in
scripts/evaluate_plate_detection.py; nothing here claims accuracy from
synthetic frames.
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
from backend.core.exceptions import DetectorUnavailableException, SafeFrameException
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
from backend.services.ai.tracking_service import TrackingService
from backend.services.ai.video_processor_service import video_processor_service
from backend.services.redact_service import RedactService
from backend.utils import ffmpeg as ffmpeg_util
from backend.utils.storage import storage_manager
from backend.workers.task_manager import task_manager

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAMPLE_IMAGE = os.path.join(PROJECT_ROOT, ".cache", "platemodel", "sample.jpeg")
FACE_IMAGE = os.path.join(PROJECT_ROOT, "_lena.jpg")
MODES = ("gaussian", "pixelate", "blackout")


def logit(score: float) -> float:
    score = min(max(score, 1e-6), 1 - 1e-6)
    return float(math.log(score / (1.0 - score)))


class FakePlateNet:
    """Deterministic stand-in for the RT-DETR ONNX graph."""

    def __init__(self, boxes, scores):
        count = len(scores)
        logits = np.full(300, logit(0.01), dtype=np.float32)
        pred = np.zeros((300, 4), dtype=np.float32)
        for index in range(count):
            logits[index] = logit(scores[index])
            pred[index] = boxes[index]
        self.logits = logits.reshape(1, 300, 1)
        self.boxes = pred.reshape(1, 300, 4)
        self.input = None

    def setInput(self, blob):
        self.input = blob

    def getUnconnectedOutLayersNames(self):
        return ("logits", "pred_boxes")

    def forward(self, names=None):
        return [self.logits, self.boxes]


class FakeFaceNet:
    def __init__(self, faces=None):
        self.faces = faces
        self.calls = 0

    def setInputSize(self, size):
        self.size = size

    def detect(self, frame):
        self.calls += 1
        return None, self.faces


class FakeModelManager:
    """Injected in place of the singleton for detector-service tests."""

    def __init__(self, plate_net=None, plate_status="LOADED", face_net=None,
                 plate_detail="ok"):
        self.plate_net = plate_net
        self.plate_status = plate_status
        self.face_net = face_net
        self.face_calls = 0
        self.plate_calls = 0
        self.plate_detail = plate_detail

    def get_model(self, key):
        self.face_calls += 1
        return {
            "model": self.face_net,
            "status": "LOADED" if self.face_net is not None else "UNAVAILABLE",
            "weights_path": "face_detection_yunet_2023mar.onnx",
            "detail": "ok" if self.face_net is not None else "missing",
        }

    def ensure_plate_detector(self):
        self.plate_calls += 1
        return {
            "model": self.plate_net,
            "status": self.plate_status,
            "weights_path": ai_settings.PLATE_MODEL_PATH,
            "detail": self.plate_detail,
            "output_names": ("logits", "pred_boxes"),
        }


def plate_detection(x, y, w, h, conf=0.12):
    return DetectionItem(
        category=TargetCategory.LICENSE_PLATES, label="PLATE", confidence=conf,
        bbox=NormalizedBoundingBox(x=x, y=y, width=w, height=h),
        source=DetectorSource.RTDETR_PLATE,
    )


def plate_track(bbox, track_id=1, state=TrackStateEnum.TRACKED,
                frames_since_update=0, confirmed=True):
    return TrackedObjectItem(
        track_id=track_id, category=TargetCategory.LICENSE_PLATES,
        label=f"PLATE #{track_id}", confidence=0.12, bbox=bbox,
        track_state=state, age=5, frames_since_update=frames_since_update,
        is_confirmed=confirmed,
    )


def noise_frame(height, width, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(1, 255, (height, width, 3), dtype=np.uint8)


def exact_rect(bbox, width, height):
    return (bbox.x / 100.0 * width, bbox.y / 100.0 * height,
            (bbox.x + bbox.width) / 100.0 * width,
            (bbox.y + bbox.height) / 100.0 * height)


class PlateDetectionDecodingTests(unittest.TestCase):
    """The ONNX graph decoding must be exact and class-correct."""

    def _service(self, boxes, scores):
        service = DetectionService()
        service.model_manager = FakeModelManager(plate_net=FakePlateNet(boxes, scores))
        return service

    def test_single_plate_is_decoded_with_exact_box_and_confidence(self):
        service = self._service([(0.5, 0.5, 0.1, 0.05)], [0.9])
        result = service.detect_entities(
            np.zeros((500, 1000, 3), dtype=np.uint8), ["license_plates"])
        self.assertEqual(result.total_detections, 1)
        detection = result.detections[0]
        self.assertEqual(detection.category, TargetCategory.LICENSE_PLATES)
        self.assertEqual(detection.source, DetectorSource.RTDETR_PLATE)
        self.assertAlmostEqual(detection.confidence, 0.9, places=3)
        # Pixel rounding on decode may move each edge by less than a pixel.
        self.assertAlmostEqual(detection.bbox.x, 45.0, delta=0.2)
        self.assertAlmostEqual(detection.bbox.y, 47.5, delta=0.2)
        self.assertAlmostEqual(detection.bbox.width, 10.0, delta=0.2)
        self.assertAlmostEqual(detection.bbox.height, 5.0, delta=0.2)
        self.assertEqual(result.categories_detected, [TargetCategory.LICENSE_PLATES])
        self.assertTrue(result.plate_detector_used)
        self.assertFalse(result.secondary_detector_used)

    def test_multiple_plates_in_one_frame(self):
        boxes = [(0.2, 0.5, 0.2, 0.08), (0.7, 0.4, 0.15, 0.06)]
        service = self._service(boxes, [0.40, 0.30])
        result = service.detect_entities(
            np.zeros((480, 640, 3), dtype=np.uint8), ["license_plates"])
        self.assertEqual(result.total_detections, 2)
        self.assertTrue(all(d.category == TargetCategory.LICENSE_PLATES
                            for d in result.detections))

    def test_small_distant_and_large_close_plates(self):
        for width_ratio, height_ratio in ((0.02, 0.01), (0.6, 0.25)):
            with self.subTest(width=width_ratio, height=height_ratio):
                service = self._service([(0.5, 0.5, width_ratio, height_ratio)], [0.55])
                result = service.detect_entities(
                    np.zeros((1080, 1920, 3), dtype=np.uint8), ["license_plates"])
                self.assertEqual(result.total_detections, 1)
                self.assertAlmostEqual(result.detections[0].bbox.width,
                                       width_ratio * 100.0, places=1)
                self.assertAlmostEqual(result.detections[0].bbox.height,
                                       height_ratio * 100.0, places=1)

    def test_rotated_or_angled_plate_boxes_keep_their_aspect_ratio(self):
        # An angled plate still produces an axis-aligned, wide rectangle; the
        # decoder must not square it off.
        service = self._service([(0.5, 0.5, 0.2, 0.04)], [0.7])
        result = service.detect_entities(
            np.zeros((720, 1280, 3), dtype=np.uint8), ["license_plates"])
        box = result.detections[0].bbox
        self.assertGreater(box.width / box.height, 3.0)

    def test_low_confidence_boxes_are_rejected(self):
        service = self._service([(0.5, 0.5, 0.1, 0.05)], [0.01])
        result = service.detect_entities(
            np.zeros((480, 640, 3), dtype=np.uint8), ["license_plates"])
        self.assertEqual(result.total_detections, 0)
        self.assertEqual(result.categories_detected, [])

    def test_overlapping_queries_are_merged_by_nms(self):
        boxes = [(0.5, 0.5, 0.1, 0.05), (0.502, 0.501, 0.1, 0.05)]
        service = self._service(boxes, [0.9, 0.8])
        result = service.detect_entities(
            np.zeros((480, 640, 3), dtype=np.uint8), ["license_plates"])
        self.assertEqual(result.total_detections, 1)
        self.assertAlmostEqual(result.detections[0].confidence, 0.9, places=2)

    def test_out_of_frame_boxes_are_clipped(self):
        service = self._service([(0.02, 0.5, 0.1, 0.05)], [0.6])
        result = service.detect_entities(
            np.zeros((480, 640, 3), dtype=np.uint8), ["license_plates"])
        box = result.detections[0].bbox
        self.assertGreaterEqual(box.x, 0.0)
        self.assertLessEqual(box.x + box.width, 100.0)

    def test_face_detector_is_not_initialized_for_plate_only_jobs(self):
        manager = FakeModelManager(plate_net=FakePlateNet(
            [(0.5, 0.5, 0.1, 0.05)], [0.9]))
        service = DetectionService()
        service.model_manager = manager
        service.detect_entities(np.zeros((100, 100, 3), np.uint8), ["license_plates"])
        self.assertEqual(manager.face_calls, 0)

    def test_plate_detector_is_not_initialized_for_face_only_jobs(self):
        manager = FakeModelManager(face_net=FakeFaceNet())
        service = DetectionService()
        service.model_manager = manager
        service.detect_entities(np.zeros((100, 100, 3), np.uint8), ["faces"])
        self.assertEqual(manager.plate_calls, 0)

    def test_combined_job_runs_both_detectors_on_one_frame(self):
        faces = np.array([[10, 10, 20, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.91]])
        manager = FakeModelManager(
            plate_net=FakePlateNet([(0.7, 0.5, 0.1, 0.05)], [0.5]),
            face_net=FakeFaceNet(faces))
        service = DetectionService()
        service.model_manager = manager
        result = service.detect_entities(
            np.zeros((100, 100, 3), np.uint8), ["faces", "license_plates"])
        self.assertEqual({d.category for d in result.detections},
                         {TargetCategory.FACES, TargetCategory.LICENSE_PLATES})
        self.assertEqual(len(result.detections), 2)
        self.assertTrue(result.secondary_detector_used)
        self.assertTrue(result.plate_detector_used)
        self.assertEqual(manager.face_calls > 0, True)
        self.assertEqual(manager.plate_calls > 0, True)

    def test_missing_plate_weights_raise_a_clear_unavailable_error(self):
        service = DetectionService()
        service.model_manager = FakeModelManager(
            plate_net=None, plate_status="UNAVAILABLE",
            plate_detail="License-plate detector weights 'plate.onnx' are not available")
        with self.assertRaises(DetectorUnavailableException) as raised:
            service.validate_available(["license_plates"])
        self.assertIn("License-plate", str(raised.exception))
        with self.assertRaises(DetectorUnavailableException):
            service.detect_entities(np.zeros((60, 60, 3), np.uint8), ["license_plates"])

    def test_unvalidated_categories_are_still_rejected(self):
        # Laptop and phone screens are validated; unknown categories are not.
        with self.assertRaises(DetectorUnavailableException):
            DetectionService().validate_available(["smartwatches"])


class PlateRedactionGeometryTests(unittest.TestCase):
    """Plates need a rectangular mask, proportional padding and full coverage."""

    def setUp(self):
        self.svc = BlurService()

    def test_padding_is_proportional_and_uses_plate_ratios(self):
        cases = [
            NormalizedBoundingBox(x=40.0, y=40.0, width=10.0, height=4.0),
            NormalizedBoundingBox(x=10.0, y=45.0, width=20.0, height=3.0),
        ]
        for bbox in cases:
            with self.subTest(bbox=bbox):
                plan = self.svc._convert_normalized_bbox_to_roi(
                    bbox, 2000, 1000, TargetCategory.LICENSE_PLATES, False)
                fx1, fy1, fx2, fy2 = exact_rect(bbox, 2000, 1000)
                core_w = math.ceil(fx2) - math.floor(fx1)
                core_h = math.ceil(fy2) - math.floor(fy1)
                expected_x = max(ai_settings.PLATE_MIN_PADDING_PX,
                                 min(ai_settings.PLATE_MAX_PADDING_PX,
                                     round(core_w * ai_settings.PLATE_PADDING_X_RATIO)))
                expected_y = max(ai_settings.PLATE_MIN_PADDING_PX,
                                 min(ai_settings.PLATE_MAX_PADDING_PX,
                                     round(core_h * ai_settings.PLATE_PADDING_Y_RATIO)))
                self.assertEqual(math.floor(fx1) - plan.x1, expected_x)
                self.assertEqual(plan.x2 - math.ceil(fx2), expected_x)
                self.assertEqual(math.floor(fy1) - plan.y1, expected_y)
                self.assertEqual(plan.y2 - math.ceil(fy2), expected_y)

    def test_whole_plate_roi_is_filtered_in_every_mode(self):
        bbox = NormalizedBoundingBox(x=41.2, y=40.3, width=12.6, height=4.1)  # ~3:1
        for mode in MODES:
            with self.subTest(mode=mode):
                frame = noise_frame(600, 900, seed=3)
                original = frame.copy()
                out, report = self.svc.render_blur(
                    frame, [plate_track(bbox)], [], blur_style=mode,
                    blur_intensity=80, frame_index=1)
                self.assertEqual(report.total_blurred_regions, 1)
                plan = report.roi_plans[0]
                roi = out[plan.y1:plan.y2, plan.x1:plan.x2]
                if mode == "blackout":
                    self.assertTrue(np.all(roi == 0))
                else:
                    self.assertFalse(np.array_equal(
                        roi, original[plan.y1:plan.y2, plan.x1:plan.x2]))
                if mode != "pixelate":
                    # Unlike a face's rounded mask, the plate's rectangle covers
                    # the ROI corner: no character fragment can survive there.
                    # (Pixelation can coincidentally keep a block's own pixel.)
                    self.assertFalse(np.array_equal(
                        out[plan.y1, plan.x1], original[plan.y1, plan.x1]))
                else:
                    changed = (roi != original[plan.y1:plan.y2,
                                             plan.x1:plan.x2]).any(axis=2)
                    self.assertGreaterEqual(float(changed.mean()), 0.9)

    def test_plate_mask_covers_wide_and_tall_plate_aspect_ratios(self):
        boxes = [
            NormalizedBoundingBox(x=30.0, y=45.0, width=25.0, height=6.0),
            NormalizedBoundingBox(x=60.0, y=20.0, width=6.0, height=18.0),
            NormalizedBoundingBox(x=45.0, y=45.0, width=10.0, height=10.0),
        ]
        frame = noise_frame(480, 640, seed=7)
        out, report = self.svc.render_blur(
            frame, [plate_track(b, i + 1) for i, b in enumerate(boxes)], [],
            blur_style="blackout", frame_index=1)
        self.assertEqual(report.total_blurred_regions, 3)
        for plan in report.roi_plans:
            self.assertTrue(np.all(out[plan.core_y1:plan.core_y2,
                                       plan.core_x1:plan.core_x2] == 0))

    def test_multiple_plates_are_redacted_independently(self):
        boxes = [
            NormalizedBoundingBox(x=5.0, y=60.0, width=12.0, height=4.0),
            NormalizedBoundingBox(x=55.0, y=62.0, width=14.0, height=5.0),
        ]
        frame = noise_frame(400, 800, seed=9)
        out, report = self.svc.render_blur(
            frame, [plate_track(b, i + 1) for i, b in enumerate(boxes)], [],
            blur_style="blackout", frame_index=1)
        self.assertEqual(report.total_blurred_regions, 2)
        # The gap between plates stays untouched.
        untouched = (out != frame).any(axis=2)
        plan_a, plan_b = sorted(report.roi_plans, key=lambda p: p.x1)
        self.assertFalse(untouched[plan_a.y1:plan_a.y2, plan_a.x2:plan_b.x1].any())

    def test_padding_does_not_swallow_the_vehicle(self):
        bbox = NormalizedBoundingBox(x=40.0, y=40.0, width=12.0, height=4.0)
        plan = self.svc._convert_normalized_bbox_to_roi(
            bbox, 1920, 1080, TargetCategory.LICENSE_PLATES, False)
        pad_x = plan.core_x1 - plan.x1
        pad_y = plan.core_y1 - plan.y1
        self.assertLessEqual(pad_x, math.ceil(12.0 / 100.0 * 1920 * 0.10))
        self.assertLessEqual(pad_y, math.ceil(4.0 / 100.0 * 1080 * 0.30))

    def test_face_and_plate_regions_are_not_merged(self):
        face = DetectionItem(
            category=TargetCategory.FACES, label="FACE", confidence=0.9,
            bbox=NormalizedBoundingBox(x=10.0, y=10.0, width=30.0, height=30.0))
        plate = plate_detection(12.0, 12.0, 28.0, 8.0, conf=0.5)
        frame = noise_frame(300, 300, seed=11)
        _, report = self.svc.render_blur(
            frame, [], [], blur_style="blackout", frame_index=1,
            detections=[face, plate])
        self.assertEqual(report.total_blurred_regions, 2)
        self.assertEqual({plan.category for plan in report.roi_plans},
                         {TargetCategory.FACES, TargetCategory.LICENSE_PLATES})


class PlateTrackingTests(unittest.TestCase):
    def setUp(self):
        self.tracker = TrackingService()
        self.frame = noise_frame(480, 640, seed=13)

    def test_low_score_plate_detections_are_tracked(self):
        # Regression: the face tracker floor (0.35) must not drop plate scores.
        result = self.tracker.update_tracks([plate_detection(20, 60, 12, 4, conf=0.10)])
        self.assertEqual(len(result.tracks), 1)
        self.assertEqual(result.tracks[0].category, TargetCategory.LICENSE_PLATES)
        self.assertEqual(result.tracks[0].label, "PLATE #1")

    def test_jittering_plate_does_not_jump_between_frames(self):
        rng = np.random.default_rng(17)
        previous = None
        ious = []
        for _ in range(20):
            detection = plate_detection(
                40.0 + float(rng.normal(0, 0.15)), 60.0 + float(rng.normal(0, 0.15)),
                12.0, 4.0, conf=0.2)
            track = self.tracker.update_tracks([detection]).tracks[0]
            current = (track.bbox.x, track.bbox.y)
            if previous is not None:
                ious.append(math.dist(previous, current))
            previous = current
        # Sub-percent movement on a 640px frame: the smoothed plate box follows
        # the detection instead of crawling with sub-pixel noise.
        self.assertLess(max(ious), 1.0, "plate box jittered between frames")

    def test_moving_plate_track_covers_every_detection(self):
        for index in range(15):
            bbox = NormalizedBoundingBox(x=10.0 + index * 3.0, y=62.0,
                                         width=12.0, height=4.0)
            track = self.tracker.update_tracks([plate_detection(
                bbox.x, bbox.y, bbox.width, bbox.height, conf=0.15)]).tracks[0]
            self.assertLessEqual(track.bbox.x, bbox.x + 1e-6)
            self.assertGreaterEqual(track.bbox.x + track.bbox.width,
                                    bbox.x + bbox.width - 1e-6)

    def test_multiple_plates_get_independent_tracks(self):
        detections = [
            plate_detection(5.0, 60.0, 12.0, 4.0, conf=0.2),
            plate_detection(55.0, 60.0, 12.0, 4.0, conf=0.2),
        ]
        tracks = self.tracker.update_tracks(detections).tracks
        self.assertEqual(len(tracks), 2)
        self.assertEqual(len({track.track_id for track in tracks}), 2)

    def test_hold_is_short_and_does_not_move_or_grow(self):
        for _ in range(3):
            self.tracker.update_tracks([plate_detection(20.0, 60.0, 12.0, 4.0, conf=0.2)])
        lost = self.tracker.update_tracks([]).tracks
        self.assertEqual(len(lost), 1)
        self.assertEqual(lost[0].track_state, TrackStateEnum.LOST)
        before = lost[0].bbox.model_dump()

        blur = BlurService()
        frame = noise_frame(480, 640, seed=15)
        _, report = blur.render_blur(frame, lost, [], blur_style="blackout", frame_index=1)
        self.assertEqual(len(report.roi_plans), 1)
        for _ in range(ai_settings.PLATE_REDACTION_HOLD_FRAMES + 1):
            tracks = self.tracker.update_tracks([]).tracks
        held = [t for t in tracks if t.track_state == TrackStateEnum.LOST]
        if held:
            self.assertEqual(held[0].bbox.model_dump(), before,
                             "a held plate box must not drift")

        _, stale_report = blur.render_blur(frame, tracks, [], blur_style="blackout",
                                           frame_index=2)
        self.assertEqual(len(stale_report.roi_plans), 0,
                         "a stale plate box kept masking after the hold window")


class PlateApiTests(unittest.TestCase):
    def setUp(self):
        task_manager.tasks.clear()

    def test_request_accepts_license_plates(self):
        request = StartRedactionRequest(
            video_id="vid_x", target_categories=["faces", "license_plates"])
        self.assertEqual({c.value for c in request.target_categories},
                         {"faces", "license_plates"})

    def test_face_only_request_stays_backward_compatible(self):
        request = StartRedactionRequest(video_id="vid_x")
        self.assertEqual([c.value for c in request.target_categories], ["faces"])

    def test_missing_model_fails_before_a_task_is_created(self):
        # The real singleton is patched to report a missing plate model.
        original = model_manager.ensure_plate_detector
        model_manager.ensure_plate_detector = lambda: {
            "model": None, "status": "UNAVAILABLE",
            "weights_path": ai_settings.PLATE_MODEL_PATH, "detail": "missing"}
        try:
            with self.assertRaises(DetectorUnavailableException):
                DetectionService().validate_available(["license_plates"])
        finally:
            model_manager.ensure_plate_detector = original

    def test_summary_reports_plate_region_counts(self):
        task_manager.create_task("task_plates", "vid_x", "fixture.mp4",
                                 ["faces", "license_plates"], "blackout", 80)
        task_manager.get_task("task_plates").update(
            status="COMPLETED", output_verified=True,
            redaction_counts={"faces": 3, "license_plates": 5},
            metrics={"processed_frames": 10, "frames_with_detections": 8,
                     "elapsed_seconds": 4},
            verification={"frame_count": 10, "width": 640, "height": 480})
        summary = RedactService().get_task_summary("task_plates")
        self.assertEqual(summary.license_plates_redacted, 5)
        self.assertEqual(summary.faces_blurred, 3)
        self.assertEqual(summary.total_objects_count, 8)

    def test_unsupported_category_is_never_silently_accepted(self):
        with self.assertRaises(DetectorUnavailableException):
            DetectionService().validate_available(["faces", "smartwatches"])

    def test_unverified_task_cannot_expose_a_summary(self):
        task_manager.create_task("task_running", "vid_x", "fixture.mp4",
                                 ["license_plates"], "gaussian", 80)
        with self.assertRaises(SafeFrameException):
            RedactService().get_task_summary("task_running")


def _write_plate_clip(path: str, frames: int = 12) -> bool:
    """Builds a real photo-based clip: a face on the left, a car plate right.

    The car image is panned slightly to exercise a moving plate. Returns False
    when the source photos are unavailable.
    """
    if not (os.path.isfile(SAMPLE_IMAGE) and os.path.isfile(FACE_IMAGE)):
        return False
    car = cv2.imread(SAMPLE_IMAGE)
    face = cv2.imread(FACE_IMAGE)
    if car is None or face is None:
        return False
    width, height = 800, 480
    car = cv2.resize(car, (480, 320))
    face = cv2.resize(face, (300, 300))
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height))
    if not writer.isOpened():
        return False
    try:
        for index in range(frames):
            canvas = np.full((height, width, 3), 40, np.uint8)
            offset = int(round(index * 2))
            canvas[0:300, 0:300] = face
            y0, x0 = 120, 310 - offset
            canvas[y0:y0 + 320, x0:x0 + 480] = car
            writer.write(canvas)
    finally:
        writer.release()
    return True


@unittest.skipUnless(os.path.isfile(model_manager.resolve_plate_path()),
                     "NOT RUN: license-plate ONNX weights are missing "
                     "(run scripts/download_plate_model.py)")
class PlatePipelineIntegrationTests(unittest.TestCase):
    """Real detector + FFmpeg on a generated photo clip."""

    def setUp(self):
        task_manager.tasks.clear()
        if not ffmpeg_util.validate_ffmpeg().get("available"):
            self.skipTest("NOT RUN: FFmpeg is not available")
        if not (os.path.isfile(SAMPLE_IMAGE) and os.path.isfile(FACE_IMAGE)):
            self.skipTest("NOT RUN: sample photos are unavailable")
        self.temp_dir = tempfile.mkdtemp(prefix="safeframe_plates_")
        self.clip = os.path.join(self.temp_dir, "plate_clip.mp4")
        raw = os.path.join(self.temp_dir, "raw.mp4")
        if not _write_plate_clip(raw):
            self.skipTest("NOT RUN: could not build the plate clip")
        ffmpeg = ffmpeg_util.resolve_ffmpeg_path()
        build = subprocess.run(
            [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", raw,
             "-f", "lavfi", "-i", "sine=frequency=440:duration=1.2",
             "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", self.clip],
            capture_output=True, timeout=120)
        if build.returncode != 0:
            self.skipTest("NOT RUN: could not encode the plate clip")
        self.video_id = f"vid_test_plate_{uuid.uuid4().hex[:8]}"
        self.staged = os.path.join(storage_manager.base_dir, "raw_videos",
                                   f"{self.video_id}_plate_clip.mp4")
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
        task_id = f"task_test_plate_{style}_{uuid.uuid4().hex[:8]}"
        self.task_ids = getattr(self, "task_ids", []) + [task_id]
        task_manager.create_task(task_id, self.video_id, "plate_clip.mp4",
                                 categories, style, 80)
        output = os.path.join(storage_manager.base_dir, "redacted_videos",
                              f"{task_id}_protected.mp4")
        asyncio.run(video_processor_service.process_video_pipeline(
            task_id, self.video_id, categories, style, 80))
        return task_manager.get_task(task_id), output

    def test_plate_only_pipeline_produces_a_verified_output(self):
        task, _ = self._run(["license_plates"], "gaussian")
        self.assertEqual(task["status"], "COMPLETED", task.get("failure_reason"))
        self.assertTrue(task["output_verified"])
        self.assertEqual(task["metrics"]["output_frame_count"], 12)
        self.assertGreater(task["metrics"]["masks_total"], 0)
        self.assertGreater(task["redaction_counts"].get("license_plates", 0), 0)
        self.assertEqual(task["redaction_counts"].get("faces", 0), 0)

    def test_all_three_modes_redact_plates_and_verify(self):
        for style in MODES:
            with self.subTest(style=style):
                task, _ = self._run(["license_plates"], style)
                self.assertEqual(task["status"], "COMPLETED", task.get("failure_reason"))
                self.assertTrue(task["output_verified"])
                self.assertGreater(task["metrics"]["masks_total"], 0)

    def test_combined_faces_and_plates_redacts_both(self):
        task, _ = self._run(["faces", "license_plates"], "blackout")
        self.assertEqual(task["status"], "COMPLETED", task.get("failure_reason"))
        self.assertTrue(task["output_verified"])
        self.assertGreater(task["redaction_counts"].get("license_plates", 0), 0)
        self.assertGreater(task["redaction_counts"].get("faces", 0), 0,
                           "the face detector stopped working when plates were added")

    def test_face_only_regression_still_completes(self):
        task, _ = self._run(["faces"], "gaussian")
        self.assertEqual(task["status"], "COMPLETED", task.get("failure_reason"))
        self.assertTrue(task["output_verified"])
        self.assertEqual(task["redaction_counts"].get("license_plates", 0), 0)

    def test_audio_is_preserved_and_resolution_matches(self):
        task, output = self._run(["license_plates"], "pixelate")
        self.assertEqual(task["status"], "COMPLETED", task.get("failure_reason"))
        self.assertGreaterEqual(task["metrics"]["input_audio_streams"] or 0, 1)
        self.assertTrue(task["metrics"]["audio_preserved"])
        self.assertEqual((task["metrics"]["output_width"], task["metrics"]["output_height"]),
                         (800, 480))

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

    def test_plate_regions_are_actually_redacted_in_every_mode(self):
        service = DetectionService()
        for style in MODES:
            with self.subTest(style=style):
                task, output = self._run(["license_plates"], style)
                self.assertEqual(task["status"], "COMPLETED", task.get("failure_reason"))

                # Re-detect on input and protected output in lockstep. A plate is
                # a survivor only when a detection on the protected frame
                # overlaps the plate that was present in the input frame;
                # unrelated boxes are over-redaction noise, not a leak.
                input_cap = cv2.VideoCapture(self.clip)
                output_cap = cv2.VideoCapture(output)
                survivors = 0
                plates_seen = 0
                frames = 0
                try:
                    while True:
                        ok_in, frame_in = input_cap.read()
                        ok_out, frame_out = output_cap.read()
                        if not ok_in or not ok_out:
                            break
                        frames += 1
                        original_plates = service._detect_plates(frame_in, frames)
                        protected = service._detect_plates(frame_out, frames)
                        for plate in original_plates:
                            plates_seen += 1
                            if any(self._iou(plate.bbox, other.bbox) >= 0.3
                                   for other in protected):
                                survivors += 1
                finally:
                    input_cap.release()
                    output_cap.release()
                self.assertGreater(frames, 0)
                self.assertGreater(plates_seen, 0,
                                   "the clip contains no detectable plate to redact")
                self.assertEqual(
                    survivors, 0,
                    f"{style}: {survivors} of {plates_seen} plate instances were "
                    "still detectable in the protected output")


if __name__ == "__main__":
    unittest.main()
