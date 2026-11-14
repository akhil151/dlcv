"""Phase 8 - face-mask alignment, proportional padding, temporal stability and
end-to-end output integrity.

The compositor tests use deterministic synthetic frames and assert geometry
(coverage of the detected face box, proportional padding, rounded silhouette,
stability across frames). The pipeline tests run the REAL detector + encoder on
the generated validation fixtures and are skipped when a dependency is absent
(YuNet weights, FFmpeg, fixture clips) - never reported as passed.
"""
import asyncio
import math
import os
import shutil
import subprocess
import tempfile
import unittest
import uuid

import cv2
import numpy as np

from backend.config.ai_settings import ai_settings
from backend.schemas.detection import DetectionItem, NormalizedBoundingBox, TargetCategory
from backend.schemas.tracking import TrackStateEnum, TrackedObjectItem
from backend.services.ai import model_manager
from backend.services.ai.blur_service import BlurService
from backend.services.ai.tracking_service import TrackingService
from backend.services.ai.video_processor_service import video_processor_service
from backend.utils import ffmpeg as ffmpeg_util
from backend.utils.storage import storage_manager
from backend.workers.task_manager import task_manager

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE_DIR = os.path.join(PROJECT_ROOT, ".cache", "validation", "videos")

MODES = ("gaussian", "pixelate", "blackout")


def detection(x, y, w, h, conf=0.9):
    return DetectionItem(
        category=TargetCategory.FACES, label="FACE", confidence=conf,
        bbox=NormalizedBoundingBox(x=x, y=y, width=w, height=h),
    )


def tracked(bbox, track_id=1, state=TrackStateEnum.TRACKED):
    return TrackedObjectItem(
        track_id=track_id, category=TargetCategory.FACES, label=f"FACE #{track_id}",
        confidence=0.9, bbox=bbox, track_state=state, age=5,
        frames_since_update=0, is_confirmed=True,
    )


def noise_frame(height, width, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(1, 255, (height, width, 3), dtype=np.uint8)


def exact_rect(bbox, width, height):
    """Exact floating-point pixel rectangle of a normalized bbox."""
    return (bbox.x / 100.0 * width, bbox.y / 100.0 * height,
            (bbox.x + bbox.width) / 100.0 * width,
            (bbox.y + bbox.height) / 100.0 * height)


def core_rect(plan):
    return plan.core_x1, plan.core_y1, plan.core_x2, plan.core_y2


def rect_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def plan_rect(plan):
    return plan.x1, plan.y1, plan.x2, plan.y2


class AlignmentTests(unittest.TestCase):
    """The mask must be built from the detected face box, not a fixed rectangle."""

    def setUp(self):
        self.svc = BlurService()

    def test_padding_is_proportional_to_face_width_and_height(self):
        ratio_x = ai_settings.BLUR_PADDING_X_RATIO
        ratio_y = ai_settings.BLUR_PADDING_Y_RATIO
        # Square-ish, wide and tall faces in a large frame: padding must follow
        # the face's own width/height, not the smaller side.
        cases = [
            NormalizedBoundingBox(x=40.0, y=40.0, width=10.0, height=10.0),
            NormalizedBoundingBox(x=10.0, y=45.0, width=20.0, height=8.0),
            NormalizedBoundingBox(x=45.0, y=10.0, width=8.0, height=20.0),
        ]
        for bbox in cases:
            with self.subTest(bbox=bbox):
                plan = self.svc._convert_normalized_bbox_to_roi(
                    bbox, 2000, 2000, TargetCategory.FACES, False)
                fx1, fy1, fx2, fy2 = exact_rect(bbox, 2000, 2000)
                core_w = math.ceil(fx2) - math.floor(fx1)
                core_h = math.ceil(fy2) - math.floor(fy1)
                pad_left = math.floor(fx1) - plan.x1
                pad_top = math.floor(fy1) - plan.y1
                pad_right = plan.x2 - math.ceil(fx2)
                pad_bottom = plan.y2 - math.ceil(fy2)
                expected_x = max(ai_settings.BLUR_MIN_PADDING_PX,
                                 min(ai_settings.BLUR_MAX_PADDING_PX,
                                     round(core_w * ratio_x)))
                expected_y = max(ai_settings.BLUR_MIN_PADDING_PX,
                                 min(ai_settings.BLUR_MAX_PADDING_PX,
                                     round(core_h * ratio_y)))
                self.assertEqual((pad_left, pad_right), (expected_x, expected_x))
                self.assertEqual((pad_top, pad_bottom), (expected_y, expected_y))

    def test_padding_grows_with_face_size(self):
        small = self.svc._convert_normalized_bbox_to_roi(
            NormalizedBoundingBox(x=45.0, y=45.0, width=5.0, height=5.0),
            1920, 1080, TargetCategory.FACES, False)
        large = self.svc._convert_normalized_bbox_to_roi(
            NormalizedBoundingBox(x=30.0, y=20.0, width=40.0, height=40.0),
            1920, 1080, TargetCategory.FACES, False)
        small_pad = small.core_x1 - small.x1
        large_pad = large.core_x1 - large.x1
        self.assertGreater(large_pad, small_pad,
                           "a large face must not get the same absolute padding as a small one")

    def test_mask_is_centred_on_the_detected_face(self):
        bbox = NormalizedBoundingBox(x=37.3, y=21.1, width=18.4, height=26.7)
        plan = self.svc._convert_normalized_bbox_to_roi(
            bbox, 1280, 720, TargetCategory.FACES, False)
        x1, y1, x2, y2 = core_rect(plan)
        fx1, fy1, fx2, fy2 = exact_rect(bbox, 1280, 720)
        face_cx = (fx1 + fx2) / 2.0
        core_cx = (x1 + x2) / 2.0
        face_cy = (fy1 + fy2) / 2.0
        core_cy = (y1 + y2) / 2.0
        # Outward rounding may add at most half a pixel on each side.
        self.assertLessEqual(abs(face_cx - core_cx), 1.0)
        self.assertLessEqual(abs(face_cy - core_cy), 1.0)
        # And the mask may exceed the face only by the proportional padding.
        self.assertLessEqual(x1, fx1)
        self.assertLessEqual(y1, fy1)
        self.assertGreaterEqual(x2, fx2)
        self.assertGreaterEqual(y2, fy2)

    def _assert_face_core_fully_filtered(self, width, height, bbox, mode):
        frame = noise_frame(height, width, seed=4)
        original = frame.copy()
        out, report = self.svc.render_blur(
            frame, [tracked(bbox)], [], blur_style=mode,
            blur_intensity=80, frame_index=1)
        self.assertEqual(report.total_blurred_regions, 1)
        x1, y1, x2, y2 = core_rect(report.roi_plans[0])
        before = original[y1:y2, x1:x2]
        after = out[y1:y2, x1:x2]
        if mode == "blackout":
            self.assertTrue(np.all(after == 0))
        elif mode == "pixelate":
            # INTER_NEAREST keeps one representative pixel per block, so a few
            # block pixels can coincide with their original value.
            changed = (after != before).any(axis=2)
            self.assertGreaterEqual(float(changed.mean()), 0.9,
                                    "the mosaic did not cover the face core")
        else:
            changed = (after != before).any(axis=2)
            self.assertTrue(np.all(changed), f"{mode} left face pixels unmodified")

    def test_small_and_large_faces_are_fully_covered_in_every_mode(self):
        cases = [
            (1280, 720, NormalizedBoundingBox(x=49.5, y=49.5, width=2.5, height=2.5)),   # ~32 px
            (1920, 1080, NormalizedBoundingBox(x=35.0, y=20.0, width=30.0, height=33.0)),  # ~576 px
            (640, 480, NormalizedBoundingBox(x=49.5, y=49.5, width=1.0, height=1.0)),     # ~5 px
        ]
        for width, height, bbox in cases:
            for mode in MODES:
                with self.subTest(w=width, h=height, bbox=bbox, mode=mode):
                    self._assert_face_core_fully_filtered(width, height, bbox, mode)

    def test_rotated_face_boxes_are_fully_covered(self):
        # A rotated head produces a wide, short or diagonal detector box; the
        # mask must follow that geometry rather than assume an upright square.
        boxes = [
            NormalizedBoundingBox(x=30.0, y=45.0, width=18.0, height=9.0),
            NormalizedBoundingBox(x=60.0, y=12.0, width=9.0, height=18.0),
            NormalizedBoundingBox(x=41.7, y=27.9, width=19.3, height=22.4),
        ]
        for bbox in boxes:
            for mode in MODES:
                with self.subTest(bbox=bbox, mode=mode):
                    self._assert_face_core_fully_filtered(1024, 768, bbox, mode)

    def test_multiple_faces_get_independent_masks(self):
        boxes = [
            NormalizedBoundingBox(x=5.0, y=20.0, width=15.0, height=20.0),
            NormalizedBoundingBox(x=45.0, y=18.0, width=12.0, height=18.0),
            NormalizedBoundingBox(x=80.0, y=25.0, width=10.0, height=15.0),
        ]
        frame = noise_frame(480, 640, seed=6)
        for mode in MODES:
            with self.subTest(mode=mode):
                out, report = self.svc.render_blur(
                    frame, [tracked(box, i + 1) for i, box in enumerate(boxes)], [],
                    blur_style=mode, blur_intensity=80, frame_index=1)
                self.assertEqual(report.total_blurred_regions, len(boxes))
                for plan in report.roi_plans:
                    x1, y1, x2, y2 = core_rect(plan)
                    if mode == "blackout":
                        self.assertTrue(np.all(out[y1:y2, x1:x2] == 0))
                    else:
                        self.assertFalse(np.array_equal(
                            out[y1:y2, x1:x2], frame[y1:y2, x1:x2]))

    def test_mask_silhouette_is_rounded_and_stays_inside_the_roi(self):
        bbox = NormalizedBoundingBox(x=40.0, y=40.0, width=20.0, height=24.0)
        frame = noise_frame(480, 640, seed=8)
        original = frame.copy()
        out, report = self.svc.render_blur(
            frame, [tracked(bbox)], [], blur_style="gaussian",
            blur_intensity=80, frame_index=1)
        plan = report.roi_plans[0]
        # Nothing outside the planned ROI may change.
        changed = (out != original).any(axis=2)
        allowed = np.zeros((480, 640), dtype=bool)
        allowed[plan.y1:plan.y2, plan.x1:plan.x2] = True
        self.assertTrue(np.all(allowed[changed]))
        # The ROI corners are part of the padding ring, not the face: a rounded
        # mask leaves them untouched instead of painting a hard rectangle.
        corner = out[plan.y1:plan.y1 + 1, plan.x1:plan.x1 + 1]
        corner_before = original[plan.y1:plan.y1 + 1, plan.x1:plan.x1 + 1]
        self.assertTrue(np.array_equal(corner, corner_before))


class TemporalStabilityTests(unittest.TestCase):
    """The mask must follow a face without crawling, shrinking or jumping."""

    def setUp(self):
        self.tracker = TrackingService()
        self.blur = BlurService()
        self.frame = noise_frame(480, 640, seed=12)

    def _step(self, detections, mode="blackout"):
        tracks = self.tracker.update_tracks(detections).tracks
        _, report = self.blur.render_blur(
            self.frame.copy(), tracks, [], blur_style=mode, frame_index=1,
            detections=detections)
        return report.roi_plans

    def test_jittering_detection_does_not_jump_between_frames(self):
        rng = np.random.default_rng(21)
        previous = None
        ious = []
        for index in range(24):
            # Real detector jitter: a few tenths of a percent per frame.
            jitter_x = float(rng.normal(0, 0.25))
            jitter_y = float(rng.normal(0, 0.25))
            plans = self._step([detection(40.0 + jitter_x, 35.0 + jitter_y, 15.0, 20.0)])
            self.assertEqual(len(plans), 1)
            current = plan_rect(plans[0])
            if previous is not None:
                ious.append(rect_iou(previous, current))
            previous = current
        self.assertGreaterEqual(min(ious), 0.90,
                                f"mask jumped between consecutive frames: min IoU {min(ious):.3f}")

    def test_moving_face_is_never_trailed_or_left_behind(self):
        for index in range(20):
            bbox = NormalizedBoundingBox(x=10.0 + index * 2.0, y=35.0, width=14.0, height=18.0)
            plans = self._step([detection(bbox.x, bbox.y, bbox.width, bbox.height)])
            x1, y1, x2, y2 = core_rect(plans[0])
            fx1, fy1, fx2, fy2 = exact_rect(bbox, 640, 480)
            self.assertLessEqual(x1, math.floor(fx1))
            self.assertGreaterEqual(x2, math.ceil(fx2))
            self.assertLessEqual(y1, math.floor(fy1))
            self.assertGreaterEqual(y2, math.ceil(fy2))

    def test_detection_gap_then_return_restores_full_coverage(self):
        # Confirmed track, then a missed frame, then the face is detected again
        # at a new position: the mask must cover the new detection immediately.
        self._step([detection(20.0, 30.0, 15.0, 20.0)])
        self._step([detection(20.0, 30.0, 15.0, 20.0)])
        held = self._step([])
        self.assertEqual(len(held), 1, "a confirmed track should be held briefly")
        returned_bbox = NormalizedBoundingBox(x=32.0, y=31.0, width=15.0, height=20.0)
        plans = self._step([detection(returned_bbox.x, returned_bbox.y,
                                      returned_bbox.width, returned_bbox.height)])
        covered = False
        fx1, fy1, fx2, fy2 = exact_rect(returned_bbox, 640, 480)
        for plan in plans:
            x1, y1, x2, y2 = core_rect(plan)
            if x1 <= math.floor(fx1) and x2 >= math.ceil(fx2) \
                    and y1 <= math.floor(fy1) and y2 >= math.ceil(fy2):
                covered = True
        self.assertTrue(covered, "the returned detection was not fully masked")

    def test_smoothing_never_makes_the_box_smaller_than_the_detection(self):
        for index in range(10):
            bbox = NormalizedBoundingBox(x=20.0 + index * 0.3, y=30.0, width=15.0, height=20.0)
            track = self.tracker.update_tracks([detection(bbox.x, bbox.y, 15.0, 20.0)]).tracks[0]
            self.assertLessEqual(track.bbox.x, bbox.x + 1e-6)
            self.assertLessEqual(track.bbox.y, bbox.y + 1e-6)
            self.assertGreaterEqual(track.bbox.x + track.bbox.width,
                                    bbox.x + bbox.width - 1e-6)
            self.assertGreaterEqual(track.bbox.y + track.bbox.height,
                                    bbox.y + bbox.height - 1e-6)


def _stage_fixture(clip_name: str) -> tuple[str, str]:
    """Copies a fixture into storage; returns (video_id, staged_path)."""
    source = os.path.join(FIXTURE_DIR, f"{clip_name}.mp4")
    video_id = f"vid_test_{clip_name}_{uuid.uuid4().hex[:8]}"
    staged = os.path.join(storage_manager.base_dir, "raw_videos",
                          f"{video_id}_{clip_name}.mp4")
    with open(source, "rb") as src, open(staged, "wb") as dst:
        dst.write(src.read())
    return video_id, staged


def _run_pipeline(clip_name: str, style: str, intensity: int = 80) -> tuple[dict, str, str]:
    video_id, staged = _stage_fixture(clip_name)
    task_id = f"task_test_{clip_name}_{style}_{uuid.uuid4().hex[:8]}"
    task_manager.create_task(task_id, video_id, f"{clip_name}.mp4", ["faces"], style, intensity)
    asyncio.run(video_processor_service.process_video_pipeline(
        task_id, video_id, ["faces"], style, intensity))
    task = task_manager.get_task(task_id)
    output = os.path.join(storage_manager.base_dir, "redacted_videos", f"{task_id}_protected.mp4")
    return task, staged, output


def _decode_count_and_size(path: str) -> tuple[int, int, int, float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0, 0, 0, 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        frames += 1
    cap.release()
    return frames, width, height, fps


@unittest.skipUnless(os.path.isdir(FIXTURE_DIR),
                     "NOT RUN: validation fixtures are missing (run scripts/make_validation_videos.py)")
class PipelineIntegrationTests(unittest.TestCase):
    """Real detector + real encoder on the generated fixtures."""

    def setUp(self):
        task_manager.tasks.clear()
        if not ffmpeg_util.validate_ffmpeg().get("available"):
            self.skipTest("NOT RUN: FFmpeg is not available")
        resolved = model_manager.resolve_yunet_path()
        if not (os.path.isfile(resolved) and os.path.getsize(resolved) > 0):
            self.skipTest("NOT RUN: YuNet ONNX weights are not available")

    @staticmethod
    def _cleanup(*paths):
        for path in paths:
            try:
                if path and os.path.isfile(path):
                    os.remove(path)
            except OSError:
                pass

    def test_no_face_fixture_masks_nothing_and_still_verifies(self):
        if not os.path.isfile(os.path.join(FIXTURE_DIR, "no_faces.mp4")):
            self.skipTest("NOT RUN: no_faces fixture missing")
        task, staged, output = _run_pipeline("no_faces", "gaussian")
        try:
            self.assertEqual(task["status"], "COMPLETED")
            self.assertTrue(task["output_verified"])
            self.assertEqual(task["metrics"]["masks_total"], 0,
                             "masks were drawn on a clip without faces")
            self.assertEqual(task["metrics"]["frames_with_detections"], 0)
            self.assertEqual(task["metrics"]["output_frame_count"], 48)
        finally:
            self._cleanup(staged, output)

    def test_all_three_modes_produce_a_verified_output(self):
        if not os.path.isfile(os.path.join(FIXTURE_DIR, "frontal_single.mp4")):
            self.skipTest("NOT RUN: frontal_single fixture missing")
        input_frames, in_w, in_h, _ = _decode_count_and_size(
            os.path.join(FIXTURE_DIR, "frontal_single.mp4"))
        for style in MODES:
            with self.subTest(style=style):
                task, staged, output = _run_pipeline("frontal_single", style)
                try:
                    self.assertEqual(task["status"], "COMPLETED", task.get("failure_reason"))
                    self.assertTrue(task["output_verified"])
                    metrics = task["metrics"]
                    self.assertEqual(metrics["output_frame_count"], input_frames)
                    self.assertEqual(metrics["output_width"], in_w)
                    self.assertEqual(metrics["output_height"], in_h)
                    self.assertGreaterEqual(metrics["masks_total"],
                                            metrics["frames_with_detections"],
                                            "a frame with faces produced no mask")
                    self.assertGreater(os.path.getsize(output), 0)
                finally:
                    self._cleanup(staged, output)

    def test_audio_stream_is_preserved_and_frames_match(self):
        ffmpeg = ffmpeg_util.resolve_ffmpeg_path()
        if not ffmpeg:
            self.skipTest("NOT RUN: FFmpeg is not available")
        temp_dir = tempfile.mkdtemp(prefix="safeframe_audio_")
        clip_path = os.path.join(temp_dir, "audio_clip.mp4")
        build = subprocess.run(
            [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=10:duration=2",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
             "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", clip_path],
            capture_output=True, timeout=120)
        if build.returncode != 0:
            self.skipTest("NOT RUN: could not generate an audio test clip")
        input_frames, in_w, in_h, _ = _decode_count_and_size(clip_path)

        video_id = f"vid_test_audio_{uuid.uuid4().hex[:8]}"
        staged = os.path.join(storage_manager.base_dir, "raw_videos",
                              f"{video_id}_audio_clip.mp4")
        shutil.copyfile(clip_path, staged)
        task_id = f"task_test_audio_{uuid.uuid4().hex[:8]}"
        task_manager.create_task(task_id, video_id, "audio_clip.mp4", ["faces"], "gaussian", 80)
        output = os.path.join(storage_manager.base_dir, "redacted_videos",
                              f"{task_id}_protected.mp4")
        try:
            asyncio.run(video_processor_service.process_video_pipeline(
                task_id, video_id, ["faces"], "gaussian", 80))
            task = task_manager.get_task(task_id)
            self.assertEqual(task["status"], "COMPLETED", task.get("failure_reason"))
            self.assertTrue(task["output_verified"])
            self.assertGreaterEqual(task["metrics"]["input_audio_streams"] or 0, 1,
                                    "the generated clip should contain audio")
            self.assertTrue(task["metrics"]["audio_preserved"],
                            "audio was dropped from the protected video")
            self.assertGreaterEqual(task["metrics"]["output_audio_streams"] or 0, 1)
            out_frames, out_w, out_h, _ = _decode_count_and_size(output)
            self.assertEqual(out_frames, input_frames)
            self.assertEqual((out_w, out_h), (in_w, in_h))
        finally:
            self._cleanup(staged, output)
            try:
                os.rmdir(temp_dir)
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main()
