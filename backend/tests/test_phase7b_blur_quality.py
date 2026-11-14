"""Phase 7B - blur compositor quality and coverage tests.

These tests exercise the ROI compositor directly with deterministic synthetic
frames. They assert geometric coverage (the whole sensitive rectangle is
redacted), strength floors, safe edge handling and fail-closed behavior.

They deliberately do NOT claim real-world privacy effectiveness from synthetic
images; they only pin down compositor behavior.
"""
import math
import unittest

import numpy as np

from backend.config.ai_settings import ai_settings
from backend.schemas.blur import BlurModeEnum
from backend.schemas.detection import NormalizedBoundingBox, TargetCategory
from backend.schemas.tracking import TrackedObjectItem, TrackStateEnum
from backend.services.ai.blur_service import BlurService

MODES = ("gaussian", "pixelate", "blackout")


def track(bbox, track_id=1):
    return TrackedObjectItem(
        track_id=track_id,
        category=TargetCategory.FACES,
        label="FACE",
        confidence=0.9,
        bbox=bbox,
        track_state=TrackStateEnum.TRACKED,
        age=5,
        frames_since_update=0,
        is_confirmed=True,
    )


def true_interior_rect(bbox, width, height):
    """Integer pixel rect strictly inside the exact normalized rectangle."""
    fx1 = bbox.x / 100.0 * width
    fy1 = bbox.y / 100.0 * height
    fx2 = (bbox.x + bbox.width) / 100.0 * width
    fy2 = (bbox.y + bbox.height) / 100.0 * height
    return math.ceil(fx1), math.ceil(fy1), math.floor(fx2), math.floor(fy2)


def noise_frame(height, width, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(1, 255, (height, width, 3), dtype=np.uint8)


def core_rect(plan):
    """The un-padded face rectangle of an ROI plan (must always be filtered)."""
    return plan.core_x1, plan.core_y1, plan.core_x2, plan.core_y2


# Boxes whose fractional pixel bounds expose truncation/shortfall behavior.
FRACTIONAL_CASES = [
    (101, 77, NormalizedBoundingBox(x=10.5, y=10.5, width=10.5, height=10.5)),
    (101, 101, NormalizedBoundingBox(x=33.3, y=33.3, width=33.3, height=33.3)),
    (640, 480, NormalizedBoundingBox(x=41.7, y=27.9, width=19.3, height=22.4)),
    (1920, 1080, NormalizedBoundingBox(x=12.9, y=8.3, width=7.4, height=9.1)),
]


class CoordinateCoverageTests(unittest.TestCase):
    def setUp(self):
        self.svc = BlurService()

    def test_roi_contains_the_exact_normalized_rectangle(self):
        for width, height, bbox in FRACTIONAL_CASES:
            with self.subTest(w=width, h=height, bbox=bbox):
                plan = self.svc._convert_normalized_bbox_to_roi(
                    bbox, width, height, TargetCategory.FACES, False)
                self.assertIsNotNone(plan)
                ix1, iy1, ix2, iy2 = true_interior_rect(bbox, width, height)
                self.assertLessEqual(plan.x1, ix1)
                self.assertLessEqual(plan.y1, iy1)
                self.assertGreaterEqual(plan.x2, ix2)
                self.assertGreaterEqual(plan.y2, iy2)

    def test_roi_is_padded_outward_beyond_the_target(self):
        for width, height, bbox in FRACTIONAL_CASES:
            with self.subTest(w=width, h=height, bbox=bbox):
                plan = self.svc._convert_normalized_bbox_to_roi(
                    bbox, width, height, TargetCategory.FACES, False)
                ix1, iy1, ix2, iy2 = true_interior_rect(bbox, width, height)
                self.assertLess(plan.x1, ix1)
                self.assertLess(plan.y1, iy1)
                self.assertGreater(plan.x2, ix2)
                self.assertGreater(plan.y2, iy2)

    def test_roi_never_exceeds_frame_bounds(self):
        for width, height, bbox in FRACTIONAL_CASES + [
            (101, 77, NormalizedBoundingBox(x=0, y=0, width=100, height=100)),
            (101, 77, NormalizedBoundingBox(x=95, y=95, width=5, height=5)),
        ]:
            plan = self.svc._convert_normalized_bbox_to_roi(
                bbox, width, height, TargetCategory.FACES, False)
            self.assertIsNotNone(plan)
            self.assertGreaterEqual(plan.x1, 0)
            self.assertGreaterEqual(plan.y1, 0)
            self.assertLessEqual(plan.x2, width)
            self.assertLessEqual(plan.y2, height)


class RegionCoverageTests(unittest.TestCase):
    """Every pixel of the target rectangle must be modified by every mode."""

    def setUp(self):
        self.svc = BlurService()

    def _assert_rect_fully_redacted(self, width, height, bbox, mode, intensity):
        frame = noise_frame(height, width, seed=7)
        original = frame.copy()
        out, report = self.svc.render_blur(
            frame, [track(bbox)], [], blur_style=mode,
            blur_intensity=intensity, frame_index=1)
        self.assertEqual(report.total_blurred_regions, 1)
        ix1, iy1, ix2, iy2 = true_interior_rect(bbox, width, height)
        self.assertGreater(ix2, ix1)
        self.assertGreater(iy2, iy1)
        target_before = original[iy1:iy2, ix1:ix2]
        target_after = out[iy1:iy2, ix1:ix2]
        if mode == "blackout":
            self.assertTrue(np.all(target_after == 0))
        elif mode == "pixelate":
            # INTER_NEAREST keeps one representative pixel per block, so allow a
            # small fraction of coincidental matches; the grid itself is opaque.
            changed = (target_after != target_before).any(axis=2)
            self.assertGreaterEqual(float(changed.mean()), 0.9)
        else:
            changed = (target_after != target_before).any(axis=2)
            self.assertTrue(np.all(changed),
                            f"{mode} left {int((~changed).sum())} target pixels unmodified")

    def test_all_modes_cover_fractional_targets_at_multiple_resolutions(self):
        for width, height, bbox in FRACTIONAL_CASES:
            for mode in MODES:
                with self.subTest(w=width, h=height, bbox=bbox, mode=mode):
                    self._assert_rect_fully_redacted(width, height, bbox, mode, 80)

    def test_small_target_is_fully_redacted(self):
        bbox = NormalizedBoundingBox(x=50.5, y=50.5, width=1.5, height=1.5)
        for mode in MODES:
            with self.subTest(mode=mode):
                self._assert_rect_fully_redacted(640, 480, bbox, mode, 80)

    def test_targets_touching_each_boundary_are_fully_redacted(self):
        boxes = [
            NormalizedBoundingBox(x=0.0, y=40.0, width=8.0, height=20.0),
            NormalizedBoundingBox(x=92.0, y=40.0, width=8.0, height=20.0),
            NormalizedBoundingBox(x=40.0, y=0.0, width=20.0, height=8.0),
            NormalizedBoundingBox(x=40.0, y=92.0, width=20.0, height=8.0),
        ]
        for bbox in boxes:
            for mode in MODES:
                with self.subTest(bbox=bbox, mode=mode):
                    self._assert_rect_fully_redacted(640, 480, bbox, mode, 80)

    def test_overlapping_targets_are_both_covered(self):
        a = NormalizedBoundingBox(x=30.0, y=30.0, width=20.0, height=20.0)
        b = NormalizedBoundingBox(x=40.0, y=40.0, width=20.0, height=20.0)
        frame = noise_frame(480, 640, seed=3)
        original = frame.copy()
        out, report = self.svc.render_blur(
            frame, [track(a, 1), track(b, 2)], [], blur_style="blackout",
            blur_intensity=80, frame_index=1)
        self.assertEqual(report.total_blurred_regions, 2)
        for bbox in (a, b):
            ix1, iy1, ix2, iy2 = true_interior_rect(bbox, 640, 480)
            self.assertTrue(np.all(out[iy1:iy2, ix1:ix2] == 0))
        del original

    def test_no_pixels_outside_planned_roi_are_modified(self):
        bbox = NormalizedBoundingBox(x=33.3, y=33.3, width=20.0, height=20.0)
        frame = noise_frame(480, 640, seed=11)
        original = frame.copy()
        out, report = self.svc.render_blur(
            frame, [track(bbox)], [], blur_style="gaussian",
            blur_intensity=80, frame_index=1)
        channelled = np.zeros((480, 640), dtype=bool)
        for plan in report.roi_plans:
            channelled[plan.y1:plan.y2, plan.x1:plan.x2] = True
        changed = (out != original).any(axis=2)
        self.assertTrue(np.all(channelled[changed]))


class ModeBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.svc = BlurService()

    def test_blackout_is_fully_opaque_over_the_face(self):
        bbox = NormalizedBoundingBox(x=40.0, y=40.0, width=20.0, height=20.0)
        frame = noise_frame(480, 640, seed=5)
        out, report = self.svc.render_blur(
            frame, [track(bbox)], [], blur_style="blackout",
            blur_intensity=80, frame_index=1)
        plan = report.roi_plans[0]
        x1, y1, x2, y2 = core_rect(plan)
        # The detected face itself is completely opaque...
        self.assertTrue(np.all(out[y1:y2, x1:x2] == 0))
        # ...while the rounded silhouette leaves the padding ring's corners
        # alone instead of painting a hard rectangle over hair/background.
        self.assertTrue(np.any(out[plan.y1:plan.y2, plan.x1:plan.x2] != 0))

    def test_gaussian_has_a_privacy_strength_floor_at_low_intensity(self):
        # A 1px bright line must be spread out even at the lowest intensity.
        bbox = NormalizedBoundingBox(x=40.0, y=40.0, width=20.0, height=20.0)
        frame = np.zeros((480, 640, 3), np.uint8)
        frame[236:244, 320] = 255
        out, _ = self.svc.render_blur(
            frame, [track(bbox)], [], blur_style="gaussian",
            blur_intensity=10, frame_index=1)
        # A sigma ~1 Gaussian keeps this ~1px line near full brightness (~90+);
        # the privacy floor (sigma >= 4) spreads it to ~25.
        self.assertLess(int(out.max()), 64,
                        "gaussian collapsed to a near-identity filter")

    def test_pixelate_blocks_have_a_minimum_size(self):
        bbox = NormalizedBoundingBox(x=40.0, y=40.0, width=20.0, height=20.0)
        frame = noise_frame(480, 640, seed=9)
        out, report = self.svc.render_blur(
            frame, [track(bbox)], [], blur_style="pixelate",
            blur_intensity=10, frame_index=1)
        plan = report.roi_plans[0]
        x1, y1, x2, y2 = core_rect(plan)
        # Measure inside the face core: the outer rim is deliberately feathered,
        # so per-pixel block runs are only meaningful over the face itself.
        row = out[(y1 + y2) // 2, x1:x2, 0]
        changes = np.flatnonzero(np.diff(row.astype(int)) != 0)
        runs = np.diff(np.concatenate(([-1], changes, [len(row) - 1])))
        # The first/last runs may be a partial block cut by the core edge.
        middle_runs = runs[1:-1] if len(runs) > 2 else runs
        self.assertGreaterEqual(int(np.median(middle_runs)), ai_settings.PIXELATE_MIN_BLOCK_PX)

    def test_pixelate_bounds_the_mosaic_grid(self):
        bbox = NormalizedBoundingBox(x=20.0, y=20.0, width=40.0, height=40.0)
        frame = noise_frame(1080, 1920, seed=13)
        out, report = self.svc.render_blur(
            frame, [track(bbox)], [], blur_style="pixelate",
            blur_intensity=50, frame_index=1)
        plan = report.roi_plans[0]
        x1, y1, x2, y2 = core_rect(plan)
        # Scan the shorter side of the face core: the mosaic grid is bounded
        # against the target's smaller dimension, independent of resolution.
        if (x2 - x1) <= (y2 - y1):
            line = out[(y1 + y2) // 2, x1:x2, 0]
        else:
            line = out[y1:y2, (x1 + x2) // 2, 0]
        changes = np.flatnonzero(np.diff(line.astype(int)) != 0)
        blocks_across = len(changes) + 1
        self.assertLessEqual(blocks_across, ai_settings.PIXELATE_MAX_BLOCKS)

    def test_feathered_never_restores_original_pixels_inside_target(self):
        bbox = NormalizedBoundingBox(x=40.0, y=40.0, width=20.0, height=20.0)
        frame = noise_frame(480, 640, seed=17)
        original = frame.copy()
        out, _ = self.svc.render_blur(
            frame, [track(bbox)], [], blur_style="feathered",
            blur_intensity=80, frame_index=1)
        ix1, iy1, ix2, iy2 = true_interior_rect(bbox, 640, 480)
        untouched = int((out[iy1:iy2, ix1:ix2] == original[iy1:iy2, ix1:ix2])
                        .all(axis=2).sum())
        self.assertEqual(untouched, 0)

    def test_unknown_mode_fails_closed_and_still_redacts(self):
        bbox = NormalizedBoundingBox(x=40.0, y=40.0, width=20.0, height=20.0)
        frame = noise_frame(480, 640, seed=19)
        out, report = self.svc.render_blur(
            frame, [track(bbox)], [], blur_style="not-a-mode",
            blur_intensity=80, frame_index=1)
        self.assertEqual(report.blur_mode, BlurModeEnum.BLACKOUT)
        ix1, iy1, ix2, iy2 = true_interior_rect(bbox, 640, 480)
        self.assertTrue(np.all(out[iy1:iy2, ix1:ix2] == 0))


class TemporalTests(unittest.TestCase):
    """Compositor output must be stable and correctly positioned frame to frame.

    This exercises compositor behavior only; it does not test the tracker or
    ghost-box algorithms, whose outputs are taken as given.
    """

    def setUp(self):
        self.svc = BlurService()

    def test_drifting_target_is_fully_covered_in_every_frame(self):
        width, height = 640, 480
        for index in range(12):
            x = 20.0 + index * 3.0
            bbox = NormalizedBoundingBox(x=x, y=40.0, width=15.0, height=20.0)
            frame = noise_frame(height, width, seed=index)
            out, report = self.svc.render_blur(
                frame, [track(bbox)], [], blur_style="blackout",
                blur_intensity=80, frame_index=index)
            self.assertEqual(report.total_blurred_regions, 1)
            ix1, iy1, ix2, iy2 = true_interior_rect(bbox, width, height)
            self.assertTrue(np.all(out[iy1:iy2, ix1:ix2] == 0),
                            f"target not covered on frame {index}")
            plan = report.roi_plans[0]
            # The mask must sit on the target, not lag behind it.
            self.assertLessEqual(plan.x1, ix1)
            self.assertGreaterEqual(plan.x2, ix2)

    def test_lost_target_leaves_no_residual_mask(self):
        width, height = 640, 480
        bbox = NormalizedBoundingBox(x=40.0, y=40.0, width=15.0, height=20.0)
        frame = noise_frame(height, width, seed=31)
        out, report = self.svc.render_blur(
            frame, [], [], blur_style="blackout", blur_intensity=80, frame_index=1)
        self.assertEqual(report.total_blurred_regions, 0)
        self.assertTrue(np.array_equal(out, frame))


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.svc = BlurService()

    def test_empty_frame_returns_safe_buffer_and_telemetry(self):
        out, report = self.svc.render_blur(None, [], [], blur_style="gaussian", frame_index=1)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(report.total_blurred_regions, 0)

    def test_zero_size_box_does_not_corrupt_or_crash(self):
        bbox = NormalizedBoundingBox(x=100.0, y=100.0, width=0.0, height=0.0)
        frame = noise_frame(480, 640, seed=23)
        out, _ = self.svc.render_blur(
            frame, [track(bbox)], [], blur_style="blackout",
            blur_intensity=80, frame_index=1)
        self.assertEqual(out.shape, frame.shape)
        self.assertEqual(out.dtype, frame.dtype)

    def test_grayscale_frames_do_not_crash_in_any_mode(self):
        gray = np.zeros((120, 160), np.uint8)
        gray[50:80, 60:100] = 255
        bbox = NormalizedBoundingBox(x=37.5, y=41.6, width=25.0, height=25.0)
        for mode in modes_with_feathered():
            with self.subTest(mode=mode):
                out, report = self.svc.render_blur(
                    gray.copy(), [track(bbox)], [], blur_style=mode,
                    blur_intensity=80, frame_index=1)
                self.assertEqual(out.shape, gray.shape)
                self.assertEqual(report.total_blurred_regions, 1)

    def test_deterministic_output_for_identical_input(self):
        bbox = NormalizedBoundingBox(x=40.0, y=40.0, width=20.0, height=20.0)
        frame = noise_frame(480, 640, seed=29)
        first, _ = self.svc.render_blur(frame.copy(), [track(bbox)], [], blur_style="gaussian", frame_index=1)
        second, _ = self.svc.render_blur(frame.copy(), [track(bbox)], [], blur_style="gaussian", frame_index=1)
        self.assertTrue(np.array_equal(first, second))


def modes_with_feathered():
    return MODES + ("feathered",)


if __name__ == "__main__":
    unittest.main()
