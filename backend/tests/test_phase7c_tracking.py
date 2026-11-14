"""Regression tests for the lightweight IoU tracker and the redaction hold.

These cover the two failure modes that matter for face redaction during a live
demo:

* two faces crossing in front of each other must not exchange identities,
* a face briefly missed by the detector may stay masked for a short hold, then
  its box must disappear instead of lingering (or growing) indefinitely.

Coverage itself never depends on the tracker: BlurService masks every detection
of the current frame directly (see test_phase7a_privacy).
"""
import unittest

import numpy as np

from backend.config.ai_settings import ai_settings
from backend.schemas.detection import DetectionItem, NormalizedBoundingBox, TargetCategory
from backend.schemas.tracking import TrackStateEnum
from backend.services.ai.blur_service import BlurService
from backend.services.ai.tracking_service import TrackingService


def face(x: float, y: float = 40.0, w: float = 20.0, h: float = 20.0, conf: float = 0.9) -> DetectionItem:
    return DetectionItem(
        category=TargetCategory.FACES,
        label="FACE",
        confidence=conf,
        bbox=NormalizedBoundingBox(x=x, y=y, width=w, height=h),
    )


class TracingTracker:
    """Runs frames through the real tracker and records track ids per box."""

    def __init__(self) -> None:
        self.tracker = TrackingService()

    def step(self, detections):
        result = self.tracker.update_tracks(detections)
        return {track.track_id: track for track in result.tracks}

    @staticmethod
    def id_at(tracks, x: float):
        """Returns the track id whose box is nearest to the given x centre."""
        closest = min(tracks.values(), key=lambda t: abs((t.bbox.x + t.bbox.width / 2) - x))
        return closest.track_id, closest.bbox.x


class CrossingFacesTests(unittest.TestCase):
    def test_crossing_faces_keep_their_own_ids(self):
        runner = TracingTracker()
        # Two faces passing each other, slightly offset vertically so this is a
        # real occlusion crossing rather than a perfect overlap.
        left_x, right_x = 10.0, 55.0
        history: dict[int, list[float]] = {}
        ids_seen = set()

        for _ in range(12):
            tracks = runner.step([face(left_x, y=30), face(right_x, y=45)])
            ids_seen.update(tracks.keys())
            for track_id, track in tracks.items():
                history.setdefault(track_id, []).append(track.bbox.x)
            left_x += 3.0
            right_x -= 3.0

        self.assertEqual(len(ids_seen), 2, f"fragmented into {len(ids_seen)} ids during a cross")
        for track_id, xs in history.items():
            direction = 1 if xs[-1] > xs[0] else -1
            steps = [b - a for a, b in zip(xs, xs[1:])]
            # A single id must follow one face: no reversal, no jump to the
            # other face's trajectory.
            self.assertTrue(all(step * direction >= 0 for step in steps),
                            f"track {track_id} reversed direction: {xs}")
            self.assertTrue(all(abs(step) <= 3.5 for step in steps),
                            f"track {track_id} jumped: {xs}")

    def test_nearby_but_separate_faces_are_not_merged(self):
        runner = TracingTracker()
        tracks = runner.step([face(10), face(35)])
        self.assertEqual(len(tracks), 2)
        second = runner.step([face(10), face(35)])
        self.assertEqual(len(second), 2, "a second detection must not be swallowed by an existing track")
        self.assertEqual(len(runner.tracker.active_tracks), 2)


class DetectionGapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = TrackingService()
        self.blur = BlurService()
        self.frame = np.full((100, 100, 3), 255, dtype=np.uint8)

    def _run(self, detections):
        result = self.tracker.update_tracks(detections)
        protected, report = self.blur.render_blur(
            self.frame.copy(), result.tracks, [], blur_style="blackout", frame_index=1)
        return result, report, protected

    def test_confirmed_track_is_held_for_a_short_gap_then_released(self):
        self._run([face(20)])
        _, _, _ = self._run([face(20)])  # confirmed after the second hit

        # Gap frame 1 and 2: the last known box stays masked.
        for gap in (1, 2):
            result, _report, protected = self._run([])
            track = next(iter(result.tracks))
            self.assertEqual(track.track_state, TrackStateEnum.LOST)
            self.assertEqual(track.frames_since_update, gap)
            self.assertTrue(np.all(protected[40:60, 20:40] == 0),
                            f"gap {gap} should still be held")

        # Gap frame 3: beyond REDACTION_HOLD_FRAMES, nothing may remain masked.
        _result, report, protected = self._run([])
        self.assertEqual(report.total_blurred_regions, 0)
        self.assertTrue(np.array_equal(protected, self.frame), "an expired hold still altered the frame")

    def test_lost_track_is_removed_after_the_buffer_window(self):
        self._run([face(20)])
        for _ in range(ai_settings.TRACK_BUFFER_FRAMES + 1):
            self._run([])
        self.assertEqual(len(self.tracker.active_tracks), 0, "a stale track was never expired")

    def test_detection_gap_does_not_create_a_duplicate_track(self):
        self._run([face(20)])
        self._run([face(20)])
        self._run([])
        result, _, _ = self._run([face(20)])
        self.assertEqual(len(result.tracks), 1, "re-detection after a one-frame gap created a new id")
        self.assertEqual(result.new_tracks_count, 0)

    def test_hold_box_does_not_grow(self):
        self._run([face(20)])
        self._run([face(20)])
        first = next(iter(self.tracker.update_tracks([]).tracks))
        second = next(iter(self.tracker.update_tracks([]).tracks))
        self.assertEqual((first.bbox.x, first.bbox.y, first.bbox.width, first.bbox.height),
                         (second.bbox.x, second.bbox.y, second.bbox.width, second.bbox.height),
                         "the held box moved or expanded without a validated motion estimate")


class FragmentationTests(unittest.TestCase):
    def test_stable_face_across_a_short_clip_yields_one_track(self):
        tracker = TrackingService()
        for index in range(30):
            # A small amount of detector jitter, as a real detector produces.
            jitter = 0.4 if index % 2 else -0.4
            result = tracker.update_tracks([face(20 + jitter)])
        self.assertEqual(tracker.total_new_tracks, 1)
        self.assertEqual(result.new_tracks_count, 0)


if __name__ == "__main__":
    unittest.main()
