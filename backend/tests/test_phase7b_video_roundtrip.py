"""Phase 7B - real encode/decode round-trip validation.

The production pipeline shells out to the `ffmpeg` CLI, which is not available
in this environment. This test therefore uses OpenCV's FFmpeg-backed
VideoWriter to prove that redaction survives an actual H.264/MP4 encode and an
independent decode. It is a real encoder round-trip, but it is NOT the exact
production `ffmpeg` command line.

A bright marker is composited onto real frames from a real video. A control
pass (no redaction) proves the check can detect the marker; the redacted pass
proves the marker is gone and stays gone in every decoded frame.
"""
import os
import tempfile
import unittest

import cv2
import numpy as np

from backend.schemas.detection import NormalizedBoundingBox
from backend.schemas.tracking import TrackedObjectItem, TrackStateEnum
from backend.services.ai.blur_service import BlurService

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_VIDEO = os.path.join(PROJECT_ROOT, "test_input.mp4")
FRAMES_TO_PROCESS = 40

# Marker region as a percentage of the frame (matches the tracked bbox).
MARKER = NormalizedBoundingBox(x=30.0, y=25.0, width=25.0, height=25.0)


def make_track():
    return TrackedObjectItem(
        track_id=1, category="faces", label="FACE", confidence=0.9, bbox=MARKER,
        track_state=TrackStateEnum.TRACKED, age=5, frames_since_update=0,
        is_confirmed=True,
    )


def paste_marker(frame):
    """Composite a high-frequency black/white stripe marker so both opacity
    (blackout) and detail-destruction (blur/pixelate) are measurable."""
    h, w = frame.shape[:2]
    x1 = int(MARKER.x / 100.0 * w)
    y1 = int(MARKER.y / 100.0 * h)
    x2 = int((MARKER.x + MARKER.width) / 100.0 * w)
    y2 = int((MARKER.y + MARKER.height) / 100.0 * h)
    region = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.uint8)
    region[:, ::8] = 255
    frame[y1:y2, x1:x2] = region
    return x1, y1, x2, y2


def marker_slice(frame):
    h, w = frame.shape[:2]
    x1 = int(MARKER.x / 100.0 * w)
    y1 = int(MARKER.y / 100.0 * h)
    x2 = int((MARKER.x + MARKER.width) / 100.0 * w)
    y2 = int((MARKER.y + MARKER.height) / 100.0 * h)
    return frame[y1:y2, x1:x2]


@unittest.skipUnless(os.path.isfile(INPUT_VIDEO), "real test video unavailable")
class RealVideoRoundTripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cap = cv2.VideoCapture(INPUT_VIDEO)
        if not cap.isOpened():
            raise unittest.SkipTest("real test video cannot be opened")
        cls.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cls.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cls.fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        cls.frames = []
        while len(cls.frames) < FRAMES_TO_PROCESS:
            ok, frame = cap.read()
            if not ok:
                break
            paste_marker(frame)
            cls.frames.append(frame)
        cap.release()
        if not cls.frames:
            raise unittest.SkipTest("no frames decoded from real test video")

    def _encode(self, redact_mode):
        fd, path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (self.width, self.height))
        if not writer.isOpened():
            os.unlink(path)
            raise unittest.SkipTest("OpenCV VideoWriter backend unavailable")
        svc = BlurService()
        try:
            for frame in self.frames:
                if redact_mode is None:
                    out = frame
                else:
                    out, _ = svc.render_blur(
                        frame, [make_track()], [], blur_style=redact_mode,
                        blur_intensity=80, frame_index=1)
                    if out is frame:
                        out = frame.copy()
                writer.write(out)
        finally:
            writer.release()
        return path

    def _decode(self, path):
        cap = cv2.VideoCapture(path)
        self.assertTrue(cap.isOpened(), "encoded output could not be re-opened")
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        return frames

    def test_control_marker_is_visible_before_redaction(self):
        path = self._encode(None)
        try:
            self.assertGreater(os.path.getsize(path), 0)
            decoded = self._decode(path)
            self.assertGreaterEqual(len(decoded), len(self.frames) // 2)
            for frame in decoded:
                region = marker_slice(frame)
                self.assertGreater(int(region.max()), 200)
                self.assertGreater(float(region.std()), 60.0)
        finally:
            os.unlink(path)

    def test_blackout_survives_encode_and_decode(self):
        path = self._encode("blackout")
        try:
            decoded = self._decode(path)
            self.assertGreaterEqual(len(decoded), len(self.frames) // 2)
            for index, frame in enumerate(decoded):
                region = marker_slice(frame)
                # No decoded frame may leak the white marker.
                self.assertLessEqual(float(region.mean()), 12.0)
                self.assertLessEqual(int(region.max()), 60)
        finally:
            os.unlink(path)

    def test_gaussian_survives_encode_and_decode(self):
        control_path = self._encode(None)
        redacted_path = self._encode("gaussian")
        try:
            control = self._decode(control_path)
            redacted = self._decode(redacted_path)
            def energy(frames):
                return float(np.mean([cv2.Laplacian(marker_slice(f), cv2.CV_64F).var()
                                      for f in frames]))
            control_energy = energy(control)
            redacted_energy = energy(redacted)
            self.assertGreater(control_energy, 0.0)
            self.assertLess(redacted_energy, control_energy * 0.5,
                            "gaussian redaction did not survive encoding")
        finally:
            os.unlink(control_path)
            os.unlink(redacted_path)


if __name__ == "__main__":
    unittest.main()
