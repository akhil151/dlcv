import asyncio
import os
import tempfile
import unittest

import cv2
import numpy as np

from backend.core.exceptions import DetectorInferenceException, DetectorUnavailableException, SafeFrameException, TaskNotFoundException, VideoNotFoundException
from backend.schemas.detection import DetectionItem, NormalizedBoundingBox, TargetCategory
from backend.schemas.tracking import TrackStateEnum, TrackedObjectItem
from backend.services.ai.blur_service import BlurService
from backend.services.ai.detection_service import DetectionService
from backend.services.ai.tracking_service import TrackingService
from backend.services.ai.video_processor_service import VideoProcessorService
from backend.services.redact_service import RedactService
from backend.schemas.redact import StartRedactionRequest
from backend.api.v1.endpoints.videos import download_video
from backend.workers.task_manager import TaskManager, task_manager


def face_detection():
    return DetectionItem(category=TargetCategory.FACES, label="FACE", confidence=.7,
        bbox=NormalizedBoundingBox(x=20, y=20, width=50, height=50))


class EmptyFaceDetector:
    def setInputSize(self, size):
        self.size = size
    def detect(self, frame):
        return None, None


class FailingFaceDetector:
    def setInputSize(self, size):
        pass
    def detect(self, frame):
        raise RuntimeError("inference boom")


class FakeModels:
    def __init__(self, detector, status="LOADED"):
        self.detector, self.status = detector, status
    def get_model(self, name):
        return {"model": self.detector, "status": self.status}


class Phase7APrivacyTests(unittest.TestCase):
    def setUp(self):
        task_manager.tasks.clear()

    def test_first_frame_new_track_is_masked(self):
        frame = np.full((20, 20, 3), 255, dtype=np.uint8)
        new_track = TrackedObjectItem(track_id=1, category=TargetCategory.FACES, label="FACE #1", confidence=.7,
            bbox=NormalizedBoundingBox(x=25, y=25, width=50, height=50), track_state=TrackStateEnum.NEW,
            age=1, frames_since_update=0, is_confirmed=False)
        result, report = BlurService().render_blur(frame, [new_track], [], blur_style="blackout", frame_index=1)
        self.assertEqual(report.active_tracks_blurred, 1)
        self.assertTrue(np.all(result[5:15, 5:15] == 0))

    def test_genuine_empty_detector_response_is_empty_not_fabricated(self):
        service = DetectionService()
        service.model_manager = FakeModels(EmptyFaceDetector())
        result = service.detect_entities(np.zeros((10, 10, 3), dtype=np.uint8), ["faces"])
        self.assertEqual(result.total_detections, 0)

    def test_detector_init_and_inference_failures_are_distinct(self):
        service = DetectionService()
        service.model_manager = FakeModels(None, "UNAVAILABLE")
        with self.assertRaises(DetectorUnavailableException):
            service.detect_entities(np.zeros((10, 10, 3), dtype=np.uint8), ["faces"])
        service.model_manager = FakeModels(FailingFaceDetector())
        with self.assertRaises(DetectorInferenceException):
            service.detect_entities(np.zeros((10, 10, 3), dtype=np.uint8), ["faces"])

    def test_unsupported_targets_are_rejected(self):
        # Unvalidated categories must still be refused. Phone screens and laptop
        # screens are validated in Phases 4 & 5.
        with self.assertRaises(DetectorUnavailableException):
            DetectionService().validate_available(["smartwatches"])
        DetectionService().validate_available(["faces", "license_plates", "laptop_screens", "phone_screens"])

    def test_tracker_instances_do_not_share_state(self):
        one, two = TrackingService(), TrackingService()
        one.update_tracks([face_detection()])
        self.assertEqual(len(one.active_tracks), 1)
        self.assertEqual(len(two.active_tracks), 0)

    def test_unknown_task_has_no_synthetic_status_or_result(self):
        service = RedactService()
        with self.assertRaises(TaskNotFoundException):
            service.get_task_status("missing")
        with self.assertRaises(TaskNotFoundException):
            service.get_task_summary("missing")

    def test_unknown_video_is_rejected_before_a_task_is_created(self):
        with self.assertRaises(VideoNotFoundException) as raised:
            asyncio.run(RedactService().start_redaction(StartRedactionRequest(video_id="vid_missing", target_categories=["faces"])))
        self.assertEqual(raised.exception.status_code, 404)
        self.assertEqual(task_manager.tasks, {})

    def test_failed_status_exposes_the_recorded_reason(self):
        task_manager.create_task("task_reason", "vid_x", "fixture.mp4", ["faces"], "blackout", 80)
        task_manager.fail_task("task_reason", "YuNet failed on frame 1")
        self.assertEqual(RedactService().get_task_status("task_reason").failure_reason, "YuNet failed on frame 1")

    def test_failed_job_cannot_be_downloaded(self):
        task_manager.create_task("task_failed", "vid_x", "fixture.mp4", ["faces"], "blackout", 80)
        task_manager.fail_task("task_failed", "encoder failed")
        with self.assertRaises(SafeFrameException) as raised:
            asyncio.run(download_video("task_failed", raw=False, audit=False))
        self.assertEqual(raised.exception.status_code, 409)

    def test_invalid_output_is_rejected(self):
        fd, path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        try:
            with self.assertRaises(RuntimeError):
                VideoProcessorService._validate_output(path)
        finally:
            os.unlink(path)

    def test_encoder_exit_failure_is_rejected(self):
        class ClosedInput:
            closed = False
            def close(self): self.closed = True
        class FailedEncoder:
            stdin = ClosedInput()
            class stderr:
                @staticmethod
                def read(): return b"encoding failed"
            @staticmethod
            def wait(timeout): return 1
        with self.assertRaisesRegex(RuntimeError, "ffmpeg exited 1"):
            VideoProcessorService._require_successful_encoder(FailedEncoder())
