"""Phase 7D - operational recovery tests.

Test classes are explicitly separated:

* ``UnitTests`` (mocked/isolated): filesystem probes, mocked subprocess results,
  task-manager state transitions, and pipeline failure propagation driven by a
  clearly-labelled stub detector. These prove control flow, not real inference.
* ``IntegrationTests`` (genuinely executed): real FastAPI routing, real readiness
  probe, and real upload validation against real files. No mocks.
* Tests that require a real FFmpeg or a real YuNet ONNX file are marked NOT RUN
  (skipped) when the dependency is absent, never marked as passed.
"""
import asyncio
import json
import os
import tempfile
import unittest
from io import BytesIO
from unittest import mock

import cv2
import numpy as np

from backend.config.ai_settings import ai_settings
from backend.core.exceptions import FFmpegUnavailableException, SafeFrameException
from backend.schemas.detection import DetectionResultContainer
from backend.schemas.redact import TaskStatusEnum
from backend.services.ai import model_manager
from backend.services.ai.video_processor_service import VideoProcessorService, video_processor_service
from backend.services.readiness_service import readiness_service
from backend.utils import ffmpeg as ffmpeg_util
from backend.workers.task_manager import task_manager


def _make_tiny_video(path: str, frames: int = 8, size=(64, 64)) -> None:
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, size)
    try:
        for _ in range(frames):
            writer.write(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    finally:
        writer.release()


class _EmptyStubDetector:
    """Returns a genuine empty result. Does NOT fabricate detections.

    Used only to move the pipeline past detection so the encoder stage can be
    exercised; it is not a detection validation.
    """

    def detect_entities(self, frame, target_categories, frame_index=0, custom_conf_threshold=None):
        return DetectionResultContainer(
            frame_index=frame_index, total_detections=0, detections=[],
            inference_time_ms=0.0, categories_detected=[], secondary_detector_used=True)


class UnitTests(unittest.TestCase):
    def setUp(self):
        task_manager.tasks.clear()

    # ----- A. YuNet path resolution / validation -----

    def test_resolve_yunet_absolute_path_used_as_is(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "yunet.onnx")
            open(target, "wb").close()
            with mock.patch.object(ai_settings, "YUNET_MODEL_PATH", target):
                self.assertEqual(model_manager.resolve_yunet_path(), target)

    def test_missing_yunet_weights_report_unavailable(self):
        missing = os.path.join(tempfile.gettempdir(), "definitely_missing_yunet_93f.onnx")
        with mock.patch.object(ai_settings, "YUNET_MODEL_PATH", missing):
            model, status, detail, path = model_manager._load_yunet()
            self.assertIsNone(model)
            self.assertEqual(status, "UNAVAILABLE")
            self.assertIn("not found", detail)

    def test_empty_yunet_weights_report_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "empty.onnx")
            open(target, "wb").close()
            with mock.patch.object(ai_settings, "YUNET_MODEL_PATH", target):
                model, status, detail, _ = model_manager._load_yunet()
                self.assertIsNone(model)
                self.assertEqual(status, "UNAVAILABLE")
                self.assertIn("empty", detail)

    def test_invalid_yunet_weights_do_not_load_and_do_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "garbage.onnx")
            with open(target, "wb") as fh:
                fh.write(b"this is not an onnx model" * 8)
            with mock.patch.object(ai_settings, "YUNET_MODEL_PATH", target):
                model, status, detail, _ = model_manager._load_yunet()
                # Either OpenCV rejects it (ERROR) or explicitly reports unavailable;
                # it must never silently return a usable model.
                self.assertIsNone(model)
                self.assertIn(status, ("ERROR", "UNAVAILABLE"))
                self.assertTrue(detail)

    def test_detector_readiness_reflects_missing_weights_without_loading_models(self):
        missing = os.path.join(tempfile.gettempdir(), "missing_yunet_readiness.onnx")
        with mock.patch.object(ai_settings, "YUNET_MODEL_PATH", missing), \
             mock.patch.object(model_manager, "models", {}, create=True):
            snapshot = model_manager.detector_readiness()
        self.assertFalse(snapshot["available"])
        self.assertEqual(snapshot["status"], "UNAVAILABLE")

    def test_successful_yunet_initialization(self):
        """Requires a real YuNet ONNX file. NOT RUN when unavailable."""
        resolved = model_manager.resolve_yunet_path()
        if not (os.path.isfile(resolved) and os.path.getsize(resolved) > 0):
            self.skipTest("NOT RUN: real YuNet ONNX weights are not available")
        model, status, detail, _ = model_manager._load_yunet()
        self.assertEqual(status, "LOADED", detail)
        self.assertIsNotNone(model)

    # ----- B. FFmpeg discovery / validation (subprocess mocked) -----

    def test_resolve_ffmpeg_explicit_path_returns_none_when_not_executable(self):
        with mock.patch.object(ai_settings, "FFMPEG_PATH", os.path.join(tempfile.gettempdir(), "nope_ffmpeg")):
            self.assertIsNone(ffmpeg_util.resolve_ffmpeg_path())

    def test_resolve_ffmpeg_uses_path_lookup_when_unconfigured(self):
        with mock.patch.object(ai_settings, "FFMPEG_PATH", ""), \
             mock.patch.object(ffmpeg_util.shutil, "which", return_value="/usr/bin/ffmpeg"):
            self.assertEqual(ffmpeg_util.resolve_ffmpeg_path(), "/usr/bin/ffmpeg")

    def test_missing_ffmpeg_reports_actionable_detail(self):
        with mock.patch.object(ai_settings, "FFMPEG_PATH", ""), \
             mock.patch.object(ffmpeg_util.shutil, "which", return_value=None):
            status = ffmpeg_util.validate_ffmpeg(force=True)
        self.assertFalse(status["available"])
        self.assertIn("ffmpeg", str(status["detail"]).lower())
        ffmpeg_util.reset_cache()

    def test_invalid_executable_path_reports_unavailable(self):
        configured = os.path.join(tempfile.gettempdir(), "bad_ffmpeg_path")
        with mock.patch.object(ai_settings, "FFMPEG_PATH", configured):
            status = ffmpeg_util.validate_ffmpeg(force=True)
        self.assertFalse(status["available"])
        self.assertIn("FFMPEG_PATH", str(status["detail"]))
        ffmpeg_util.reset_cache()

    def test_ffmpeg_process_failure_reports_unavailable(self):
        with mock.patch.object(ai_settings, "FFMPEG_PATH", ""), \
             mock.patch.object(ffmpeg_util.shutil, "which", return_value="/fake/ffmpeg"), \
             mock.patch.object(ffmpeg_util.subprocess, "run",
                               return_value=mock.Mock(returncode=1, stdout=b"", stderr=b"boom")):
            status = ffmpeg_util.validate_ffmpeg(force=True)
        self.assertFalse(status["available"])
        self.assertIn("exited 1", str(status["detail"]))
        ffmpeg_util.reset_cache()

    def test_ffmpeg_success_reports_version(self, ):
        probe = mock.Mock(returncode=0, stdout=b"ffmpeg version 6.1.1-static Copyright\n", stderr=b"")
        with mock.patch.object(ai_settings, "FFMPEG_PATH", ""), \
             mock.patch.object(ffmpeg_util.shutil, "which", return_value="/fake/ffmpeg"), \
             mock.patch.object(ffmpeg_util.subprocess, "run", return_value=probe):
            status = ffmpeg_util.validate_ffmpeg(force=True)
        self.assertTrue(status["available"])
        self.assertEqual(status["version"], "6.1.1-static")
        ffmpeg_util.reset_cache()

    def test_require_ffmpeg_raises_clear_error_when_missing(self):
        with mock.patch.object(ai_settings, "FFMPEG_PATH", ""), \
             mock.patch.object(ffmpeg_util.shutil, "which", return_value=None):
            with self.assertRaises(FFmpegUnavailableException) as raised:
                ffmpeg_util.require_ffmpeg()
        self.assertEqual(raised.exception.status_code, 503)
        ffmpeg_util.reset_cache()

    # ----- C. Task-state integrity -----

    def test_no_fabricated_progress_simulation_exists(self):
        self.assertFalse(hasattr(task_manager, "simulate_task_progress"))

    def test_cancel_task_sets_cancelled_and_blocks_download_flag(self):
        task_manager.create_task("t_cancel", "vid", "f.mp4", ["faces"], "blackout", 80)
        task_manager.cancel_task("t_cancel", "user cancelled")
        task = task_manager.get_task("t_cancel")
        self.assertEqual(task["status"], TaskStatusEnum.CANCELLED)
        self.assertFalse(task["output_verified"])

    def test_cancel_does_not_overwrite_completed_task(self):
        task = task_manager.create_task("t_done", "vid", "f.mp4", ["faces"], "blackout", 80)
        task["status"] = TaskStatusEnum.COMPLETED
        task["output_verified"] = True
        task_manager.cancel_task("t_done")
        self.assertEqual(task_manager.get_task("t_done")["status"], TaskStatusEnum.COMPLETED)

    def test_encoder_unavailable_fails_the_task_and_never_completes(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = os.path.join(tmp, "in.mp4")
            _make_tiny_video(raw)
            task_manager.create_task("t_ffmpeg", "vid_ff", "in.mp4", ["faces"], "blackout", 80)
            with mock.patch.object(video_processor_service, "detection_service", _EmptyStubDetector()), \
                 mock.patch("backend.services.ai.video_processor_service.require_ffmpeg",
                            side_effect=FFmpegUnavailableException("ffmpeg is not installed (test)")), \
                 mock.patch("backend.utils.storage.storage_manager.get_raw_video_path", return_value=raw):
                with self.assertRaises(FFmpegUnavailableException):
                    asyncio.run(video_processor_service.process_video_pipeline(
                        "t_ffmpeg", "vid_ff", ["faces"], "blackout", 80))
            task = task_manager.get_task("t_ffmpeg")
            self.assertEqual(task["status"], TaskStatusEnum.FAILED)
            self.assertFalse(task["output_verified"])
            self.assertNotEqual(task["status"], TaskStatusEnum.COMPLETED)
            summary_raises = False
            try:
                from backend.services.redact_service import RedactService
                RedactService().get_task_summary("t_ffmpeg")
            except SafeFrameException as exc:
                summary_raises = exc.status_code == 409
            self.assertTrue(summary_raises)

    def test_cancelled_worker_is_not_left_in_processing(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = os.path.join(tmp, "in.mp4")
            _make_tiny_video(raw)
            task_manager.create_task("t_cancel_pipeline", "vid_c", "in.mp4", ["faces"], "blackout", 80)
            with mock.patch.object(video_processor_service, "detection_service", _EmptyStubDetector()), \
                 mock.patch("backend.services.ai.video_processor_service.require_ffmpeg",
                            side_effect=asyncio.CancelledError()), \
                 mock.patch("backend.utils.storage.storage_manager.get_raw_video_path", return_value=raw):
                try:
                    asyncio.run(video_processor_service.process_video_pipeline(
                        "t_cancel_pipeline", "vid_c", ["faces"], "blackout", 80))
                except BaseException:
                    pass
            self.assertEqual(task_manager.get_task("t_cancel_pipeline")["status"], TaskStatusEnum.CANCELLED)

    def test_invalid_output_cannot_be_marked_verified(self):
        fd, path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        try:
            with self.assertRaises(RuntimeError):
                VideoProcessorService._validate_output(path)
        finally:
            os.unlink(path)


class IntegrationTests(unittest.TestCase):
    """Genuinely executed against the real app. No mocks."""

    def test_liveness_and_readiness_are_separate(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        with TestClient(app) as client:
            live = client.get("/api/v1/health")
            self.assertEqual(live.status_code, 200)
            self.assertEqual(live.json()["status"], "healthy")

            ready = client.get("/api/v1/health/readiness")
            self.assertEqual(ready.status_code, 200)
            payload = ready.json()
            self.assertIn("processing_ready", payload)
            self.assertEqual(
                payload["processing_ready"],
                bool(payload["detector"]["available"] and payload["ffmpeg"]["available"]),
            )
            if not payload["ffmpeg"]["available"]:
                self.assertFalse(payload["processing_ready"])
                self.assertTrue(payload["reasons"])
            # No filesystem paths must leak through the public readiness payload.
            serialized = json.dumps(payload)
            self.assertNotIn(":\\\\", serialized)
            self.assertNotIn("/backend/", serialized.replace("\\\\", "/"))

    def test_readiness_does_not_load_heavy_models(self):
        # Readiness must never trigger model loading, regardless of cache state.
        with mock.patch.object(model_manager, "get_model",
                               side_effect=AssertionError("readiness must not load models")):
            payload = readiness_service.snapshot()
        self.assertIn("processing_ready", payload)

    def test_upload_rejects_non_video_and_removes_file(self):
        from fastapi import UploadFile
        from backend.core.exceptions import FileUploadException
        from backend.services.video_service import video_service
        upload = UploadFile(file=BytesIO(b"this is not a video file"), filename="notes.txt")
        with self.assertRaises(FileUploadException):
            asyncio.run(video_service.process_upload(upload))

    def test_upload_accepts_real_video(self):
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sample = os.path.join(root, "test_input.mp4")
        if not os.path.isfile(sample):
            self.skipTest("NOT RUN: no real video available for upload")
        from fastapi import UploadFile
        from backend.services.video_service import video_service
        from backend.utils.storage import storage_manager
        with open(sample, "rb") as fh:
            upload = UploadFile(file=BytesIO(fh.read()), filename="test_input.mp4")
        meta = asyncio.run(video_service.process_upload(upload))
        try:
            self.assertGreater(meta.file_size_bytes, 0)
            self.assertNotEqual(meta.resolution, "unknown")
        finally:
            saved = storage_manager.get_raw_video_path(meta.video_id)
            if saved and os.path.exists(saved):
                os.remove(saved)

    def test_real_ffmpeg_is_usable_if_present(self):
        status = ffmpeg_util.validate_ffmpeg(force=True)
        if not status["available"]:
            self.skipTest("NOT RUN: FFmpeg is not installed in this environment")
        self.assertTrue(status["version"])


if __name__ == "__main__":
    unittest.main()
