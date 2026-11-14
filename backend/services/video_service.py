import os
import cv2
import uuid
from fastapi import UploadFile
from backend.schemas.video import VideoMetadataResponse
from backend.utils.storage import storage_manager
from backend.core.exceptions import FileUploadException
from backend.core.logging import logger

class VideoService:
    @staticmethod
    def _discard(path: str) -> None:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError as err:  # pragma: no cover - best effort cleanup
            logger.warning(f"Failed to remove rejected upload '{path}': {err}")

    async def process_upload(self, file: UploadFile) -> VideoMetadataResponse:
        if not file.filename:
            raise FileUploadException("Filename is missing.")

        video_id = f"vid_{uuid.uuid4().hex[:10]}"

        # Save raw video to storage
        saved_path = await storage_manager.save_raw_video(file, video_id)

        file_size = os.path.getsize(saved_path) if os.path.exists(saved_path) else 0
        if file_size == 0:
            self._discard(saved_path)
            raise FileUploadException("the uploaded file is empty.")

        # An upload is only accepted when OpenCV can actually decode it. A file
        # that cannot be opened/decoded is rejected explicitly instead of later
        # failing (or worse, appearing to succeed) inside the pipeline.
        resolution = "unknown"
        duration_sec = 0
        fps = 0.0
        frame_count = 0
        try:
            cap = cv2.VideoCapture(saved_path)
            if not cap.isOpened():
                cap.release()
                self._discard(saved_path)
                raise FileUploadException(
                    "the uploaded file could not be opened as a video (unsupported or corrupted input).")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_val = float(cap.get(cv2.CAP_PROP_FPS))
            readable, probe_frame = cap.read()
            cap.release()
            if not readable or probe_frame is None or getattr(probe_frame, "size", 0) == 0:
                self._discard(saved_path)
                raise FileUploadException("the uploaded video contains no decodable frames.")
            if width > 0 and height > 0:
                resolution = f"{width} x {height}"
            if fps_val > 0:
                fps = fps_val
            if frame_count_meta > 0:
                frame_count = frame_count_meta
            if frame_count > 0 and fps > 0:
                duration_sec = int(round(frame_count / fps))
        except FileUploadException:
            raise
        except Exception as err:
            self._discard(saved_path)
            logger.warning(f"Failed to inspect uploaded file '{file.filename}': {err}")
            raise FileUploadException("the uploaded file could not be read as a valid video.") from err

        container = os.path.splitext(file.filename)[1].lstrip(".").upper() or "UNKNOWN"
        logger.info(
            "Successfully processed upload for '%s' (%s, %.2ffps, %d frames) -> video_id: '%s'",
            file.filename, resolution, fps, frame_count, video_id)

        return VideoMetadataResponse(
            video_id=video_id,
            filename=file.filename,
            file_size_bytes=file_size,
            format=f"{container} container",
            resolution=resolution,
            duration_seconds=duration_sec,
            fps=round(fps, 2),
            frame_count=frame_count,
            # Rough local estimate: ~10 processed frames/second on CPU-only
            # YuNet inference. It is labeled as an estimate in the UI.
            estimated_processing_seconds=max(5, int(round(frame_count / 10.0))) if frame_count else 5,
        )

video_service = VideoService()
