import uuid
from fastapi import UploadFile
from backend.schemas.video import VideoMetadataResponse
from backend.utils.storage import storage_manager
from backend.core.exceptions import FileUploadException
from backend.core.logging import logger

class VideoService:
    async def process_upload(self, file: UploadFile) -> VideoMetadataResponse:
        if not file.filename:
            raise FileUploadException("Filename is missing.")

        video_id = f"vid_{uuid.uuid4().hex[:10]}"

        # Save raw video to storage
        await storage_manager.save_raw_video(file, video_id)

        # Mock metadata calculation
        file_size = getattr(file, "size", 142500000) or 142500000
        logger.info(f"Successfully processed upload for '{file.filename}' -> video_id: '{video_id}'")

        return VideoMetadataResponse(
            video_id=video_id,
            filename=file.filename,
            file_size_bytes=file_size,
            format="H.264 / MP4",
            resolution="3840 x 2160 (4K)",
            duration_seconds=180,
            estimated_processing_seconds=252,
        )

video_service = VideoService()
