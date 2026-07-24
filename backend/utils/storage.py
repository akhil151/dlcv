import os
import shutil
from typing import Optional
from fastapi import UploadFile
from backend.config.settings import settings
from backend.core.logging import logger

class StorageManager:
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or settings.STORAGE_DIR
        self._ensure_directories()

    def _ensure_directories(self):
        os.makedirs(os.path.join(self.base_dir, "raw_videos"), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, "redacted_videos"), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, "audit_logs"), exist_ok=True)
        logger.info(f"Storage directories initialized at {self.base_dir}")

    async def save_raw_video(self, file: UploadFile, video_id: str) -> str:
        filename = f"{video_id}_{file.filename}"
        file_path = os.path.join(self.base_dir, "raw_videos", filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Saved raw video file to {file_path}")
        return file_path

    def get_raw_video_path(self, video_id: str) -> Optional[str]:
        raw_dir = os.path.join(self.base_dir, "raw_videos")
        if not os.path.exists(raw_dir):
            return None

        for fname in os.listdir(raw_dir):
            if fname.startswith(video_id):
                return os.path.join(raw_dir, fname)
        return None

    def get_redacted_video_path(self, task_id: str) -> Optional[str]:
        redacted_dir = os.path.join(self.base_dir, "redacted_videos")
        file_path = os.path.join(redacted_dir, f"{task_id}_protected.mp4")
        if os.path.exists(file_path):
            return file_path
        return None

storage_manager = StorageManager()
