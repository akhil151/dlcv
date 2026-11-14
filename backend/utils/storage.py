import os
import re
from typing import Optional
from fastapi import UploadFile
from backend.config.settings import settings
from backend.core.exceptions import FileUploadException
from backend.core.logging import logger

_CHUNK_SIZE = 1024 * 1024
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(name: Optional[str]) -> str:
    """Returns a filesystem-safe basename for a client-supplied filename.

    Path separators are stripped (no traversal) and unusual characters are
    replaced, while the extension is preserved for diagnostics.
    """
    base = os.path.basename((name or "").replace("\\", "/")).strip()
    if not base:
        return "video"
    cleaned = _SAFE_NAME.sub("_", base).lstrip(".")
    return cleaned[:120] or "video"


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
        """Streams an upload to disk, rejecting it if it exceeds the size limit."""
        filename = f"{video_id}_{safe_filename(file.filename)}"
        file_path = os.path.join(self.base_dir, "raw_videos", filename)
        limit_bytes = int(settings.MAX_UPLOAD_SIZE_MB) * 1024 * 1024
        written = 0

        try:
            with open(file_path, "wb") as buffer:
                while True:
                    chunk = await file.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > limit_bytes:
                        raise FileUploadException(
                            f"the uploaded file exceeds the {settings.MAX_UPLOAD_SIZE_MB} MB limit.")
                    buffer.write(chunk)
        except FileUploadException:
            try:
                os.remove(file_path)
            except OSError:
                pass
            raise
        except Exception as exc:
            try:
                os.remove(file_path)
            except OSError:
                pass
            raise FileUploadException(f"the uploaded file could not be saved: {exc}") from exc

        logger.info(f"Saved raw video file to {file_path} ({written} bytes)")
        return file_path

    def get_raw_video_path(self, video_id: str) -> Optional[str]:
        """Finds the stored upload for an id using an exact prefix match."""
        raw_dir = os.path.join(self.base_dir, "raw_videos")
        if not os.path.isdir(raw_dir) or not video_id:
            return None

        prefix = f"{video_id}_"
        for fname in os.listdir(raw_dir):
            if fname.startswith(prefix):
                return os.path.join(raw_dir, fname)
        return None

    def get_redacted_video_path(self, task_id: str) -> Optional[str]:
        redacted_dir = os.path.join(self.base_dir, "redacted_videos")
        file_path = os.path.join(redacted_dir, f"{task_id}_protected.mp4")
        if os.path.exists(file_path):
            return file_path
        return None


storage_manager = StorageManager()
