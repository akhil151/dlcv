"""FFmpeg discovery and validation.

Resolution order:
1. An explicitly configured executable path (``ai_settings.FFMPEG_PATH``).
2. The system ``PATH`` (``shutil.which``).

There is deliberately no fallback to another encoder: if FFmpeg cannot be
resolved and executed, the pipeline must fail explicitly rather than silently
switching to an unverified or fake encoder.
"""
import os
import shutil
import subprocess
import threading
from typing import Dict, Optional

from backend.config.ai_settings import ai_settings
from backend.core.exceptions import FFmpegUnavailableException
from backend.core.logging import logger

_lock = threading.Lock()
_cached_status: Optional[Dict[str, object]] = None

_VERSION_TIMEOUT_SECONDS = 10


def resolve_ffmpeg_path() -> Optional[str]:
    """Returns an executable FFmpeg path, or ``None`` if none can be resolved."""
    configured = (ai_settings.FFMPEG_PATH or "").strip()
    if configured:
        if os.path.isfile(configured) and os.access(configured, os.X_OK):
            return configured
        return None
    return shutil.which("ffmpeg")


def _probe() -> Dict[str, object]:
    configured = (ai_settings.FFMPEG_PATH or "").strip()
    path = resolve_ffmpeg_path()
    if path is None:
        detail = (
            f"configured FFMPEG_PATH '{configured}' does not exist or is not executable"
            if configured
            else "ffmpeg was not found on PATH and FFMPEG_PATH is not configured"
        )
        return {"available": False, "path": None, "version": None, "detail": detail}

    try:
        proc = subprocess.run(
            [path, "-version"],
            capture_output=True,
            timeout=_VERSION_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return {"available": False, "path": path, "version": None,
                "detail": f"'{path} -version' timed out after {_VERSION_TIMEOUT_SECONDS}s"}
    except OSError as exc:
        return {"available": False, "path": path, "version": None,
                "detail": f"failed to execute '{path}': {exc}"}

    if proc.returncode != 0:
        stderr = (proc.stderr or b"").decode("utf-8", "replace").strip()
        return {"available": False, "path": path, "version": None,
                "detail": f"'{path} -version' exited {proc.returncode}: {stderr[-200:]}"}

    stdout = (proc.stdout or b"").decode("utf-8", "replace")
    first_line = stdout.splitlines()[0] if stdout.strip() else ""
    version = None
    if first_line.startswith("ffmpeg version "):
        version = first_line[len("ffmpeg version "):].split(" ")[0]
    return {"available": True, "path": path, "version": version, "detail": "ok"}


def validate_ffmpeg(force: bool = False) -> Dict[str, object]:
    """Validates FFmpeg once and caches the result (avoids probing per request)."""
    global _cached_status
    with _lock:
        if _cached_status is not None and not force:
            return _cached_status
        status = _probe()
        _cached_status = status
        if status["available"]:
            logger.info(f"[FFmpeg] Using '{status['path']}' (version {status['version']}).")
        else:
            logger.warning(f"[FFmpeg] Unavailable: {status['detail']}")
        return status


def reset_cache() -> None:
    """Clears the cached probe (used by tests and explicit revalidation)."""
    global _cached_status
    with _lock:
        _cached_status = None


def require_ffmpeg() -> str:
    """Returns the validated FFmpeg path or raises an actionable 503 error."""
    status = validate_ffmpeg()
    if not status["available"]:
        raise FFmpegUnavailableException(str(status["detail"]))
    return str(status["path"])


def _resolve_ffprobe(ffmpeg_path: str) -> Optional[str]:
    """Resolves ffprobe from PATH, then next to the resolved ffmpeg binary."""
    found = shutil.which("ffprobe")
    if found:
        return found
    directory = os.path.dirname(ffmpeg_path)
    for name in ("ffprobe.exe", "ffprobe"):
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate):
            return candidate
    return None


def probe_audio_streams(video_path: str) -> Optional[int]:
    """Returns the number of audio streams in a file, or None if unknown.

    Used only for reporting whether audio was preserved. A missing ffprobe is
    reported as unknown rather than guessed.
    """
    try:
        ffmpeg_path = require_ffmpeg()
    except FFmpegUnavailableException:
        return None
    ffprobe = _resolve_ffprobe(ffmpeg_path)
    if not ffprobe:
        return None
    try:
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", video_path],
            capture_output=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("[FFmpeg] ffprobe failed for audio check: %s", exc)
        return None
    if proc.returncode != 0:
        return None
    lines = [line for line in (proc.stdout or b"").decode("utf-8", "replace").splitlines() if line.strip()]
    return len(lines)
