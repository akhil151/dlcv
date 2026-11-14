import os

from fastapi import APIRouter, UploadFile, File, Query
from fastapi.responses import FileResponse

from backend.schemas.video import VideoMetadataResponse
from backend.services.video_service import video_service
from backend.utils.storage import storage_manager
from backend.core.exceptions import SafeFrameException, TaskNotFoundException
from backend.workers.task_manager import task_manager

router = APIRouter()


def _status_value(task) -> str:
    return getattr(task.get("status"), "value", str(task.get("status")))


@router.post("/videos/upload", response_model=VideoMetadataResponse, tags=["Videos"])
async def upload_video(file: UploadFile = File(...)):
    """Uploads a raw video file for redaction processing."""
    return await video_service.process_upload(file)


@router.get("/videos/download/{task_id}", tags=["Videos"])
async def download_video(
    task_id: str,
    raw: bool = Query(False, description="Deprecated: use /videos/original/{task_id}"),
    audit: bool = Query(False, description="Audit-log export is not implemented"),
):
    """Downloads the verified protected video for a completed task."""
    task = task_manager.get_task(task_id)
    if not task:
        raise TaskNotFoundException(task_id)
    if raw:
        raise SafeFrameException(
            "Original videos are not served by the protected download endpoint; "
            "use /api/v1/videos/original/{task_id}.",
            status_code=400)
    if audit:
        raise SafeFrameException("Audit-log export is not implemented.", status_code=404)
    if _status_value(task) != "COMPLETED" or not task.get("output_verified"):
        raise SafeFrameException(
            "A verified protected video is not available for this task.", status_code=409)

    redacted_path = storage_manager.get_redacted_video_path(task_id)
    if redacted_path and os.path.exists(redacted_path):
        return FileResponse(
            path=redacted_path,
            filename=f"safeframe_{task_id}_protected.mp4",
            media_type="video/mp4",
        )

    raise SafeFrameException("Verified protected video is missing from storage.", status_code=410)


@router.get("/videos/original/{task_id}", tags=["Videos"])
async def download_original_video(task_id: str):
    """Serves the unredacted input of a verified task, for before/after review.

    This is deliberately limited to completed, verified tasks. The original is
    sensitive: it is the unredacted source material.
    """
    task = task_manager.get_task(task_id)
    if not task:
        raise TaskNotFoundException(task_id)
    if _status_value(task) != "COMPLETED" or not task.get("output_verified"):
        raise SafeFrameException(
            "The original video is only available for a verified completed task.",
            status_code=409)

    raw_path = storage_manager.get_raw_video_path(task.get("video_id", ""))
    if not raw_path or not os.path.isfile(raw_path):
        raise SafeFrameException("The original upload is no longer available.", status_code=410)

    extension = os.path.splitext(raw_path)[1].lower() or ".mp4"
    media_type = "video/mp4" if extension == ".mp4" else "application/octet-stream"
    return FileResponse(
        path=raw_path,
        filename=f"safeframe_{task_id}_original{extension}",
        media_type=media_type,
    )
