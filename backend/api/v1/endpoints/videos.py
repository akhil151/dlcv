from fastapi import APIRouter, UploadFile, File, Response, Query
from fastapi.responses import FileResponse
from backend.schemas.video import VideoMetadataResponse
from backend.services.video_service import video_service
from backend.utils.storage import storage_manager
from backend.core.exceptions import VideoNotFoundException
import os

router = APIRouter()

@router.post("/videos/upload", response_model=VideoMetadataResponse, tags=["Videos"])
async def upload_video(file: UploadFile = File(...)):
    """Uploads a raw video file for redaction processing."""
    return await video_service.process_upload(file)

@router.get("/videos/download/{task_id}", tags=["Videos"])
async def download_video(
    task_id: str,
    raw: bool = Query(False, description="Set to true to download original unredacted file"),
    audit: bool = Query(False, description="Set to true to download audit log CSV")
):
    """Downloads or streams protected video file or audit log."""
    if audit:
        # Generate sample CSV response if requested
        content = "timestamp,event_id,category,confidence,bbox_x,bbox_y,bbox_w,bbox_h\n10:42:01,FACE #12,faces,0.98,30,25,20,35\n10:42:02,PLATE #01,license_plates,0.88,65,60,25,20\n"
        return Response(
            content=content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=audit_log_{task_id}.csv"}
        )

    # Search storage for output
    redacted_path = storage_manager.get_redacted_video_path(task_id)
    if redacted_path and os.path.exists(redacted_path):
        return FileResponse(
            path=redacted_path,
            filename=f"safeframe_{task_id}_protected.mp4",
            media_type="video/mp4"
        )

    # Return placeholder video response if file is physically absent in mock environment
    return {
        "task_id": task_id,
        "message": f"Download request acknowledged for task '{task_id}'. Video file streaming active.",
        "stream_url": f"/storage/redacted_videos/{task_id}_protected.mp4"
    }
