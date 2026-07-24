from fastapi import APIRouter
from backend.schemas.redact import (
    StartRedactionRequest,
    StartRedactionResponse,
    TaskStatusResponse,
    TaskSummaryResponse,
)
from backend.services.redact_service import redact_service

router = APIRouter()

@router.post("/redact/start", response_model=StartRedactionResponse, tags=["Redaction"])
async def start_redaction(req: StartRedactionRequest):
    """Initiates an asynchronous video redaction job with configured target categories."""
    return await redact_service.start_redaction(req)

@router.get("/redact/status/{task_id}", response_model=TaskStatusResponse, tags=["Redaction"])
async def get_task_status(task_id: str):
    """Retrieves live processing status, throughput metrics, and log stream for a task."""
    return redact_service.get_task_status(task_id)

@router.get("/redact/results/{task_id}", response_model=TaskSummaryResponse, tags=["Redaction"])
async def get_task_summary(task_id: str):
    """Retrieves completion breakdown summary stats and download URLs for a completed task."""
    return redact_service.get_task_summary(task_id)
