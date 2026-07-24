import uuid
import asyncio
from typing import Dict, Any
from backend.schemas.redact import (
    StartRedactionRequest,
    StartRedactionResponse,
    TaskStatusResponse,
    TaskSummaryResponse,
    TaskStatusEnum,
    BlurStyleEnum,
    ProcessingMetricsSchema,
)
from backend.workers.task_manager import task_manager
from backend.services.ai.evaluation_service import evaluation_service
from backend.core.exceptions import TaskNotFoundException
from backend.core.logging import logger

class RedactService:
    async def start_redaction(self, req: StartRedactionRequest) -> StartRedactionResponse:
        task_id = f"task_{uuid.uuid4().hex[:10]}"

        # Initialize task in task manager
        task_data = task_manager.create_task(
            task_id=task_id,
            video_id=req.video_id,
            filename="surveillance_city_square.mp4",
            target_categories=[c.value if hasattr(c, 'value') else str(c) for c in req.target_categories],
            blur_style=req.blur_style.value if hasattr(req.blur_style, 'value') else str(req.blur_style),
            blur_intensity=req.blur_intensity,
        )

        # Trigger async background progress simulation / pipeline processing
        asyncio.create_task(task_manager.simulate_task_progress(task_id))

        logger.info(f"Initiated redaction pipeline task '{task_id}' for video '{req.video_id}'")
        return StartRedactionResponse(task_id=task_id, status=TaskStatusEnum.PROCESSING)

    def get_task_status(self, task_id: str) -> TaskStatusResponse:
        task = task_manager.get_task(task_id)
        if not task:
            # Provide default mock response for demo-task if not dynamically generated yet
            if task_id == "demo-task":
                task = task_manager.create_task(
                    task_id="demo-task",
                    video_id="vid_demo",
                    filename="surveillance_city_square_04.mp4",
                    target_categories=["faces", "license_plates", "id_cards", "laptop_screens", "phone_screens"],
                    blur_style="gaussian",
                    blur_intensity=80,
                )
            else:
                raise TaskNotFoundException(task_id)

        metrics_data = task["metrics"]
        metrics_schema = ProcessingMetricsSchema(
            fps=metrics_data["fps"],
            elapsed_seconds=metrics_data["elapsed_seconds"],
            remaining_seconds=metrics_data["remaining_seconds"],
            throughput_frames=metrics_data["throughput_frames"],
            total_frames=metrics_data["total_frames"],
            processed_frames=metrics_data["processed_frames"],
        )

        return TaskStatusResponse(
            task_id=task["task_id"],
            video_id=task["video_id"],
            filename=task["filename"],
            status=task["status"],
            progress_pct=task["progress_pct"],
            target_categories=task["target_categories"],
            blur_style=task["blur_style"],
            blur_intensity=task["blur_intensity"],
            metrics=metrics_schema,
            recent_logs=task["recent_logs"],
        )

    def get_task_summary(self, task_id: str) -> Dict[str, Any]:
        task = task_manager.get_task(task_id)
        total_frames = task["metrics"]["total_frames"] if task else 300
        
        # Calculate full evaluation report
        eval_report = evaluation_service.evaluate_task_pipeline(task_id, total_frames=total_frames)

        return {
            "task_id": task_id,
            "faces_blurred": 12,
            "license_plates_redacted": 3,
            "screens_hidden": 1,
            "id_cards_found": 0,
            "phones_found": 0,
            "total_objects_count": 16,
            "processing_time_seconds": task["metrics"]["elapsed_seconds"] if task else 45,
            "original_video_url": f"/api/v1/videos/download/{task_id}?raw=true",
            "protected_video_url": f"/api/v1/videos/download/{task_id}",
            "audit_log_url": f"/api/v1/videos/download/{task_id}?audit=true",
            "evaluation_report": eval_report.model_dump() if hasattr(eval_report, 'model_dump') else eval_report.dict(),
        }

redact_service = RedactService()
