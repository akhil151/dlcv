import time
from typing import Dict, Any, Optional
from backend.schemas.redact import TaskStatusEnum
from backend.core.logging import logger


class TaskManager:
    """Process-local task registry.

    Only the real processing pipeline may move a task to COMPLETED, and only
    after the output has been independently verified. There is deliberately no
    simulation/progress-fabrication path: progress is written exclusively by the
    running pipeline.
    """

    def __init__(self):
        # task_id -> task dict
        self.tasks: Dict[str, Dict[str, Any]] = {}

    def create_task(self, task_id: str, video_id: str, filename: str, target_categories: list,
                    blur_style: str, blur_intensity: int, debug_mode: bool = False) -> Dict[str, Any]:
        task_data = {
            "task_id": task_id,
            "video_id": video_id,
            "filename": filename,
            "status": TaskStatusEnum.PROCESSING,
            "progress_pct": 0.0,
            "target_categories": target_categories,
            "blur_style": blur_style,
            "blur_intensity": blur_intensity,
            "debug_mode": bool(debug_mode),
            "cancel_requested": False,
            "output_verified": False,
            "failure_reason": None,
            "metrics": {
                "fps": 0,
                "elapsed_seconds": 0,
                "remaining_seconds": 0,
                "throughput_frames": 0,
                "total_frames": 0,
                "processed_frames": 0,
                "frames_with_detections": 0,
                "masks_total": 0,
                "new_tracks_total": 0,
                "detection_latency_ms": 0.0,
                "blur_latency_ms": 0.0,
                "avg_processing_fps": 0.0,
                "debug_frames_saved": 0,
            },
            "recent_logs": [
                f"[Worker] Task '{task_id}' initialized for file '{filename}'",
            ],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self.tasks[task_id] = task_data
        logger.info(f"Created task '{task_id}' in TaskManager")
        return task_data

    def fail_task(self, task_id: str, reason: str) -> None:
        task = self.tasks.get(task_id)
        if task:
            task["status"] = TaskStatusEnum.FAILED
            task["failure_reason"] = reason
            task["output_verified"] = False
            task["recent_logs"].append(f"[Worker] FAILED: {reason}")
            logger.warning(f"Task '{task_id}' marked FAILED: {reason}")

    def cancel_task(self, task_id: str, reason: str = "processing cancelled") -> None:
        task = self.tasks.get(task_id)
        if task:
            task["cancel_requested"] = True
            if task["status"] not in (TaskStatusEnum.COMPLETED, TaskStatusEnum.FAILED):
                task["status"] = TaskStatusEnum.CANCELLED
                task["output_verified"] = False
                task["failure_reason"] = reason
                task["recent_logs"].append(f"[Worker] CANCELLED: {reason}")
                logger.warning(f"Task '{task_id}' marked CANCELLED: {reason}")

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self.tasks.get(task_id)


task_manager = TaskManager()
