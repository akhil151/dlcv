import asyncio
import time
from typing import Dict, Any, Optional
from backend.schemas.redact import TaskStatusEnum
from backend.websocket.connection_manager import ws_manager
from backend.core.logging import logger

class TaskManager:
    def __init__(self):
        # task_id -> task dict
        self.tasks: Dict[str, Dict[str, Any]] = {}

    def create_task(self, task_id: str, video_id: str, filename: str, target_categories: list, blur_style: str, blur_intensity: int) -> Dict[str, Any]:
        task_data = {
            "task_id": task_id,
            "video_id": video_id,
            "filename": filename,
            "status": TaskStatusEnum.PROCESSING,
            "progress_pct": 0.0,
            "target_categories": target_categories,
            "blur_style": blur_style,
            "blur_intensity": blur_intensity,
            "metrics": {
                "fps": 30,
                "elapsed_seconds": 0,
                "remaining_seconds": 120,
                "throughput_frames": 0,
                "total_frames": 3600,
                "processed_frames": 0,
            },
            "recent_logs": [
                f"[Worker] Task '{task_id}' initialized for file '{filename}'",
                "[Worker] CUDA GPU acceleration context ready",
            ],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self.tasks[task_id] = task_data
        logger.info(f"Created task '{task_id}' in TaskManager")
        return task_data

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self.tasks.get(task_id)

    async def simulate_task_progress(self, task_id: str):
        """Simulates async background processing updates and broadcasts WebSocket packets."""
        task = self.tasks.get(task_id)
        if not task:
            return

        for pct in range(0, 101, 10):
            await asyncio.sleep(1.5)
            task["progress_pct"] = float(pct)
            task["metrics"]["elapsed_seconds"] += 2
            task["metrics"]["remaining_seconds"] = max(0, 120 - task["metrics"]["elapsed_seconds"])
            task["metrics"]["processed_frames"] = int((pct / 100.0) * 3600)
            task["metrics"]["throughput_frames"] = task["metrics"]["processed_frames"]

            log_entry = f"[{time.strftime('%H:%M:%S')}] Frame #{task['metrics']['processed_frames']}: Processing stage {pct}% complete"
            task["recent_logs"].append(log_entry)

            if pct >= 100:
                task["status"] = TaskStatusEnum.COMPLETED
                task["recent_logs"].append("[Worker] Pipeline finished successfully. Video rendered.")

            # Broadcast WebSocket packet
            telemetry_packet = {
                "task_id": task_id,
                "frame_index": task["metrics"]["processed_frames"],
                "total_frames": task["metrics"]["total_frames"],
                "progress_pct": task["progress_pct"],
                "fps": task["metrics"]["fps"],
                "status": task["status"].value if isinstance(task["status"], TaskStatusEnum) else str(task["status"]),
                "log_line": log_entry,
                "detected_objects": [
                    {"id": "FACE #12", "category": "faces", "label": "FACE #12", "confidence": 0.98, "bbox": {"x": 30, "y": 25, "width": 20, "height": 35}},
                    {"id": "PLATE #01", "category": "license_plates", "label": "PLATE #01", "confidence": 0.88, "bbox": {"x": 65, "y": 60, "width": 25, "height": 20}},
                ]
            }
            await ws_manager.broadcast_to_task(task_id, telemetry_packet)

task_manager = TaskManager()
