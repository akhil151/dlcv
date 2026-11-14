import os
from typing import Any, Dict, List

from backend.schemas.dashboard import (
    DashboardOverviewResponse,
    DashboardStats,
    RecentVideoSchema,
)
from backend.utils.storage import storage_manager
from backend.workers.task_manager import task_manager


def _status_value(task: Dict[str, Any]) -> str:
    status = task.get("status")
    return getattr(status, "value", str(status))


class DashboardService:
    """Derives dashboard data from the real, process-local task registry.

    There is no persistence and no demo/sample data: before any job runs the
    dashboard legitimately reports zeros and an empty recent-jobs list.
    """

    def _file_size_mb(self, task: Dict[str, Any]) -> float:
        raw_path = storage_manager.get_raw_video_path(task.get("video_id", ""))
        if raw_path and os.path.exists(raw_path):
            # Two decimals so short demo clips do not round down to "0 MB".
            return round(os.path.getsize(raw_path) / (1024 * 1024), 2)
        return 0.0

    def get_overview(self) -> DashboardOverviewResponse:
        tasks: List[Dict[str, Any]] = list(task_manager.tasks.values())
        completed = [t for t in tasks if _status_value(t) == "COMPLETED"]

        objects_detected = sum(
            sum((t.get("redaction_counts") or {}).values()) for t in completed
        )
        elapsed_values = [int(t.get("metrics", {}).get("elapsed_seconds", 0) or 0) for t in completed]
        avg_processing = int(sum(elapsed_values) / len(elapsed_values)) if elapsed_values else 0

        recent_videos: List[RecentVideoSchema] = []
        for task in sorted(tasks, key=lambda item: item.get("created_at", ""), reverse=True)[:20]:
            metrics = task.get("metrics", {}) or {}
            total_frames = int(metrics.get("total_frames", 0) or 0)
            processed_frames = int(metrics.get("processed_frames", 0) or 0)
            duration = f"{processed_frames}/{total_frames} frames" if total_frames else "—"
            recent_videos.append(RecentVideoSchema(
                id=task.get("task_id", ""),
                filename=task.get("filename", "unknown"),
                file_size_mb=self._file_size_mb(task),
                targets=list(task.get("target_categories", []) or []),
                thumbnail_url="",
                created_at=task.get("created_at", ""),
                duration=duration,
                status=_status_value(task),
            ))

        return DashboardOverviewResponse(
            stats=DashboardStats(
                videos_processed=len(completed),
                objects_detected=int(objects_detected),
                avg_processing_time_seconds=avg_processing,
            ),
            recent_videos=recent_videos,
        )


dashboard_service = DashboardService()
