import uuid
import asyncio
import os
from typing import Any, Dict

from backend.schemas.redact import (
    StartRedactionRequest,
    StartRedactionResponse,
    TaskStatusResponse,
    TaskSummaryResponse,
    TaskStatusEnum,
    ProcessingMetricsSchema,
)
from backend.workers.task_manager import task_manager
from backend.core.exceptions import SafeFrameException, TaskNotFoundException, VideoNotFoundException
from backend.utils.storage import storage_manager
from backend.core.logging import logger

from backend.services.ai.video_processor_service import video_processor_service
from backend.services.ai.detection_service import detection_service
from backend.utils.ffmpeg import require_ffmpeg

TERMINAL_STATUSES = (TaskStatusEnum.COMPLETED, TaskStatusEnum.FAILED, TaskStatusEnum.CANCELLED)


def _status_value(task: Dict[str, Any]) -> str:
    return getattr(task.get("status"), "value", str(task.get("status")))


class RedactService:
    async def start_redaction(self, req: StartRedactionRequest) -> StartRedactionResponse:
        raw_path = storage_manager.get_raw_video_path(req.video_id)
        if not raw_path or not os.path.isfile(raw_path):
            raise VideoNotFoundException(req.video_id)

        cats = [c.value if hasattr(c, "value") else str(c) for c in req.target_categories]
        if not cats:
            raise SafeFrameException("at least one redaction target is required.", status_code=422)

        b_style = req.blur_style.value if hasattr(req.blur_style, "value") else str(req.blur_style)

        # Fail before a task exists when the job could never be executed.
        detection_service.validate_available(cats)
        require_ffmpeg()

        task_id = f"task_{uuid.uuid4().hex[:10]}"
        task_data = task_manager.create_task(
            task_id=task_id,
            video_id=req.video_id,
            filename=os.path.basename(raw_path),
            target_categories=cats,
            blur_style=b_style,
            blur_intensity=req.blur_intensity,
            debug_mode=req.debug_mode,
        )

        # Keep a reference to the worker so it cannot be garbage-collected while
        # it runs; cancellation is cooperative through task["cancel_requested"].
        worker = asyncio.create_task(
            video_processor_service.process_video_pipeline(
                task_id=task_id,
                video_id=req.video_id,
                target_categories=cats,
                blur_style=b_style,
                blur_intensity=req.blur_intensity,
                debug_mode=req.debug_mode,
            )
        )
        task_data["worker"] = worker

        logger.info("Initiated redaction task '%s' for video '%s'", task_id, req.video_id)
        return StartRedactionResponse(task_id=task_id, status=TaskStatusEnum.PROCESSING)

    def get_task_status(self, task_id: str) -> TaskStatusResponse:
        task = task_manager.get_task(task_id)
        if not task:
            raise TaskNotFoundException(task_id)

        metrics_data = task["metrics"]
        metrics_schema = ProcessingMetricsSchema(
            fps=metrics_data.get("fps", 0),
            elapsed_seconds=metrics_data.get("elapsed_seconds", 0),
            remaining_seconds=metrics_data.get("remaining_seconds", 0),
            throughput_frames=metrics_data.get("throughput_frames", 0),
            total_frames=metrics_data.get("total_frames", 0),
            processed_frames=metrics_data.get("processed_frames", 0),
            frames_with_detections=metrics_data.get("frames_with_detections", 0),
            masks_total=metrics_data.get("masks_total", 0),
            new_tracks_total=metrics_data.get("new_tracks_total", 0),
            detection_latency_ms=metrics_data.get("detection_latency_ms", 0.0),
            blur_latency_ms=metrics_data.get("blur_latency_ms", 0.0),
            avg_processing_fps=metrics_data.get("avg_processing_fps", 0.0),
            debug_frames_saved=metrics_data.get("debug_frames_saved", 0),
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
            created_at=task.get("created_at"),
            debug_mode=bool(task.get("debug_mode", False)),
            failure_reason=task.get("failure_reason"),
            output_verified=task.get("output_verified"),
        )

    def cancel_redaction(self, task_id: str) -> TaskStatusResponse:
        """Cooperatively cancels a running task.

        The worker checks the flag between frames, closes the encoder and deletes
        any partial output. A completed task cannot be cancelled.
        """
        task = task_manager.get_task(task_id)
        if not task:
            raise TaskNotFoundException(task_id)
        if task["status"] == TaskStatusEnum.COMPLETED:
            raise SafeFrameException(
                "this task already completed; its verified output cannot be cancelled.",
                status_code=409)
        if task["status"] == TaskStatusEnum.FAILED:
            raise SafeFrameException("this task already failed.", status_code=409)

        task_manager.cancel_task(task_id, "cancelled by client")
        logger.info("Cancellation requested for task '%s'", task_id)
        return self.get_task_status(task_id)

    def get_task_summary(self, task_id: str) -> TaskSummaryResponse:
        task = task_manager.get_task(task_id)
        if not task:
            raise TaskNotFoundException(task_id)
        if task["status"] != TaskStatusEnum.COMPLETED or not task.get("output_verified"):
            raise SafeFrameException(
                "A verified protected result is not available for this task.", status_code=409)

        metrics = task.get("metrics", {})
        counts = task.get("redaction_counts", {}) or {}
        frames_processed = int(metrics.get("processed_frames", 0) or 0)
        frames_with_detections = int(metrics.get("frames_with_detections", 0) or 0)
        detection_frame_pct = (
            round(frames_with_detections / frames_processed * 100.0, 1) if frames_processed else 0.0)

        verification = task.get("verification") or {}
        output_resolution = None
        if verification.get("width") and verification.get("height"):
            output_resolution = f"{verification['width']} x {verification['height']}"

        audio_preserved = metrics.get("audio_preserved")
        if audio_preserved is not None:
            audio_preserved = bool(audio_preserved)

        # The document detector is class-agnostic: ID cards and passports are
        # located by the same validated graph, and the pipeline records them in
        # the canonical "id_cards" bucket. They are reported together as
        # document regions instead of inventing per-type counts.
        documents_redacted = int(counts.get("id_cards", 0) + counts.get("passports", 0))

        # Laptop-screen quality: how many masked screen regions came from a
        # refined display quad versus the whole-device fallback. The fallback is
        # privacy-safe but over-covers, and is reported instead of being
        # presented as an accurate screen boundary.
        laptop_screens_redacted = int(counts.get("laptop_screens", 0))
        laptop_quad_regions = int(metrics.get("laptop_screen_quad_regions", 0) or 0)
        laptop_fallback_regions = int(metrics.get("laptop_screen_fallback_regions", 0) or 0)
        laptop_total_regions = laptop_quad_regions + laptop_fallback_regions
        laptop_quad_pct = (round(laptop_quad_regions / laptop_total_regions * 100.0, 1)
                           if laptop_total_regions else 0.0)
        if laptop_screens_redacted == 0:
            laptop_boundary = "not_applicable"
        elif laptop_fallback_regions == 0:
            laptop_boundary = "refined_quad"
        elif laptop_quad_regions == 0:
            laptop_boundary = "device_box_fallback"
        else:
            laptop_boundary = "mixed"

        # Phone-screen quality: how many masked screen regions came from a
        # refined display quad versus the whole-device fallback.
        phone_screens_redacted = int(counts.get("phone_screens", 0))
        phone_quad_regions = int(metrics.get("phone_screen_quad_regions", 0) or 0)
        phone_fallback_regions = int(metrics.get("phone_screen_fallback_regions", 0) or 0)
        phone_total_regions = phone_quad_regions + phone_fallback_regions
        phone_quad_pct = (round(phone_quad_regions / phone_total_regions * 100.0, 1)
                          if phone_total_regions else 0.0)
        if phone_screens_redacted == 0:
            phone_boundary = "not_applicable"
        elif phone_fallback_regions == 0:
            phone_boundary = "refined_quad"
        elif phone_quad_regions == 0:
            phone_boundary = "device_box_fallback"
        else:
            phone_boundary = "mixed"

        return TaskSummaryResponse(
            task_id=task_id,
            faces_blurred=int(counts.get("faces", 0)),
            license_plates_redacted=int(counts.get("license_plates", 0)),
            screens_hidden=int(counts.get("laptop_screens", 0) + counts.get("phone_screens", 0)),
            documents_redacted=documents_redacted,
            laptop_screens_redacted=laptop_screens_redacted,
            laptop_screen_quad_regions=laptop_quad_regions,
            laptop_screen_fallback_regions=laptop_fallback_regions,
            laptop_screen_quad_pct=laptop_quad_pct,
            laptop_screen_boundary=laptop_boundary,
            phone_screens_redacted=phone_screens_redacted,
            phone_screen_quad_regions=phone_quad_regions,
            phone_screen_fallback_regions=phone_fallback_regions,
            phone_screen_quad_pct=phone_quad_pct,
            phone_screen_boundary=phone_boundary,
            # Kept for API backward compatibility. A per-type count is not
            # claimed because the detector does not classify the document type.
            id_cards_found=0,
            phones_found=int(counts.get("phone_screens", 0)),
            total_objects_count=int(sum(counts.values())),
            processing_time_seconds=int(metrics.get("elapsed_seconds", 0) or 0),
            frames_processed=frames_processed,
            frames_with_detections=frames_with_detections,
            detection_frame_pct=detection_frame_pct,
            avg_processing_fps=float(metrics.get("avg_processing_fps", 0.0) or 0.0),
            verified_frame_count=verification.get("frame_count"),
            output_resolution=output_resolution,
            audio_preserved=audio_preserved,
            blur_style=task.get("blur_style"),
            blur_intensity=task.get("blur_intensity"),
            original_video_url=f"/api/v1/videos/original/{task_id}",
            protected_video_url=f"/api/v1/videos/download/{task_id}",
            # No audit-log export exists yet; a URL pointing at a 404 would be
            # misleading, so the field stays null.
            audit_log_url=None,
        )


redact_service = RedactService()
