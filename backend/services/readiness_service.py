"""Backend readiness composition.

Distinguishes a running process (liveness, see /health) from a fully
operational redaction engine (this snapshot). The check is cheap: FFmpeg is
probed once and cached, and the detector dependency is inspected via a
filesystem probe unless the models are already loaded. No inference is run and
no heavy model is loaded on behalf of a health request.
"""
from typing import Any, Dict

from backend.services.ai import model_manager
from backend.utils import ffmpeg as ffmpeg_util


class ReadinessService:
    def snapshot(self) -> Dict[str, Any]:
        ffmpeg = ffmpeg_util.validate_ffmpeg()
        detector = model_manager.detector_readiness()

        reasons = []
        if not detector["available"]:
            if detector.get("status") == "ERROR":
                reasons.append("face detector failed to initialize (invalid or incompatible weights)")
            else:
                reasons.append("face detector weights are not available")
        if not ffmpeg["available"]:
            reasons.append("ffmpeg executable is not available")

        processing_ready = bool(detector["available"] and ffmpeg["available"])
        return {
            "status": "ready" if processing_ready else "unavailable",
            "process": "running",
            "models_loaded": model_manager.models_loaded,
            "detector": {
                "name": "YUNET",
                "available": bool(detector["available"]),
                "status": detector.get("status"),
                "device": "cpu",
            },
            "ffmpeg": {
                "available": bool(ffmpeg["available"]),
                "version": ffmpeg.get("version"),
            },
            "processing_ready": processing_ready,
            "reasons": reasons,
        }


readiness_service = ReadinessService()
