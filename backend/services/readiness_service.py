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
        plate_detector = model_manager.plate_detector_readiness()
        document_detector = model_manager.document_detector_readiness()
        laptop_detector = model_manager.laptop_detector_readiness()
        phone_detector = model_manager.phone_detector_readiness()

        reasons = []
        if not detector["available"]:
            if detector.get("status") == "ERROR":
                reasons.append("face detector failed to initialize (invalid or incompatible weights)")
            else:
                reasons.append("face detector weights are not available")
        if not ffmpeg["available"]:
            reasons.append("ffmpeg executable is not available")

        # Face detection remains the baseline requirement (a face-only job must
        # stay runnable); plate readiness is reported separately so a missing
        # optional plate model does not mark the whole engine unavailable.
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
            "plate_detector": {
                "name": "RTDETR_R18_PLATE",
                "available": bool(plate_detector["available"]),
                "status": plate_detector.get("status"),
                "device": "cpu",
            },
            "license_plates_ready": bool(plate_detector["available"]),
            "document_detector": {
                "name": "DOCALIGNER_FASTVIT_SA24",
                "available": bool(document_detector["available"]),
                "status": document_detector.get("status"),
                "device": "cpu",
            },
            # Both document categories share this one validated detector.
            "documents_ready": bool(document_detector["available"]),
            "laptop_detector": {
                "name": "RTDETR_R18VD_COCO",
                "available": bool(laptop_detector["available"]),
                "status": laptop_detector.get("status"),
                "device": "cpu",
            },
            "laptop_screens_ready": bool(laptop_detector["available"]),
            "phone_detector": {
                "name": "RTDETR_R18VD_COCO_PHONE",
                "available": bool(phone_detector["available"]),
                "status": phone_detector.get("status"),
                "device": "cpu",
            },
            "phone_screens_ready": bool(phone_detector["available"]),
            "ffmpeg": {
                "available": bool(ffmpeg["available"]),
                "version": ffmpeg.get("version"),
            },
            "processing_ready": processing_ready,
            "reasons": reasons,
        }


readiness_service = ReadinessService()
