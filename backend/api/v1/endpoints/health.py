import asyncio
from typing import Any, Dict

from fastapi import APIRouter

from backend.services.readiness_service import readiness_service

router = APIRouter()


@router.get("/health", tags=["System"])
async def get_health() -> Dict[str, Any]:
    """Liveness: the process is up.

    Intentionally independent of model/encoder readiness so basic diagnostics
    keep working when a dependency is missing.
    """
    return {
        "status": "healthy",
        "service": "SafeFrame AI Backend",
        "version": "1.0.0"
    }


@router.get("/health/readiness", tags=["System"])
async def get_readiness() -> Dict[str, Any]:
    """Readiness: whether the redaction engine can actually process a video.

    Reports real dependency state (face detector availability and FFmpeg
    availability/version). The probe is cached and executed off the event loop.
    """
    return await asyncio.to_thread(readiness_service.snapshot)
