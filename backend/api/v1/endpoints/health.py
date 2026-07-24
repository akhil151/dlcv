from fastapi import APIRouter

router = APIRouter()

@router.get("/health", tags=["System"])
async def get_health():
    return {
        "status": "healthy",
        "service": "SafeFrame AI Backend",
        "version": "1.0.0"
    }
