from fastapi import APIRouter
from backend.api.v1.endpoints import health, videos, redact, dashboard

api_v1_router = APIRouter()

api_v1_router.include_router(health.router)
api_v1_router.include_router(videos.router)
api_v1_router.include_router(redact.router)
api_v1_router.include_router(dashboard.router)
