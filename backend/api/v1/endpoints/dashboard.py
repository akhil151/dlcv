from fastapi import APIRouter
from backend.schemas.dashboard import DashboardOverviewResponse
from backend.services.dashboard_service import dashboard_service

router = APIRouter()

@router.get("/dashboard/overview", response_model=DashboardOverviewResponse, tags=["Dashboard"])
async def get_dashboard_overview():
    """Retrieves aggregate platform statistics and recent video processing logs."""
    return dashboard_service.get_overview()
