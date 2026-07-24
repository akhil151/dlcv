from pydantic import BaseModel
from typing import List

class DashboardStats(BaseModel):
    videos_processed: int
    objects_detected: int
    avg_processing_time_seconds: int

class RecentVideoSchema(BaseModel):
    id: str
    filename: str
    file_size_mb: int
    targets: List[str]
    thumbnail_url: str
    created_at: str
    duration: str
    status: str

class DashboardOverviewResponse(BaseModel):
    stats: DashboardStats
    recent_videos: List[RecentVideoSchema]
