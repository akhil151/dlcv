from backend.schemas.dashboard import (
    DashboardOverviewResponse,
    DashboardStats,
    RecentVideoSchema,
)

class DashboardService:
    def get_overview(self) -> DashboardOverviewResponse:
        stats = DashboardStats(
            videos_processed=12,
            objects_detected=1402,
            avg_processing_time_seconds=45,
        )

        recent_videos = [
            RecentVideoSchema(
                id="1",
                filename="interview_research_01.mp4",
                file_size_mb=424,
                targets=["Faces", "Text"],
                thumbnail_url="https://images.unsplash.com/photo-1517245386807-bb43f82c33c4?auto=format&fit=crop&w=400&q=80",
                created_at="12 mins ago",
                duration="01:20",
                status="COMPLETED",
            ),
            RecentVideoSchema(
                id="2",
                filename="surveillance_city_square_04.mp4",
                file_size_mb=812,
                targets=["Faces", "Plates", "Screens"],
                thumbnail_url="https://images.unsplash.com/photo-1509198397868-475647b2a1e5?auto=format&fit=crop&w=400&q=80",
                created_at="1 hour ago",
                duration="04:45",
                status="COMPLETED",
            ),
            RecentVideoSchema(
                id="3",
                filename="confidential_briefing_09.mp4",
                file_size_mb=215,
                targets=["ID Cards", "Screens"],
                thumbnail_url="https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&w=400&q=80",
                created_at="3 hours ago",
                duration="00:55",
                status="COMPLETED",
            ),
        ]

        return DashboardOverviewResponse(stats=stats, recent_videos=recent_videos)

dashboard_service = DashboardService()
