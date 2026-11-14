from pydantic import BaseModel, Field
from typing import Optional

class VideoMetadataResponse(BaseModel):
    video_id: str = Field(..., example="vid_987654321")
    filename: str = Field(..., example="surveillance_city_square.mp4")
    file_size_bytes: int = Field(..., example=142500000)
    format: str = Field(..., example="MP4 container")
    resolution: str = Field(..., example="1920 x 1080")
    duration_seconds: int = Field(..., example=180)
    fps: float = Field(default=0.0, example=30.0)
    frame_count: int = Field(default=0, example=5400)
    estimated_processing_seconds: int = Field(..., example=252)
