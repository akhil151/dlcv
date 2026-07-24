from pydantic import BaseModel, Field
from typing import Optional

class VideoMetadataResponse(BaseModel):
    video_id: str = Field(..., example="vid_987654321")
    filename: str = Field(..., example="surveillance_city_square.mp4")
    file_size_bytes: int = Field(..., example=142500000)
    format: str = Field(..., example="H.264 / MP4")
    resolution: str = Field(..., example="3840 x 2160 (4K)")
    duration_seconds: int = Field(..., example=180)
    estimated_processing_seconds: int = Field(..., example=252)
