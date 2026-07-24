from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from backend.schemas.detection import TargetCategory

class BlurModeEnum(str, Enum):
    GAUSSIAN = "gaussian"
    PIXELATE = "pixelate"
    BLACKOUT = "blackout"
    FEATHERED = "feathered"

class BlurROIPlan(BaseModel):
    """
    Clipped pixel coordinate ROI bounding box for a single blur filter operation.
    """
    x1: int = Field(..., ge=0, description="Top-left X pixel coordinate")
    y1: int = Field(..., ge=0, description="Top-left Y pixel coordinate")
    x2: int = Field(..., ge=0, description="Bottom-right X pixel coordinate")
    y2: int = Field(..., ge=0, description="Bottom-right Y pixel coordinate")
    category: TargetCategory
    is_ghost: bool = Field(default=False)

class BlurFrameResult(BaseModel):
    """
    Telemetry container returned by BlurService.
    """
    frame_index: int
    blur_mode: BlurModeEnum
    blur_intensity: int
    total_blurred_regions: int
    active_tracks_blurred: int
    ghost_regions_blurred: int
    blur_latency_ms: float
    roi_plans: List[BlurROIPlan]
