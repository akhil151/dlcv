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
    # Un-padded rectangle that tightly contains the requested target. The region
    # between the core and the padded ROI is margin; feathered rendering may only
    # blend original pixels in that margin, never inside the core.
    core_x1: Optional[int] = Field(default=None, ge=0)
    core_y1: Optional[int] = Field(default=None, ge=0)
    core_x2: Optional[int] = Field(default=None, ge=0)
    core_y2: Optional[int] = Field(default=None, ge=0)
    # Four-corner polygon in frame pixels (laptop screens). When present the
    # mask is this polygon grown outward by ``pad`` pixels, not a rectangle.
    quad: Optional[List[List[float]]] = Field(default=None)
    region_kind: str = Field(default="box")
    pad: int = Field(default=0, ge=0)

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
