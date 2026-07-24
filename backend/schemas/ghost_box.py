from typing import List, Optional, Dict, Any
from backend.schemas.detection import NormalizedBoundingBox, TargetCategory

try:
    from pydantic import BaseModel, Field  # type: ignore
except ImportError:
    class BaseModel:  # type: ignore
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def dict(self) -> Dict[str, Any]:
            res = {}
            for k, v in self.__dict__.items():
                if hasattr(v, 'dict'):
                    res[k] = v.dict()
                elif isinstance(v, list):
                    res[k] = [i.dict() if hasattr(i, 'dict') else (i.value if hasattr(i, 'value') else i) for i in v]
                else:
                    res[k] = v
            return res

    def Field(default=..., **kwargs):
        return default

class GhostBoxItem(BaseModel):
    """
    Strongly-typed Pydantic model representing a single predictive Ghost Box.
    """
    original_track_id: int = Field(..., description="Historical track ID being predicted")
    category: TargetCategory
    label: str = Field(..., example="GHOST FACE #12")
    predicted_bbox: NormalizedBoundingBox = Field(..., description="Predicted and expanded bounding box")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Decayed prediction confidence score")
    ghost_age: int = Field(..., ge=1, description="Consecutive frames object has been in Ghost state")
    covariance_norm: float = Field(..., ge=0.0, description="Norm of Kalman error covariance matrix P")
    expansion_factor: float = Field(..., ge=1.0, description="Current spatial expansion multiplier")
    active: bool = Field(default=True, description="True if Ghost Box is active")

class GhostBoxFrameResult(BaseModel):
    """
    Per-frame telemetry output from GhostBoxService.
    """
    frame_index: int
    active_ghosts_count: int
    new_ghosts_count: int
    recovered_ghosts_count: int
    expired_ghosts_count: int
    prediction_latency_ms: float
    ghost_boxes: List[GhostBoxItem]
