from typing import List, Optional, Dict, Any
from enum import Enum
from backend.schemas.detection import NormalizedBoundingBox, TargetCategory

try:
    from pydantic import BaseModel, Field
except ImportError:
    class BaseModel:
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
                elif hasattr(v, 'value'):
                    res[k] = v.value
                else:
                    res[k] = v
            return res

    def Field(default=..., **kwargs):
        return default

class TrackStateEnum(str, Enum):
    NEW = "NEW"
    TRACKED = "TRACKED"
    LOST = "LOST"
    REMOVED = "REMOVED"

class TrackedObjectItem(BaseModel):
    """
    Strongly-typed object representing a single tracked entity across video frames.
    """
    track_id: int = Field(..., ge=1, description="Persistent unique trajectory ID")
    category: TargetCategory
    label: str = Field(..., example="FACE #12")
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: NormalizedBoundingBox
    track_state: TrackStateEnum = Field(default=TrackStateEnum.TRACKED)
    age: int = Field(..., ge=1, description="Total consecutive frames track has existed")
    frames_since_update: int = Field(..., ge=0, description="Frames since last detection match")
    is_confirmed: bool = Field(default=True, description="True if track has surpassed min hit threshold")

class TrackingFrameResult(BaseModel):
    """
    Per-frame output container from TrackingService.
    """
    frame_index: int
    active_tracks_count: int
    new_tracks_count: int
    removed_tracks_count: int
    id_switches_count: int
    tracking_latency_ms: float
    tracks: List[TrackedObjectItem]
