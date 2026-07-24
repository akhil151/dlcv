from typing import List, Optional, Dict, Any
from enum import Enum

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

class TargetCategory(str, Enum):
    FACES = "faces"
    LICENSE_PLATES = "license_plates"
    ID_CARDS = "id_cards"
    LAPTOP_SCREENS = "laptop_screens"
    PHONE_SCREENS = "phone_screens"

class DetectorSource(str, Enum):
    YOLO11M = "yolo11m"
    SECONDARY_FACE = "secondary_face"

class NormalizedBoundingBox(BaseModel):
    """
    Normalized bounding box relative to frame dimensions [0.0 to 100.0 percent].
    """
    x: float = Field(..., ge=0.0, le=100.0, description="Center or top-left X coordinate percentage")
    y: float = Field(..., ge=0.0, le=100.0, description="Center or top-left Y coordinate percentage")
    width: float = Field(..., ge=0.0, le=100.0, description="Width percentage")
    height: float = Field(..., ge=0.0, le=100.0, description="Height percentage")

class DetectionItem(BaseModel):
    """
    Single detected sensitive entity.
    """
    category: TargetCategory
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: NormalizedBoundingBox
    source: DetectorSource = DetectorSource.YOLO11M

class DetectionResultContainer(BaseModel):
    """
    Container for frame detection results and performance telemetry.
    """
    frame_index: int
    total_detections: int
    detections: List[DetectionItem]
    inference_time_ms: float
    categories_detected: List[TargetCategory]
    secondary_detector_used: bool = False
