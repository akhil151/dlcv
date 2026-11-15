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
    # ID cards and passports are both located by the same validated,
    # class-agnostic document detector. Detections are attributed to ID_CARDS
    # internally; the requested categories decide which detector runs.
    ID_CARDS = "id_cards"
    PASSPORTS = "passports"
    LAPTOP_SCREENS = "laptop_screens"
    PHONE_SCREENS = "phone_screens"

class DetectorSource(str, Enum):
    YOLO11M = "yolo11m"
    SECONDARY_FACE = "secondary_face"
    RTDETR_PLATE = "rtdetr_plate"
    DOCUMENT_CORNER = "document_corner"
    RTDETR_LAPTOP = "rtdetr_laptop"
    RTDETR_PHONE = "rtdetr_phone"

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
    # Four display corners in frame pixels, when the detector produced a
    # polygon (laptop screens). Empty for box-only categories.
    quad: List[List[float]] = Field(default_factory=list)
    # "screen_quad" when the region is a refined display boundary,
    # "laptop_fallback" when the whole device box had to be masked, "box"
    # otherwise. Reported so a fallback is never presented as a screen match.
    region_kind: str = "box"

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
    # Additive: whether the plate detector ran for this frame (independent of
    # whether it found a plate).
    plate_detector_used: bool = False
    # Additive: whether the identity-document detector ran for this frame
    # (independent of whether it found a document).
    document_detector_used: bool = False
    # Additive: whether the laptop-screen detector ran for this frame.
    laptop_detector_used: bool = False
    # Additive: wall time spent in the laptop model itself (ms).
    laptop_inference_ms: float = 0.0
    # Additive: whether the phone-screen detector ran for this frame.
    phone_detector_used: bool = False
    # Additive: wall time spent in the phone model itself (ms).
    phone_inference_ms: float = 0.0
