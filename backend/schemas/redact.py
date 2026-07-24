from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class RedactionCategoryEnum(str, Enum):
    FACES = "faces"
    LICENSE_PLATES = "license_plates"
    ID_CARDS = "id_cards"
    LAPTOP_SCREENS = "laptop_screens"
    PHONE_SCREENS = "phone_screens"

class BlurStyleEnum(str, Enum):
    GAUSSIAN = "gaussian"
    PIXELATE = "pixelate"
    BLACKOUT = "blackout"

class TaskStatusEnum(str, Enum):
    PENDING = "PENDING"
    UPLOADING = "UPLOADING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class StartRedactionRequest(BaseModel):
    video_id: str = Field(..., example="vid_987654321")
    target_categories: List[RedactionCategoryEnum] = Field(
        default=[
            RedactionCategoryEnum.FACES,
            RedactionCategoryEnum.LICENSE_PLATES,
            RedactionCategoryEnum.ID_CARDS,
            RedactionCategoryEnum.LAPTOP_SCREENS,
            RedactionCategoryEnum.PHONE_SCREENS,
        ]
    )
    blur_style: BlurStyleEnum = Field(default=BlurStyleEnum.GAUSSIAN)
    blur_intensity: int = Field(default=80, ge=10, le=100)

class StartRedactionResponse(BaseModel):
    task_id: str = Field(..., example="task_123456789")
    status: TaskStatusEnum = Field(default=TaskStatusEnum.PROCESSING)

class BoundingBoxSchema(BaseModel):
    x: float
    y: float
    width: float
    height: float

class DetectedEntitySchema(BaseModel):
    id: str
    category: RedactionCategoryEnum
    label: str
    confidence: float
    bbox: BoundingBoxSchema

class ProcessingMetricsSchema(BaseModel):
    fps: int
    elapsed_seconds: int
    remaining_seconds: int
    throughput_frames: int
    total_frames: int
    processed_frames: int

class TaskStatusResponse(BaseModel):
    task_id: str
    video_id: str
    filename: str
    status: TaskStatusEnum
    progress_pct: float
    target_categories: List[RedactionCategoryEnum]
    blur_style: BlurStyleEnum
    blur_intensity: int
    metrics: ProcessingMetricsSchema
    recent_logs: List[str]

class TaskSummaryResponse(BaseModel):
    task_id: str
    faces_blurred: int
    license_plates_redacted: int
    screens_hidden: int
    id_cards_found: int
    phones_found: int
    total_objects_count: int
    processing_time_seconds: int
    original_video_url: Optional[str] = None
    protected_video_url: Optional[str] = None
    audit_log_url: Optional[str] = None
