from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class RedactionCategoryEnum(str, Enum):
    FACES = "faces"
    LICENSE_PLATES = "license_plates"
    ID_CARDS = "id_cards"
    PASSPORTS = "passports"
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
    CANCELLED = "CANCELLED"

class StartRedactionRequest(BaseModel):
    video_id: str = Field(..., example="vid_987654321")
    # Face redaction is the only validated capability. The other categories are
    # rejected explicitly by the detector preflight rather than silently ignored.
    target_categories: List[RedactionCategoryEnum] = Field(
        default=[RedactionCategoryEnum.FACES]
    )
    blur_style: BlurStyleEnum = Field(default=BlurStyleEnum.GAUSSIAN)
    blur_intensity: int = Field(default=80, ge=10, le=100)
    # Saves annotated/original/protected frames under storage/debug/<task_id>/.
    # Those images are never part of the protected video stream.
    debug_mode: bool = Field(default=False)

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
    frames_with_detections: int = 0
    masks_total: int = 0
    new_tracks_total: int = 0
    detection_latency_ms: float = 0.0
    blur_latency_ms: float = 0.0
    avg_processing_fps: float = 0.0
    debug_frames_saved: int = 0

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
    created_at: Optional[str] = None
    debug_mode: bool = False
    # Present only for a terminal processing failure. This makes asynchronous
    # detector/encoder failures explicit without changing successful payloads.
    failure_reason: Optional[str] = None
    # Additive: lets clients require a verified output before offering a
    # download, instead of inferring success from status alone.
    output_verified: Optional[bool] = None

class TaskSummaryResponse(BaseModel):
    task_id: str
    # Region-frames: the number of face regions masked across all frames.
    faces_blurred: int
    license_plates_redacted: int
    screens_hidden: int
    # Document regions masked across all frames. The document detector is
    # class-agnostic, so ID cards and passports are counted together here; the
    # legacy per-type fields stay 0 rather than reporting unverified type counts.
    documents_redacted: int = 0
    # Laptop-screen region-frames masked by the Phase 4 screen detector. The
    # quad/fallback counts make the display-boundary quality explicit: a
    # fallback means the whole device box was masked because the screen
    # boundary could not be reliably separated.
    laptop_screens_redacted: int = 0
    laptop_screen_quad_regions: int = 0
    laptop_screen_fallback_regions: int = 0
    laptop_screen_quad_pct: float = 0.0
    laptop_screen_boundary: str = "not_applicable"
    # Phone-screen region-frames masked by the Phase 5 phone screen detector.
    phone_screens_redacted: int = 0
    phone_screen_quad_regions: int = 0
    phone_screen_fallback_regions: int = 0
    phone_screen_quad_pct: float = 0.0
    phone_screen_boundary: str = "not_applicable"
    id_cards_found: int
    phones_found: int
    total_objects_count: int
    processing_time_seconds: int
    frames_processed: int = 0
    frames_with_detections: int = 0
    detection_frame_pct: float = 0.0
    avg_processing_fps: float = 0.0
    verified_frame_count: Optional[int] = None
    output_resolution: Optional[str] = None
    audio_preserved: Optional[bool] = None
    blur_style: Optional[str] = None
    blur_intensity: Optional[int] = None
    original_video_url: Optional[str] = None
    protected_video_url: Optional[str] = None
    audit_log_url: Optional[str] = None
