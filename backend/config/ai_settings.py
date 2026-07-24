import os
from typing import List, Optional

try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
except ImportError:
    try:
        from pydantic import BaseModel as BaseSettings, Field
    except ImportError:
        # Fallback dataclass configuration if pydantic is absent
        class BaseSettings:
            pass
        def Field(default=None, **kwargs):
            return default

class AISettings(BaseSettings):
    """
    Centralized configuration settings for SafeFrame AI Pipeline models,
    detection thresholds, hardware acceleration, and tracking settings.
    """
    DEVICE: str = "cuda"
    FORCE_CPU: bool = False
    HALF_PRECISION: bool = True

    # Model Paths
    YOLO_MODEL_PATH: str = "yolo11m.pt"
    REID_MODEL_NAME: str = "osnet_x0_25"
    FACE_DETECTOR_MODEL: str = "yunet"

    # Detection & Confidence Thresholds
    DETECTION_CONF_THRESHOLD: float = 0.35
    SECONDARY_FACE_CONF_THRESHOLD: float = 0.25
    DETECTION_IOU_THRESHOLD: float = 0.45

    # BoT-SORT Tracking Parameters
    TRACK_CONF_THRESHOLD: float = 0.50
    TRACK_BUFFER_FRAMES: int = 30
    MATCH_THRESHOLD: float = 0.80

    # Re-Identification (OSNet) Parameters
    REID_ENABLED: bool = True
    REID_SIMILARITY_THRESHOLD: float = 0.75
    EMBEDDING_DIM: int = 512

    # Ghost Box Parameters (Kalman Motion Predictor)
    GHOST_BOX_ENABLED: bool = True
    GHOST_MAX_FRAMES: int = 20
    GHOST_EXPANSION_FACTOR: float = 1.08
    GHOST_MOTION_THRESHOLD: float = 2.0

    # Blur Rendering Parameters
    DEFAULT_BLUR_STYLE: str = "gaussian"
    GAUSSIAN_KERNEL_SIZE: int = 51
    PIXELATE_BLOCK_SIZE: int = 16
    FEATHER_RADIUS: int = 10

    # Memory & Streaming Constraints
    MAX_MEMORY_BUFFER_FRAMES: int = 32

ai_settings = AISettings()
