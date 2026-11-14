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
    # YuNet face detector weights. An absolute path is used as-is; a relative
    # path is resolved against the project root so the backend can be launched
    # from any working directory.
    YUNET_MODEL_PATH: str = "face_detection_yunet_2023mar.onnx"

    # Detection & Confidence Thresholds
    # A face at or above DETECTION_CONF_THRESHOLD is masked on the frame it is
    # detected. 0.60 was chosen from measurements on this checkout: real faces
    # scored 0.86-0.95 across scale (36px to 270px), 45-degree rotation and low
    # light, while YuNet's spurious background boxes on a crowded photo scored
    # 0.36-0.51. YuNet's own internal candidate threshold is the lower secondary
    # value. Raise/lower this deliberately: it trades recall against blur over
    # non-face regions.
    DETECTION_CONF_THRESHOLD: float = 0.60
    SECONDARY_FACE_CONF_THRESHOLD: float = 0.25
    DETECTION_IOU_THRESHOLD: float = 0.45

    # Tracking Parameters (lightweight IoU association)
    # Association uses the same threshold as detection so a detected face is
    # never dropped from redaction merely because it is a weak detection.
    TRACK_CONF_THRESHOLD: float = 0.35
    TRACK_BUFFER_FRAMES: int = 30
    # Minimum IoU required to continue an existing trajectory.
    MATCH_IOU_THRESHOLD: float = 0.30
    # Redaction-box smoothing: below the snap ratio the box is blended towards
    # the measurement (damps detector jitter of a still face); above it the box
    # follows the measurement directly so a moving face is never trailed. The
    # rendered box is always expanded to contain the current detection, so
    # smoothing can only add coverage, never remove it.
    TRACK_SMOOTHING_ALPHA: float = 0.55
    TRACK_SMOOTHING_SNAP_RATIO: float = 0.08

    # Re-Identification (OSNet) is NOT active: no validated weights are
    # configured and no association decision depends on embeddings.
    REID_ENABLED: bool = False
    REID_SIMILARITY_THRESHOLD: float = 0.75
    EMBEDDING_DIM: int = 512

    # Ghost Box Parameters (predictive boxes for recently lost tracks).
    # Disabled by default: the predictor has no validated motion model, and an
    # unvalidated prediction is allowed to hide pixels only for the short hold
    # window below, never with uncontrolled expansion.
    GHOST_BOX_ENABLED: bool = False
    GHOST_MAX_FRAMES: int = 10
    GHOST_EXPANSION_FACTOR: float = 1.02
    GHOST_MOTION_THRESHOLD: float = 2.0

    # Conservative redaction hold: if a confirmed face track is temporarily
    # missed, its last known box is still masked for this many frames. It does
    # not move and does not grow, so it cannot smear over unrelated content.
    REDACTION_HOLD_FRAMES: int = 2

    # Debug frames: annotated detection/ROI images saved separately from the
    # protected video. Never enabled for normal runs.
    DEBUG_SAVE_FRAMES: bool = False
    DEBUG_MAX_FRAMES: int = 300

    # Blur Rendering Parameters
    DEFAULT_BLUR_STYLE: str = "gaussian"
    # Gaussian strength is derived from the target's smaller side so protection
    # scales with the region (and therefore with frame resolution) instead of
    # being a fixed absolute kernel. Sigma = min_side * (MIN_RATIO + GAIN*scale).
    # Measured on this checkout: a face of ~88px was still detected by YuNet at
    # score 0.36-0.55 with sigma 10px, and not detected at all from ~16px. The
    # MIN_RATIO therefore keeps every intensity setting above that floor, and
    # the slider refines strength upward. Do not lower it without re-measuring.
    GAUSSIAN_SIGMA_MIN_RATIO: float = 0.18
    GAUSSIAN_SIGMA_GAIN: float = 0.12
    GAUSSIAN_MIN_SIGMA_PX: float = 6.0
    GAUSSIAN_MAX_KERNEL: int = 201
    # Pixelation bounds the mosaic to at most PIXELATE_MAX_BLOCKS across the
    # target's smaller side, with a minimum block size, so recognizable detail
    # cannot survive at any resolution. 8 blocks was measured on the multi-face
    # validation clip: a 10-block grid left a 52x71 px face re-detectable by
    # YuNet at a 0.30 threshold on 17 of 96 frames, an 8-block grid on none.
    PIXELATE_MAX_BLOCKS: int = 8
    PIXELATE_MIN_BLOCK_PX: int = 8
    # Mask geometry.
    # Every target is expanded outward by a fraction of its OWN width and height
    # (not a fixed pixel rectangle) so the complete sensitive region is masked
    # even when the detector box is slightly tight or jitters frame to frame.
    # Horizontal and vertical padding are separate because a face box is taller
    # than it is wide: the chin/jaw side needs a little more room than the
    # temples, and a fixed square margin either over-covers the sides or clips
    # the chin.
    BLUR_PADDING_X_RATIO: float = 0.12
    BLUR_PADDING_Y_RATIO: float = 0.14
    BLUR_MIN_PADDING_PX: int = 2
    BLUR_MAX_PADDING_PX: int = 64
    # The rendered mask is a superellipse (rounded rectangle) inscribed in the
    # padded ROI instead of a raw rectangle. Exponent 4 keeps the rounded shape
    # wide enough to contain the WHOLE detected face box (corners included) at
    # these padding ratios, so the rounded silhouette never trades coverage for
    # looks. Blur/pixelation fade smoothly from the face box edge to that
    # outline; blackout uses the same outline with a hard edge.
    MASK_SUPERELLIPSE_EXPONENT: float = 4.0
    # Fraction of the padding ring that stays fully filtered before the soft
    # ramp starts. The detector box is regularly a few pixels tighter than the
    # real face edge (YuNet under-reports wide cheeks/jaw), so most of the ring
    # must stay solid; only the outer third fades into the background.
    MASK_SOLID_MARGIN_RATIO: float = 0.65

    # Memory & Streaming Constraints
    MAX_MEMORY_BUFFER_FRAMES: int = 32

    # FFmpeg Encoding Parameters
    # FFMPEG_PATH: optional explicit executable path. Empty means "resolve via PATH".
    FFMPEG_PATH: str = ""
    FFMPEG_PRESET: str = "fast"
    FFMPEG_CRF: int = 23

ai_settings = AISettings()
