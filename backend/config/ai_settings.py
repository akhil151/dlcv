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
    # License-plate detector: RT-DETRv2 (R18) fine-tuned on the Open Images V7
    # "Vehicle registration plate" class, exported to ONNX (Apache-2.0).
    # Downloaded and checksum-verified by scripts/download_plate_model.py.
    PLATE_DETECTOR_MODEL: str = "rtdetr_v2_r18_plate"
    PLATE_MODEL_PATH: str = "plate_detection_rtdetr_r18.onnx"
    PLATE_DETECTOR_INPUT_SIZE: int = 640
    # Measured on the 40-image CC-BY-4.0 plate validation split used by
    # scripts/evaluate_plate_detection.py: threshold 0.10 gave precision 0.889 /
    # recall 0.909 at IoU>=0.50 (the model emits compressed single-class
    # scores, so 0.10 is high-confidence for this graph, not a weak cutoff).
    PLATE_CONF_THRESHOLD: float = 0.10
    PLATE_NMS_IOU_THRESHOLD: float = 0.45
    PLATE_MAX_DETECTIONS: int = 50
    # Plate mask padding, proportional to the plate's own width/height. Plates
    # are wide and short, so vertical padding is larger: plate characters sit
    # close to the top/bottom edge and a detector box is often a few pixels
    # tight there. Horizontal padding stays small to avoid covering the vehicle.
    PLATE_PADDING_X_RATIO: float = 0.06
    PLATE_PADDING_Y_RATIO: float = 0.20
    PLATE_MIN_PADDING_PX: int = 2
    PLATE_MAX_PADDING_PX: int = 48

    # Identity-document (ID card / passport) detector.
    # DocAligner FastViT-SA24 corner-heatmap regressor (Apache-2.0, DocsaidLab),
    # trained on MIDV-500/MIDV-2020 identity-document captures. The ONNX graph is
    # re-hosted by 7RPlus GmbH (pagescan-weights, Apache-2.0) and downloaded +
    # SHA-256 verified by scripts/download_document_model.py.
    # The graph is class-agnostic: it locates a document quadrilateral, it does
    # NOT label ID cards vs passports. Both UI categories run this one validated
    # detector, and every detected document is masked in full.
    DOCUMENT_DETECTOR_MODEL: str = "DOCALIGNER_FASTVIT_SA24"
    DOCUMENT_MODEL_PATH: str = "document_detection_docaligner_fastvit.onnx"
    DOCUMENT_DETECTOR_INPUT_SIZE: int = 256
    DOCUMENT_HEATMAP_OUTPUT_SIZE: int = 128
    # Heatmap pixels below this are background. Measured on the 60-frame MIDV
    # validation split: true corners peak ~0.9, non-document frames ~0.07.
    DOCUMENT_HEATMAP_THRESHOLD: float = 0.30
    # Minimum mean corner-heatmap peak for a detection to count. The same
    # measurement found a clear gap between document and background peaks.
    DOCUMENT_CONF_THRESHOLD: float = 0.30
    # Downstream geometry sanity: a detected document must occupy between 0.2%
    # and 90% of the frame and have a plausible aspect ratio. This rejects
    # full-frame "document" hallucinations and banner-shaped rectangles.
    DOCUMENT_MIN_AREA_RATIO: float = 0.002
    DOCUMENT_MAX_AREA_RATIO: float = 0.90
    DOCUMENT_MIN_ASPECT_RATIO: float = 0.30
    DOCUMENT_MAX_ASPECT_RATIO: float = 3.30
    # DocAligner returns one document per inference. Overlapping half-strips are
    # scanned as well so several documents in one frame can be found, and so a
    # partially visible document near a frame edge gets a second chance. This is
    # real inference, not interpolation. Each extra strip costs one inference.
    DOCUMENT_MULTI_SCAN_ENABLED: bool = True
    DOCUMENT_SCAN_STRIP_OVERLAP: float = 0.24
    DOCUMENT_MAX_DOCS_PER_FRAME: int = 3
    # Two detected quads whose masks overlap this much are the same document.
    DOCUMENT_MERGE_MASK_IOU: float = 0.35
    # Document mask padding, proportional to the document's own width/height.
    # Documents carry sensitive text and a portrait right up to the edge, so the
    # padding is smaller than a face's but still guarantees the border is covered.
    DOCUMENT_PADDING_X_RATIO: float = 0.06
    DOCUMENT_PADDING_Y_RATIO: float = 0.06
    DOCUMENT_MIN_PADDING_PX: int = 4
    DOCUMENT_MAX_PADDING_PX: int = 64
    # Documents are text-dense: the 4-block mosaic cap used for plates also
    # keeps document text unreadable.
    DOCUMENT_PIXELATE_MAX_BLOCKS: int = 4
    DOCUMENT_REDACTION_HOLD_FRAMES: int = 3

    # Laptop-screen detector (Phase 4).
    # Device proposal: RT-DETR R18 COCO detector, Apache-2.0 source weights
    # (PekingU/rtdetr_r18vd), ONNX export hosted as
    # onnx-community/rtdetr_r18vd and run through onnxruntime. The graph
    # localises the LAPTOP, not the display: the visible-screen quadrilateral is
    # estimated by the dedicated refinement stage in
    # laptop_screen_detection_service.py. When refinement is not confident the
    # whole device box is masked instead and reported as a fallback, so the
    # pipeline never claims a screen boundary it did not find.
    LAPTOP_DETECTOR_MODEL: str = "RTDETR_R18VD_COCO"
    LAPTOP_MODEL_PATH: str = "laptop_screen_detection_rtdetr_r18.onnx"
    # COCO label index used by the pinned RT-DETR graph (config id2label[63]).
    LAPTOP_CLASS_INDEX: int = 63
    LAPTOP_DETECTOR_INPUT_SIZE: int = 640
    # COCO scores from this graph are true class probabilities: laptop photos in
    # the validation set score 0.53-0.98 and the first non-laptop competitors
    # stay below 0.31. 0.35 keeps weak background "laptops" out while still
    # catching a partially occluded device.
    LAPTOP_CONF_THRESHOLD: float = 0.35
    LAPTOP_NMS_IOU_THRESHOLD: float = 0.50
    LAPTOP_MAX_DETECTIONS: int = 10
    LAPTOP_MIN_AREA_RATIO: float = 0.003
    LAPTOP_MAX_AREA_RATIO: float = 0.98
    # Screen-quad refinement. The device crop is scaled down to this longest
    # side before edge/line analysis so parameters are resolution independent.
    LAPTOP_SCREEN_REFINE_SCALE: int = 480
    # A display covers a large part of the device box and starts at its top
    # edge. These gates reject inner windows/panels and "quads" that drift into
    # the keyboard deck; anything rejected falls back to the device box.
    LAPTOP_SCREEN_MIN_COVERAGE: float = 0.18
    LAPTOP_SCREEN_MAX_COVERAGE: float = 1.05
    LAPTOP_SCREEN_MIN_ASPECT: float = 1.00
    LAPTOP_SCREEN_MAX_ASPECT: float = 2.75
    LAPTOP_SCREEN_TOP_TOLERANCE: float = 0.13
    LAPTOP_SCREEN_MIN_BOTTOM: float = 0.30
    # The lid occupies the upper part of a device box. A candidate whose bottom
    # reaches further down is usually covering the keyboard deck, so it is
    # rejected and the device-box fallback is used instead.
    LAPTOP_SCREEN_MAX_BOTTOM: float = 0.70
    # The bottom of the accepted quad must be backed by a detected line that
    # spans this fraction of the quad's own bottom edge. Without it the "quad"
    # is usually an inner window and would leave display content below visible.
    LAPTOP_SCREEN_MIN_BOTTOM_SPAN: float = 0.55
    # Fraction of the quad's perimeter that must lie on an image edge.
    LAPTOP_SCREEN_MIN_EDGE_SUPPORT: float = 0.70
    # Measured on the validation photos: refined quads clear 0.34, while the
    # under-covering inner-window case sits at 0.12. A genuine low-contrast
    # display therefore falls back to the device box (privacy-safe).
    LAPTOP_SCREEN_MIN_BORDER_CONTRAST: float = 0.18
    LAPTOP_SCREEN_MIN_SCORE: float = 0.80
    # Screen masks use a hard, dilated polygon: padding is a fraction of the
    # screen's own size and grows the mask outward, never inward.
    LAPTOP_PADDING_RATIO: float = 0.04
    LAPTOP_MIN_PADDING_PX: int = 4
    LAPTOP_MAX_PADDING_PX: int = 48
    # Screen content is high-contrast text/UI: the 4-block mosaic cap used for
    # plates and documents is reused so no readable glyph survives.
    LAPTOP_PIXELATE_MAX_BLOCKS: int = 4
    LAPTOP_REDACTION_HOLD_FRAMES: int = 2

    # Phone-screen detector (Phase 5).
    # Device proposal: RT-DETR R18 COCO detector, Apache-2.0 source weights
    # (PekingU/rtdetr_r18vd), ONNX export hosted as
    # onnx-community/rtdetr_r18vd and run through onnxruntime. The graph
    # localises the CELL PHONE (class 67), not the display: the visible-screen
    # quadrilateral is estimated by the dedicated refinement stage in
    # phone_screen_detection_service.py. When refinement is not confident the
    # whole device box is masked instead and reported as a fallback, so the
    # pipeline never claims a screen boundary it did not find.
    PHONE_DETECTOR_MODEL: str = "RTDETR_R18VD_COCO_PHONE"
    PHONE_MODEL_PATH: str = "phone_screen_detection_rtdetr_r18.onnx"
    # COCO label index used by the pinned RT-DETR graph (config id2label[67]).
    PHONE_CLASS_INDEX: int = 67
    PHONE_DETECTOR_INPUT_SIZE: int = 640
    # COCO scores from this graph for phones: confident phones score 0.40-0.98.
    # 0.30 keeps weak background clutter out while catching partially occluded or
    # handheld phones.
    PHONE_CONF_THRESHOLD: float = 0.30
    PHONE_NMS_IOU_THRESHOLD: float = 0.50
    PHONE_MAX_DETECTIONS: int = 15
    PHONE_MIN_AREA_RATIO: float = 0.001
    PHONE_MAX_AREA_RATIO: float = 0.98
    # Screen-quad refinement. The phone crop is scaled down to this longest
    # side before edge/line analysis so parameters are resolution independent.
    PHONE_SCREEN_REFINE_SCALE: int = 480
    # A phone display covers the central/inner area of the front face (typically
    # 40% to 95%). These gates reject small icons/buttons and candidates that
    # drift into the user's hand/arm.
    PHONE_SCREEN_MIN_COVERAGE: float = 0.28
    PHONE_SCREEN_MAX_COVERAGE: float = 1.05
    # Supports portrait and landscape: phone display aspect ratios (max/min side)
    # sit between 1.10 (square-ish / tilted) and 3.20 (tall 21:9 or perspective).
    PHONE_SCREEN_MIN_ASPECT: float = 1.10
    PHONE_SCREEN_MAX_ASPECT: float = 3.20
    # Quad center must be within this fraction of the crop size from the phone box center.
    PHONE_SCREEN_MAX_CENTER_DIST: float = 0.35
    # Fraction of the quad's perimeter that must lie on an image edge.
    PHONE_SCREEN_MIN_EDGE_SUPPORT: float = 0.38
    # Minimum luminance step across the quad border (display content vs bezel/hand).
    PHONE_SCREEN_MIN_BORDER_CONTRAST: float = 0.06
    PHONE_SCREEN_MIN_SCORE: float = 0.55
    # Screen masks use a hard, dilated polygon: padding grows the mask outward only.
    PHONE_PADDING_RATIO: float = 0.04
    PHONE_MIN_PADDING_PX: int = 3
    PHONE_MAX_PADDING_PX: int = 36
    # Screen content is high-contrast text/UI: the 4-block mosaic cap keeps text unreadable.
    PHONE_PIXELATE_MAX_BLOCKS: int = 4
    PHONE_REDACTION_HOLD_FRAMES: int = 2

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
    # Same conservative hold for plates. It only bridges a one/two frame
    # detector gap; the box neither moves nor grows, so it cannot mask a
    # neighbouring vehicle or a different part of the frame.
    PLATE_REDACTION_HOLD_FRAMES: int = 2

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
    # Plates are short and high-contrast: the face-tuned 8-block grid still left
    # a plate-shaped mosaic detected at 0.108-0.126 on the comparison photo and a
    # panning clip. A 4-block cap produced zero re-detections in both. Faces keep
    # their measured 8-block grid.
    PLATE_PIXELATE_MAX_BLOCKS: int = 4
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
