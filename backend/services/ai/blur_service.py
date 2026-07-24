import cv2
import time
import numpy as np
from typing import List, Tuple, Optional, Any
from backend.config.ai_settings import ai_settings
from backend.schemas.tracking import TrackedObjectItem, TrackStateEnum
from backend.schemas.ghost_box import GhostBoxItem
from backend.schemas.blur import BlurModeEnum, BlurROIPlan, BlurFrameResult
from backend.core.logging import logger

class BlurService:
    """
    High-Performance ROI-Restricted Image Anonymization Service for SafeFrame AI Engine.
    
    Responsibilities:
    - Merges active tracked entities and predictive Ghost Boxes into a unified ROI execution plan.
    - Applies Gaussian Blur, Pixelation, Blackout, or Feathered Blur strictly inside ROI slices in-place.
    - Safely clips bounding box coordinates to image boundaries [0, width] and [0, height].
    - Handles overlapping, nested, or out-of-bounds regions seamlessly.
    """

    def __init__(self) -> None:
        self.default_style = ai_settings.DEFAULT_BLUR_STYLE
        self.gaussian_ksize = ai_settings.GAUSSIAN_KERNEL_SIZE
        self.pixelate_block = ai_settings.PIXELATE_BLOCK_SIZE
        self.feather_radius = ai_settings.FEATHER_RADIUS

    def render_blur(
        self,
        frame: Optional[np.ndarray],
        tracks: List[TrackedObjectItem],
        ghosts: List[GhostBoxItem],
        blur_style: str = "gaussian",
        blur_intensity: int = 80,
        frame_index: int = 0,
    ) -> Tuple[np.ndarray, BlurFrameResult]:
        """
        Applies anonymization filters to target ROI regions inside the frame.
        Returns (anonymized_frame, BlurFrameResult).
        """
        start_time = time.time()

        # Parse requested blur mode enum
        try:
            mode = BlurModeEnum(blur_style.lower())
        except ValueError:
            mode = BlurModeEnum.GAUSSIAN

        # Edge Case 1: Empty or corrupted frame
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            logger.warning(f"[BlurService] Frame #{frame_index}: Empty frame buffer provided.")
            empty_result = BlurFrameResult(
                frame_index=frame_index,
                blur_mode=mode,
                blur_intensity=blur_intensity,
                total_blurred_regions=0,
                active_tracks_blurred=0,
                ghost_regions_blurred=0,
                blur_latency_ms=0.0,
                roi_plans=[],
            )
            return frame if frame is not None else np.zeros((1080, 1920, 3), dtype=np.uint8), empty_result

        frame_height, frame_width = frame.shape[:2]
        output_frame = frame.copy()

        # Step 1: Merge active tracks and ghosts into ROI Plans
        roi_plans: List[BlurROIPlan] = []
        active_tracks_count = 0
        ghost_regions_count = 0

        # Process active confirmed tracks
        for t in tracks:
            if t.is_confirmed and t.track_state != TrackStateEnum.REMOVED:
                plan = self._convert_normalized_bbox_to_roi(t.bbox, frame_width, frame_height, t.category, is_ghost=False)
                if plan:
                    roi_plans.append(plan)
                    active_tracks_count += 1

        # Process active ghost boxes
        for g in ghosts:
            if g.active:
                plan = self._convert_normalized_bbox_to_roi(g.predicted_bbox, frame_width, frame_height, g.category, is_ghost=True)
                if plan:
                    roi_plans.append(plan)
                    ghost_regions_count += 1

        # Edge Case 2: Zero active regions to blur
        if not roi_plans:
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            result = BlurFrameResult(
                frame_index=frame_index,
                blur_mode=mode,
                blur_intensity=blur_intensity,
                total_blurred_regions=0,
                active_tracks_blurred=0,
                ghost_regions_blurred=0,
                blur_latency_ms=elapsed_ms,
                roi_plans=[],
            )
            return output_frame, result

        # Step 2: In-place ROI Blur Application
        for plan in roi_plans:
            self._apply_roi_filter(output_frame, plan, mode, blur_intensity)

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        logger.debug(
            f"[BlurService] Frame #{frame_index} anonymized in {elapsed_ms}ms | "
            f"Mode: {mode.value} ({blur_intensity}%) | Blurred Regions: {len(roi_plans)} "
            f"({active_tracks_count} active tracks, {ghost_regions_count} ghosts)"
        )

        result = BlurFrameResult(
            frame_index=frame_index,
            blur_mode=mode,
            blur_intensity=blur_intensity,
            total_blurred_regions=len(roi_plans),
            active_tracks_blurred=active_tracks_count,
            ghost_regions_blurred=ghost_regions_count,
            blur_latency_ms=elapsed_ms,
            roi_plans=roi_plans,
        )

        return output_frame, result

    def _convert_normalized_bbox_to_roi(
        self,
        bbox: Any,
        width: int,
        height: int,
        category: Any,
        is_ghost: bool
    ) -> Optional[BlurROIPlan]:
        """
        Converts normalized percentage coordinates to clipped integer pixel boundaries.
        """
        x1 = int((bbox.x / 100.0) * width)
        y1 = int((bbox.y / 100.0) * height)
        w = int((bbox.width / 100.0) * width)
        h = int((bbox.height / 100.0) * height)

        # Clip pixel bounds safely to image bounds [0, width] and [0, height]
        cx1 = max(0, min(width - 1, x1))
        cy1 = max(0, min(height - 1, y1))
        cx2 = max(cx1 + 1, min(width, x1 + w))
        cy2 = max(cy1 + 1, min(height, y1 + h))

        if cx2 <= cx1 or cy2 <= cy1:
            return None

        return BlurROIPlan(x1=cx1, y1=cy1, x2=cx2, y2=cy2, category=category, is_ghost=is_ghost)

    def _apply_roi_filter(
        self,
        frame: np.ndarray,
        plan: BlurROIPlan,
        mode: BlurModeEnum,
        intensity: int
    ) -> None:
        """
        Applies chosen blur mode directly to ROI slice in-place.
        """
        roi = frame[plan.y1:plan.y2, plan.x1:plan.x2]
        if roi.size == 0:
            return

        h, w = roi.shape[:2]

        if mode == BlurModeEnum.GAUSSIAN:
            # Scale Gaussian kernel size with intensity (must be odd integer >= 3)
            ksize = int(self.gaussian_ksize * (intensity / 100.0))
            if ksize % 2 == 0:
                ksize += 1
            ksize = max(3, ksize)
            blurred_roi = cv2.GaussianBlur(roi, (ksize, ksize), 0)
            frame[plan.y1:plan.y2, plan.x1:plan.x2] = blurred_roi

        elif mode == BlurModeEnum.PIXELATE:
            # Downscale then upscale with INTER_NEAREST
            block_size = max(2, int(self.pixelate_block * (intensity / 50.0)))
            small_w = max(1, w // block_size)
            small_h = max(1, h // block_size)
            small_roi = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
            pixelated_roi = cv2.resize(small_roi, (w, h), interpolation=cv2.INTER_NEAREST)
            frame[plan.y1:plan.y2, plan.x1:plan.x2] = pixelated_roi

        elif mode == BlurModeEnum.BLACKOUT:
            # Solid blackout mask
            frame[plan.y1:plan.y2, plan.x1:plan.x2] = 0

        elif mode == BlurModeEnum.FEATHERED:
            # Gaussian blur with alpha feathering around borders
            ksize = int(self.gaussian_ksize * (intensity / 100.0))
            if ksize % 2 == 0:
                ksize += 1
            blurred_roi = cv2.GaussianBlur(roi, (max(3, ksize), max(3, ksize)), 0)

            # Create elliptical alpha mask for feathered edge blending
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(mask, (w // 2, h // 2), (w // 2, h // 2), 0, 0, 360, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (21, 21), 0)
            mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

            blended_roi = (blurred_roi * mask_3ch + roi * (1.0 - mask_3ch)).astype(np.uint8)
            frame[plan.y1:plan.y2, plan.x1:plan.x2] = blended_roi

blur_service = BlurService()
