import cv2
import math
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from backend.config.ai_settings import ai_settings
from backend.schemas.detection import TargetCategory
from backend.schemas.tracking import TrackedObjectItem, TrackStateEnum
from backend.schemas.ghost_box import GhostBoxItem
from backend.schemas.blur import BlurModeEnum, BlurROIPlan, BlurFrameResult
from backend.core.logging import logger

class BlurService:
    """
    High-Performance ROI-Restricted Image Anonymization Service for SafeFrame AI Engine.

    Responsibilities:
    - Merges active tracked entities and predictive Ghost Boxes into a unified ROI execution plan.
    - Builds each mask from the detected target box with proportional padding.
      Faces get a rounded (superellipse) silhouette; license plates get a full
      rectangular mask matching their own geometry.
    - Applies Gaussian Blur, Pixelation, Blackout, or Feathered Blur strictly inside ROI slices in-place.
    - Safely clips bounding box coordinates to image boundaries [0, width] and [0, height].
    - Handles overlapping, nested, or out-of-bounds regions seamlessly.

    Privacy contract: whatever the mask looks like, every pixel of the detected
    target box itself is fully filtered. A face's rounded edge and soft ramp
    only live in the padding ring; a plate's rectangle covers the padded ROI
    with no soft rim, so no character can survive at the edges.
    """

    def __init__(self) -> None:
        self.default_style = ai_settings.DEFAULT_BLUR_STYLE
        self.gaussian_sigma_min_ratio = ai_settings.GAUSSIAN_SIGMA_MIN_RATIO
        self.gaussian_sigma_gain = ai_settings.GAUSSIAN_SIGMA_GAIN
        self.gaussian_min_sigma = ai_settings.GAUSSIAN_MIN_SIGMA_PX
        self.gaussian_max_kernel = ai_settings.GAUSSIAN_MAX_KERNEL
        self.pixelate_max_blocks = ai_settings.PIXELATE_MAX_BLOCKS
        self.pixelate_min_block = ai_settings.PIXELATE_MIN_BLOCK_PX
        self.plate_pixelate_max_blocks = ai_settings.PLATE_PIXELATE_MAX_BLOCKS
        self.padding_x_ratio = ai_settings.BLUR_PADDING_X_RATIO
        self.padding_y_ratio = ai_settings.BLUR_PADDING_Y_RATIO
        self.min_padding = ai_settings.BLUR_MIN_PADDING_PX
        self.max_padding = ai_settings.BLUR_MAX_PADDING_PX
        # Plates are wide and short; their padding is tuned separately so the
        # mask covers characters at the top/bottom edge without swallowing the
        # surrounding vehicle body.
        self.plate_padding_x_ratio = ai_settings.PLATE_PADDING_X_RATIO
        self.plate_padding_y_ratio = ai_settings.PLATE_PADDING_Y_RATIO
        self.plate_min_padding = ai_settings.PLATE_MIN_PADDING_PX
        self.plate_max_padding = ai_settings.PLATE_MAX_PADDING_PX
        self.plate_hold_frames = ai_settings.PLATE_REDACTION_HOLD_FRAMES
        # Documents use their own padding/hold/pixelation settings: they are
        # text-dense and the mask must cover the whole card/passport including
        # the machine-readable zone.
        self.document_padding_x_ratio = ai_settings.DOCUMENT_PADDING_X_RATIO
        self.document_padding_y_ratio = ai_settings.DOCUMENT_PADDING_Y_RATIO
        self.document_min_padding = ai_settings.DOCUMENT_MIN_PADDING_PX
        self.document_max_padding = ai_settings.DOCUMENT_MAX_PADDING_PX
        self.document_hold_frames = ai_settings.DOCUMENT_REDACTION_HOLD_FRAMES
        self.document_pixelate_max_blocks = ai_settings.DOCUMENT_PIXELATE_MAX_BLOCKS
        # Laptop-screen masks are polygons, not rectangles: the display is a
        # perspective quadrilateral. Padding grows the polygon outward only.
        self.laptop_padding_ratio = ai_settings.LAPTOP_PADDING_RATIO
        self.laptop_min_padding = ai_settings.LAPTOP_MIN_PADDING_PX
        self.laptop_max_padding = ai_settings.LAPTOP_MAX_PADDING_PX
        self.laptop_hold_frames = ai_settings.LAPTOP_REDACTION_HOLD_FRAMES
        self.laptop_pixelate_max_blocks = ai_settings.LAPTOP_PIXELATE_MAX_BLOCKS
        # Phone-screen masks are polygons: the display is a quadrilateral.
        # Padding grows the polygon outward only.
        self.phone_padding_ratio = ai_settings.PHONE_PADDING_RATIO
        self.phone_min_padding = ai_settings.PHONE_MIN_PADDING_PX
        self.phone_max_padding = ai_settings.PHONE_MAX_PADDING_PX
        self.phone_hold_frames = ai_settings.PHONE_REDACTION_HOLD_FRAMES
        self.phone_pixelate_max_blocks = ai_settings.PHONE_PIXELATE_MAX_BLOCKS
        self.superellipse_exponent = max(2.0, ai_settings.MASK_SUPERELLIPSE_EXPONENT)
        self.solid_margin_ratio = max(0.0, min(1.0, ai_settings.MASK_SOLID_MARGIN_RATIO))
        self.hold_frames = ai_settings.REDACTION_HOLD_FRAMES
        # Mask shape depends only on (ROI size, core inset, softness) and frame
        # geometry is constant per task, so the alpha mask is reused per shape.
        self._alpha_cache: Dict[Tuple[Any, ...], np.ndarray] = {}

    def render_blur(
        self,
        frame: Optional[np.ndarray],
        tracks: List[TrackedObjectItem],
        ghosts: List[GhostBoxItem],
        blur_style: str = "gaussian",
        blur_intensity: int = 80,
        frame_index: int = 0,
        detections: Optional[List[Any]] = None,
        hold_frames: Optional[int] = None,
    ) -> Tuple[np.ndarray, BlurFrameResult]:
        """
        Applies anonymization filters to target ROI regions inside the frame.
        Returns (anonymized_frame, BlurFrameResult).

        Redaction sources, in priority order:
        1. ``detections`` from the current frame - masked immediately, on the
           very first frame a face appears (fail-safe).
        2. Live tracks (NEW/TRACKED) - their smoothed box never excludes the
           detection it was matched to.
        3. Recently lost tracks, held for ``hold_frames`` frames at their last
           known box, without movement or expansion.
        4. Predictive ghost boxes, only if explicitly supplied.

        Overlapping sources describing the same face are merged into a single
        union region, so the same pixels are never filtered twice and a slightly
        shifted smoothed box can never leave part of the detection unmasked.
        """
        start_time = time.time()

        # Parse requested blur mode enum. An unrecognized style must not silently
        # downgrade to a weaker filter, so fail closed to fully-opaque blackout.
        try:
            mode = BlurModeEnum(str(blur_style).lower())
        except (ValueError, AttributeError):
            logger.warning(
                f"[BlurService] Frame #{frame_index}: unsupported blur style "
                f"{blur_style!r}; failing closed to blackout."
            )
            mode = BlurModeEnum.BLACKOUT

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

        # Step 1: Merge detections, tracks, held and ghost boxes into ROI plans.
        roi_plans: List[BlurROIPlan] = []
        active_tracks_count = 0
        ghost_regions_count = 0
        held_tracks_count = 0
        hold_limit = self.hold_frames if hold_frames is None else int(hold_frames)
        plate_hold_limit = self.plate_hold_frames if hold_frames is None else int(hold_frames)
        document_hold_limit = self.document_hold_frames if hold_frames is None else int(hold_frames)
        laptop_hold_limit = self.laptop_hold_frames if hold_frames is None else int(hold_frames)
        phone_hold_limit = self.phone_hold_frames if hold_frames is None else int(hold_frames)

        def add_plan(bbox: Any, category: Any, is_ghost: bool,
                     quad: Any = None, region_kind: str = "box") -> bool:
            """Adds a plan, or folds it into an equivalent existing region."""
            plan = self._convert_normalized_bbox_to_roi(
                bbox, frame_width, frame_height, category, is_ghost=is_ghost,
                quad=quad, region_kind=region_kind)
            if plan is None:
                return False
            for index, existing in enumerate(roi_plans):
                if self._same_region(plan, existing):
                    roi_plans[index] = self._merge_plans(existing, plan)
                    return False
            roi_plans.append(plan)
            return True

        # (1) Current-frame detections: masked immediately, including the first
        # frame a face appears. This is the primary redaction source.
        for detection in (detections or []):
            if add_plan(detection.bbox, detection.category, is_ghost=False,
                        quad=getattr(detection, "quad", None),
                        region_kind=getattr(detection, "region_kind", "box")):
                active_tracks_count += 1

        # (2) Live tracks. Their box is a union of the smoothed trajectory and
        # the current detection, so folding it in can only grow the mask.
        for track in tracks:
            if track.track_state in (TrackStateEnum.NEW, TrackStateEnum.TRACKED):
                if add_plan(track.bbox, track.category, is_ghost=False,
                            quad=getattr(track, "quad", None),
                            region_kind=getattr(track, "region_kind", "box")):
                    active_tracks_count += 1

        # (3) Conservative hold for a confirmed face that was just missed. The
        # box does not move or grow; the brief hold only bridges a one/two frame
        # detector gap.
        if hold_limit > 0:
            for track in tracks:
                if track.track_state != TrackStateEnum.LOST or not track.is_confirmed:
                    continue
                if track.category == TargetCategory.LICENSE_PLATES:
                    limit = plate_hold_limit
                elif track.category in (TargetCategory.ID_CARDS, TargetCategory.PASSPORTS):
                    limit = document_hold_limit
                elif track.category == TargetCategory.LAPTOP_SCREENS:
                    limit = laptop_hold_limit
                elif track.category == TargetCategory.PHONE_SCREENS:
                    limit = phone_hold_limit
                else:
                    limit = hold_limit
                if track.frames_since_update > limit:
                    continue
                if add_plan(track.bbox, track.category, is_ghost=False,
                            quad=getattr(track, "quad", None),
                            region_kind=getattr(track, "region_kind", "box")):
                    held_tracks_count += 1

        # (4) Predictive ghost boxes (disabled by default; see ai_settings).
        for ghost in ghosts:
            if ghost.active and add_plan(ghost.predicted_bbox, ghost.category, is_ghost=True):
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
            f"({active_tracks_count} detected/tracked, {held_tracks_count} held, "
            f"{ghost_regions_count} ghosts)"
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

    # ------------------------------------------------------------------ geometry

    @staticmethod
    def _core(plan: BlurROIPlan) -> Tuple[int, int, int, int]:
        """Un-padded face rectangle of a plan (falls back to the ROI itself)."""
        return (
            plan.core_x1 if plan.core_x1 is not None else plan.x1,
            plan.core_y1 if plan.core_y1 is not None else plan.y1,
            plan.core_x2 if plan.core_x2 is not None else plan.x2,
            plan.core_y2 if plan.core_y2 is not None else plan.y2,
        )

    @classmethod
    def _core_overlap_ratio(cls, a: BlurROIPlan, b: BlurROIPlan) -> float:
        """Intersection over the smaller core rectangle (0.0 - 1.0)."""
        ax1, ay1, ax2, ay2 = cls._core(a)
        bx1, by1, bx2, by2 = cls._core(b)
        ix = max(0, min(ax2, bx2) - max(ax1, bx1))
        iy = max(0, min(ay2, by2) - max(ay1, by1))
        intersection = ix * iy
        if intersection <= 0:
            return 0.0
        smaller = min(max(1, (ax2 - ax1) * (ay2 - ay1)), max(1, (bx2 - bx1) * (by2 - by1)))
        return intersection / smaller

    def _same_region(self, plan: BlurROIPlan, existing: BlurROIPlan) -> bool:
        """True when two sources refer to effectively the same target region.

        A region is merged when at least 80% of the smaller of the two is
        already covered. This prevents a detection and its own track (or a stale
        held box) from filtering the same pixels twice. Different categories are
        never merged: a face and a plate need different mask geometry.
        """
        if plan.category != existing.category:
            return False
        return self._core_overlap_ratio(plan, existing) >= 0.80

    @staticmethod
    def _merge_plans(existing: BlurROIPlan, plan: BlurROIPlan) -> BlurROIPlan:
        """Union of two plans describing the same target.

        Taking the union guarantees that a smoothed/held box can never cut a
        corner off the current detection. Polygon plans (laptop screens) are
        merged as the convex hull of both regions, which still contains every
        pixel either source had to cover.
        """
        ex1, ey1, ex2, ey2 = BlurService._core(existing)
        px1, py1, px2, py2 = BlurService._core(plan)
        merged_quad = None
        region_kind = existing.region_kind
        if existing.quad or plan.quad:
            polygons = []
            for item in (existing, plan):
                if item.quad:
                    polygons.append(np.asarray(item.quad, dtype=np.float32).reshape(-1, 2))
                else:
                    ix1, iy1, ix2, iy2 = BlurService._core(item)
                    polygons.append(np.array([[ix1, iy1], [ix2, iy1],
                                              [ix2, iy2], [ix1, iy2]], dtype=np.float32))
            hull = cv2.convexHull(np.concatenate(polygons, axis=0)).reshape(-1, 2)
            merged_quad = [[float(x), float(y)] for x, y in hull]
            if existing.region_kind != plan.region_kind:
                # One of the two sources could not separate the display: the
                # merged region is a device-box fallback, and says so. The
                # fallback kind is category-specific so a phone region is never
                # reported as a laptop fallback.
                region_kind = ("phone_fallback"
                               if existing.category == TargetCategory.PHONE_SCREENS
                               else "laptop_fallback")
        return BlurROIPlan(
            x1=min(existing.x1, plan.x1), y1=min(existing.y1, plan.y1),
            x2=max(existing.x2, plan.x2), y2=max(existing.y2, plan.y2),
            category=existing.category,
            is_ghost=existing.is_ghost and plan.is_ghost,
            core_x1=min(ex1, px1), core_y1=min(ey1, py1),
            core_x2=max(ex2, px2), core_y2=max(ey2, py2),
            quad=merged_quad, region_kind=region_kind,
            pad=max(existing.pad, plan.pad),
        )

    def _padding(self, core_w: int, core_h: int, is_plate: bool = False,
                 is_document: bool = False, is_laptop: bool = False,
                 is_phone: bool = False) -> Tuple[int, int]:
        """Padding proportional to the target's own width and height."""
        if is_phone:
            ratio_x = ratio_y = self.phone_padding_ratio
            min_pad, max_pad = self.phone_min_padding, self.phone_max_padding
        elif is_laptop:
            ratio_x = ratio_y = self.laptop_padding_ratio
            min_pad, max_pad = self.laptop_min_padding, self.laptop_max_padding
        elif is_document:
            ratio_x, ratio_y = self.document_padding_x_ratio, self.document_padding_y_ratio
            min_pad, max_pad = self.document_min_padding, self.document_max_padding
        elif is_plate:
            ratio_x, ratio_y = self.plate_padding_x_ratio, self.plate_padding_y_ratio
            min_pad, max_pad = self.plate_min_padding, self.plate_max_padding
        else:
            ratio_x, ratio_y = self.padding_x_ratio, self.padding_y_ratio
            min_pad, max_pad = self.min_padding, self.max_padding
        pad_x = int(round(core_w * ratio_x))
        pad_y = int(round(core_h * ratio_y))
        pad_x = max(min_pad, min(max_pad, pad_x))
        pad_y = max(min_pad, min(max_pad, pad_y))
        return pad_x, pad_y

    def _convert_normalized_bbox_to_roi(
        self,
        bbox: Any,
        width: int,
        height: int,
        category: Any,
        is_ghost: bool,
        quad: Any = None,
        region_kind: str = "box",
    ) -> Optional[BlurROIPlan]:
        """
        Converts normalized percentage coordinates to clipped integer pixel boundaries.

        The exact rectangle is rounded *outward* (floor the origin, ceil the far
        edge) so the ROI always contains the whole requested region, then expanded
        by padding derived from the target's own width and height and clipped to
        the frame. When ``quad`` is supplied the polygon is clipped to the frame
        and the ROI is its bounding box plus padding; the mask itself stays a
        polygon, so a perspective screen is not turned into a rectangle.
        """
        polygon: Optional[np.ndarray] = None
        if quad:
            try:
                points = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
            except (TypeError, ValueError):
                points = None
            if points is not None and points.shape[0] >= 3:
                points[:, 0] = np.clip(points[:, 0], 0.0, float(width - 1))
                points[:, 1] = np.clip(points[:, 1], 0.0, float(height - 1))
                if cv2.contourArea(points.astype(np.int32)) >= 4.0:
                    polygon = points

        if polygon is not None:
            fx1 = float(polygon[:, 0].min())
            fy1 = float(polygon[:, 1].min())
            fx2 = float(polygon[:, 0].max())
            fy2 = float(polygon[:, 1].max())
        else:
            # Exact floating-point pixel rectangle.
            fx1 = (bbox.x / 100.0) * width
            fy1 = (bbox.y / 100.0) * height
            fx2 = ((bbox.x + bbox.width) / 100.0) * width
            fy2 = ((bbox.y + bbox.height) / 100.0) * height

        # Round outward: never shrink below the requested region.
        core_x1 = math.floor(fx1)
        core_y1 = math.floor(fy1)
        core_x2 = math.ceil(fx2)
        core_y2 = math.ceil(fy2)

        core_w = max(1, core_x2 - core_x1)
        core_h = max(1, core_y2 - core_y1)
        is_plate = category == TargetCategory.LICENSE_PLATES
        is_document = category in (TargetCategory.ID_CARDS, TargetCategory.PASSPORTS)
        is_laptop = category == TargetCategory.LAPTOP_SCREENS
        is_phone = category == TargetCategory.PHONE_SCREENS
        pad_x, pad_y = self._padding(core_w, core_h, is_plate=is_plate,
                                     is_document=is_document, is_laptop=is_laptop,
                                     is_phone=is_phone)
        if polygon is not None:
            # A polygon mask grows uniformly; separate x/y padding would make
            # the dilated outline anisotropic.
            pad_x = pad_y = max(pad_x, pad_y)

        # Clip padded bounds safely to image bounds [0, width] and [0, height].
        cx1 = max(0, min(width - 1, core_x1 - pad_x))
        cy1 = max(0, min(height - 1, core_y1 - pad_y))
        cx2 = max(cx1 + 1, min(width, core_x2 + pad_x))
        cy2 = max(cy1 + 1, min(height, core_y2 + pad_y))

        if cx2 <= cx1 or cy2 <= cy1:
            return None

        # Clamp the core into the final ROI so the mask builder can rely on it.
        ccx1 = max(cx1, min(cx2, core_x1))
        ccy1 = max(cy1, min(cy2, core_y1))
        ccx2 = max(ccx1, min(cx2, core_x2))
        ccy2 = max(ccy1, min(cy2, core_y2))

        return BlurROIPlan(
            x1=cx1, y1=cy1, x2=cx2, y2=cy2, category=category, is_ghost=is_ghost,
            core_x1=ccx1, core_y1=ccy1, core_x2=ccx2, core_y2=ccy2,
            quad=[[float(x), float(y)] for x, y in polygon] if polygon is not None else None,
            region_kind=region_kind, pad=int(max(pad_x, pad_y)),
        )

    # -------------------------------------------------------------------- masks

    def _alpha_mask(self, plan: BlurROIPlan, soft: bool, power: float) -> np.ndarray:
        """Alpha mask for one ROI.

        Faces: a superellipse (rounded rectangle) inscribed in the padded ROI.
        At the configured padding ratios this silhouette contains the whole
        detected face rectangle, so the rounded outline does not shave off any
        part of the face. Blur/pixelation additionally fade from the face box
        edge outwards (`soft=True`); blackout uses the same outline with a hard,
        anti-aliased edge for a clean look.

        License plates: a full rectangle matching the plate's own geometry. A
        plate is a flat, high-contrast rectangle, and any soft or rounded rim
        could leave a character fragment readable, so the entire padded ROI is
        filtered at alpha 1.0.

        Returns a float32 array of shape (h, w) with values in [0, 1], and is
        cached because the shape only depends on the integers below.
        """
        w = plan.x2 - plan.x1
        h = plan.y2 - plan.y1

        if plan.quad:
            # Screen region: hard, full-opacity polygon grown outward by the
            # configured padding. No soft rim, so no original pixel inside the
            # display can survive at the edge.
            points = np.round(np.asarray(plan.quad, dtype=np.float32)
                              - np.array([plan.x1, plan.y1], dtype=np.float32)).astype(np.int32)
            pad = int(max(0, plan.pad))
            key = ("poly", w, h, pad, tuple(points.reshape(-1).tolist()))
            cached = self._alpha_cache.get(key)
            if cached is not None:
                return cached
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [points], 1)
            if pad > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
                mask = cv2.dilate(mask, kernel)
            alpha = mask.astype(np.float32)
            if len(self._alpha_cache) >= 64:
                self._alpha_cache.clear()
            self._alpha_cache[key] = alpha
            return alpha

        if plan.category in (TargetCategory.LICENSE_PLATES, TargetCategory.ID_CARDS,
                             TargetCategory.PASSPORTS):
            key = ("rect", w, h)
            cached = self._alpha_cache.get(key)
            if cached is None:
                cached = np.ones((h, w), dtype=np.float32)
                if len(self._alpha_cache) >= 64:
                    self._alpha_cache.clear()
                self._alpha_cache[key] = cached
            return cached
        cx1, cy1, cx2, cy2 = self._core(plan)
        ix1 = max(0, min(w, cx1 - plan.x1))
        iy1 = max(0, min(h, cy1 - plan.y1))
        ix2 = max(ix1, min(w, cx2 - plan.x1))
        iy2 = max(iy1, min(h, cy2 - plan.y1))

        key = (w, h, ix1, iy1, ix2, iy2, 1 if soft else 0, round(power, 2))
        cached = self._alpha_cache.get(key)
        if cached is not None:
            return cached

        alpha = np.zeros((h, w), dtype=np.float32)
        # Hard face core: always fully filtered, no matter what the rim does.
        core = np.zeros((h, w), dtype=np.float32)
        if ix2 > ix1 and iy2 > iy1:
            core[iy1:iy2, ix1:ix2] = 1.0

        half_w = max(0.5, w / 2.0)
        half_h = max(0.5, h / 2.0)
        center_x = (w - 1) / 2.0
        center_y = (h - 1) / 2.0
        ys = np.arange(h, dtype=np.float32)[:, None]
        xs = np.arange(w, dtype=np.float32)[None, :]
        nx = np.abs((xs - center_x) / half_w)
        ny = np.abs((ys - center_y) / half_h)
        quad = nx ** self.superellipse_exponent + ny ** self.superellipse_exponent

        if soft:
            # Normalize the ramp so it is 1.0 exactly at the face box's corners
            # and 0.0 on the superellipse outline: the visible falloff lives in
            # the padding ring, never over face pixels.
            rx = max(abs(ix1 - center_x), abs((ix2 - 1) - center_x)) / half_w
            ry = max(abs(iy1 - center_y), abs((iy2 - 1) - center_y)) / half_h
            core_q = rx ** self.superellipse_exponent + ry ** self.superellipse_exponent
            core_q = float(np.clip(core_q, 0.05, 0.95))
            # Keep the inner part of the padding ring fully filtered (detector
            # boxes are sometimes a pixel tight); ramp only over the outer part.
            solid_q = core_q + self.solid_margin_ratio * (1.0 - core_q)
            solid_q = float(np.clip(solid_q, 0.05, 0.995))
            alpha = np.clip((1.0 - quad) / (1.0 - solid_q), 0.0, 1.0)
            if power != 1.0:
                alpha = np.power(alpha, power, dtype=np.float32)
        else:
            # Hard edge with ~1px anti-aliasing (no visible stair-stepping).
            alpha = np.clip((1.0 - quad) / 0.08, 0.0, 1.0)

        alpha = np.maximum(alpha, core)
        alpha = alpha.astype(np.float32, copy=False)

        if len(self._alpha_cache) >= 64:
            self._alpha_cache.clear()
        self._alpha_cache[key] = alpha
        return alpha

    def mask_outline(self, plan: BlurROIPlan) -> Optional[np.ndarray]:
        """Pixel contour of the rendered mask, for debug annotation only.

        Lets the debug frames show the exact silhouette that was filtered
        instead of a misleading rectangle. Returns an (N, 2) int32 array of
        frame coordinates, or None when the mask is empty.
        """
        alpha = self._alpha_mask(plan, soft=False, power=1.0)
        binary = (alpha >= 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
        offset = np.array([plan.x1, plan.y1], dtype=np.int32)
        return contour.astype(np.int32) + offset

    # ------------------------------------------------------------------ filters

    def _gaussian_sigma(self, width: int, height: int, intensity: int) -> float:
        """
        Region-relative Gaussian sigma with a privacy floor.

        Sigma scales with the target's smaller side so protection tracks the
        target and frame resolution. Even the lowest intensity stays above the
        measured "still detectable by YuNet" threshold (~0.18 x min side on this
        checkout); the slider raises strength from there.
        """
        scale = max(0.1, min(1.0, intensity / 100.0))
        min_side = max(1, min(width, height))
        sigma = min_side * (self.gaussian_sigma_min_ratio + self.gaussian_sigma_gain * scale)
        return max(self.gaussian_min_sigma, sigma)

    def _gaussian_blur(self, roi: np.ndarray, sigma: float) -> np.ndarray:
        """Fast, strong Gaussian blur for a single ROI.

        For large sigmas the ROI is blurred at a reduced resolution and scaled
        back up. That carries the same low-pass effect at a fraction of the cost
        of a huge convolution kernel (and adds no detail back).
        """
        h, w = roi.shape[:2]
        downscale = int(min(4, max(1, sigma // 6)))
        if downscale <= 1:
            ksize = self._gaussian_kernel(sigma)
            return cv2.GaussianBlur(roi, (ksize, ksize), sigmaX=sigma)

        small_w = max(1, w // downscale)
        small_h = max(1, h // downscale)
        small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_AREA)
        local_sigma = max(1.0, sigma / downscale)
        ksize = self._gaussian_kernel(local_sigma)
        small = cv2.GaussianBlur(small, (ksize, ksize), sigmaX=local_sigma)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    def _gaussian_kernel(self, sigma: float) -> int:
        """Odd kernel covering ~4 sigma, clamped to GAUSSIAN_MAX_KERNEL."""
        ksize = int(math.ceil(sigma * 4.0))
        if ksize % 2 == 0:
            ksize += 1
        ksize = max(3, min(self.gaussian_max_kernel, ksize))
        if ksize % 2 == 0:
            ksize += 1
        return ksize

    def _pixelate_block(self, width: int, height: int, intensity: int,
                        max_blocks: Optional[int] = None) -> int:
        """
        Mosaic block size that bounds the grid against the target's smaller side
        and keeps a minimum block, so recognizable detail cannot survive.
        """
        scale = max(0.1, min(1.0, intensity / 100.0))
        min_side = max(1, min(width, height))
        cap = self.pixelate_max_blocks if max_blocks is None else max_blocks
        blocks = max(2, int(round(cap * (1.0 - 0.5 * scale))))
        return max(self.pixelate_min_block, int(math.ceil(min_side / blocks)))

    def _filter_roi(self, roi: np.ndarray, mode: BlurModeEnum, intensity: int,
                    max_blocks: Optional[int] = None) -> np.ndarray:
        """Returns the filtered version of a whole ROI rectangle."""
        h, w = roi.shape[:2]

        if mode == BlurModeEnum.GAUSSIAN:
            return self._gaussian_blur(roi, self._gaussian_sigma(w, h, intensity))

        if mode == BlurModeEnum.PIXELATE:
            # Downscale then upscale with INTER_NEAREST; the block size is bounded
            # against the target's smaller side and has a minimum.
            block_size = self._pixelate_block(w, h, intensity, max_blocks=max_blocks)
            small_w = max(1, w // block_size)
            small_h = max(1, h // block_size)
            small_roi = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
            return cv2.resize(small_roi, (w, h), interpolation=cv2.INTER_NEAREST)

        if mode == BlurModeEnum.BLACKOUT:
            return np.zeros_like(roi)

        return self._gaussian_blur(roi, self._gaussian_sigma(w, h, intensity))

    def _apply_roi_filter(
        self,
        frame: np.ndarray,
        plan: BlurROIPlan,
        mode: BlurModeEnum,
        intensity: int
    ) -> None:
        """
        Applies the chosen filter to an ROI slice, composited through the mask.

        The target box itself always receives alpha 1.0, so it is fully filtered
        by every mode; for faces the rounded, feathered silhouette only shapes
        the padding ring around it. Plates use a solid rectangular mask.
        """
        roi = frame[plan.y1:plan.y2, plan.x1:plan.x2]
        if roi.size == 0:
            return

        h, w = roi.shape[:2]
        slice_y = slice(plan.y1, plan.y2)
        slice_x = slice(plan.x1, plan.x2)

        if plan.category == TargetCategory.LAPTOP_SCREENS:
            max_blocks = self.laptop_pixelate_max_blocks
        elif plan.category == TargetCategory.PHONE_SCREENS:
            max_blocks = self.phone_pixelate_max_blocks
        elif plan.category == TargetCategory.LICENSE_PLATES:
            max_blocks = self.plate_pixelate_max_blocks
        elif plan.category in (TargetCategory.ID_CARDS, TargetCategory.PASSPORTS):
            max_blocks = self.document_pixelate_max_blocks
        else:
            max_blocks = None
        filtered = self._filter_roi(roi, mode, intensity, max_blocks=max_blocks)

        if plan.quad:
            # Polygon masks are hard in every mode: a feathered screen edge
            # would leave readable content just inside the display border.
            alpha = self._alpha_mask(plan, soft=False, power=1.0)
        elif mode == BlurModeEnum.BLACKOUT or mode == BlurModeEnum.PIXELATE:
            # Hard, anti-aliased rounded silhouette: a blended mosaic rim both
            # looks muddy and lets recognizable original pixels ghost through.
            alpha = self._alpha_mask(plan, soft=False, power=1.0)
        else:
            alpha = self._alpha_mask(plan, soft=True, power=1.0)

        if bool(np.all(alpha >= 1.0)):
            frame[slice_y, slice_x] = filtered
            return

        channel_count = roi.shape[2] if roi.ndim == 3 else 1
        mask = alpha if channel_count == 1 else np.repeat(alpha[:, :, np.newaxis], channel_count, axis=2)
        blended = (filtered.astype(np.float32) * mask
                   + roi.astype(np.float32) * (1.0 - mask))
        frame[slice_y, slice_x] = blended.astype(np.uint8)

blur_service = BlurService()
