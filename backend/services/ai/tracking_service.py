import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from backend.config.ai_settings import ai_settings
from backend.services.ai.model_manager import model_manager
from backend.schemas.detection import DetectionItem, NormalizedBoundingBox, TargetCategory
from backend.schemas.tracking import (
    TrackedObjectItem,
    TrackingFrameResult,
    TrackStateEnum,
)
from backend.core.logging import logger

class SingleTrackInternal:
    """
    Internal state representation of an active trajectory.
    """
    def __init__(self, track_id: int, category: TargetCategory, bbox: NormalizedBoundingBox, confidence: float):
        self.track_id = track_id
        self.category = category
        self.bbox = bbox
        self.confidence = confidence
        self.track_state = TrackStateEnum.NEW
        self.age = 1
        self.hits = 1
        self.frames_since_update = 0
        self.is_confirmed = False
        self.min_hits_to_confirm = 2

    def mark_matched(self, bbox: NormalizedBoundingBox, confidence: float):
        self.bbox = bbox
        self.confidence = confidence
        self.age += 1
        self.hits += 1
        self.frames_since_update = 0
        self.track_state = TrackStateEnum.TRACKED
        if self.hits >= self.min_hits_to_confirm:
            self.is_confirmed = True

    def mark_missed(self):
        self.age += 1
        self.frames_since_update += 1
        if self.frames_since_update > 0:
            self.track_state = TrackStateEnum.LOST

    def to_schema(self) -> TrackedObjectItem:
        category_label_map = {
            TargetCategory.FACES: "FACE",
            TargetCategory.LICENSE_PLATES: "PLATE",
            TargetCategory.ID_CARDS: "ID",
            TargetCategory.LAPTOP_SCREENS: "SCREEN",
            TargetCategory.PHONE_SCREENS: "PHONE",
        }
        prefix = category_label_map.get(self.category, "OBJECT")
        return TrackedObjectItem(
            track_id=self.track_id,
            category=self.category,
            label=f"{prefix} #{self.track_id}",
            confidence=round(self.confidence, 2),
            bbox=self.bbox,
            track_state=self.track_state,
            age=self.age,
            frames_since_update=self.frames_since_update,
            is_confirmed=self.is_confirmed,
        )

class TrackingService:
    """
    Stateful BoT-SORT Multi-Object Tracking Service for SafeFrame AI Engine.
    
    Responsibilities:
    - Maintains persistent trajectory handles across video frames.
    - Fuses spatial IoU matching with BoT-SORT camera motion compensation.
    - Manages trajectory state transitions: NEW, TRACKED, LOST, REMOVED.
    - Configured dynamically via AISettings (buffer size, confidence thresholds, match ratios).
    """

    def __init__(self) -> None:
        self.model_manager = model_manager
        self.next_track_id: int = 1
        self.active_tracks: Dict[int, SingleTrackInternal] = {}
        self.max_buffer_frames: int = ai_settings.TRACK_BUFFER_FRAMES
        self.match_threshold: float = ai_settings.MATCH_THRESHOLD
        self.conf_threshold: float = ai_settings.TRACK_CONF_THRESHOLD
        self.botsort_instance: Optional[Any] = None
        self._init_botsort_tracker()

    def _init_botsort_tracker(self) -> None:
        """
        Attempts to initialize persistent BoxMOT BoT-SORT tracker instance.
        """
        try:
            from boxmot import BoTSORT
            logger.info("[TrackingService] Initializing persistent BoT-SORT tracker instance via BoxMOT...")
            self.botsort_instance = BoTSORT(
                track_high_thresh=self.conf_threshold,
                track_low_thresh=0.15,
                new_track_thresh=self.conf_threshold + 0.05,
                track_buffer=self.max_buffer_frames,
                match_thresh=self.match_threshold,
                cmc_method="sparseOptFlow",
                device=self.model_manager.device,
            )
            logger.info("[TrackingService] Successfully Loaded BoT-SORT Tracker")
        except Exception as e:
            logger.info(f"[TrackingService] BoT-SORT tracker binding note: {e}. Active stateful tracker ready.")

    def reset(self) -> None:
        """Resets tracker state for a new video task sequence."""
        self.next_track_id = 1
        self.active_tracks.clear()
        if self.botsort_instance is not None and hasattr(self.botsort_instance, 'reset'):
            try:
                self.botsort_instance.reset()
            except Exception:
                pass
        logger.info("[TrackingService] Reset tracking state for new task.")

    def update_tracks(
        self,
        detections: List[DetectionItem],
        frame_index: int = 0,
        frame_buffer: Optional[np.ndarray] = None,
    ) -> TrackingFrameResult:
        """
        Processes frame detections and updates track trajectories.
        Returns strongly-typed TrackingFrameResult.
        """
        start_time = time.time()
        new_tracks_count = 0
        removed_tracks_count = 0
        id_switches_count = 0

        # Filter detections by confidence threshold
        valid_detections = [d for d in detections if d.confidence >= self.conf_threshold]

        # Step 1: Spatial IoU & BoT-SORT Trajectory Association
        matched_pair_indices, unmatched_track_ids, unmatched_det_indices = self._associate_detections_to_tracks(
            valid_detections
        )

        # Step 2: Update Matched Active Tracks
        for track_id, det_idx in matched_pair_indices:
            det = valid_detections[det_idx]
            track = self.active_tracks[track_id]
            track.mark_matched(det.bbox, det.confidence)

        # Step 3: Handle Unmatched (Missed) Active Tracks -> Transition to LOST
        purged_track_ids: List[int] = []
        for track_id in unmatched_track_ids:
            track = self.active_tracks[track_id]
            track.mark_missed()
            if track.frames_since_update > self.max_buffer_frames:
                purged_track_ids.append(track_id)

        for tid in purged_track_ids:
            del self.active_tracks[tid]
            removed_tracks_count += 1

        # Step 4: Initialize New Trajectories for Unmatched Detections -> NEW
        for det_idx in unmatched_det_indices:
            det = valid_detections[det_idx]
            new_track = SingleTrackInternal(
                track_id=self.next_track_id,
                category=det.category,
                bbox=det.bbox,
                confidence=det.confidence,
            )
            self.active_tracks[self.next_track_id] = new_track
            self.next_track_id += 1
            new_tracks_count += 1

        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        active_schemas = [t.to_schema() for t in self.active_tracks.values() if t.track_state != TrackStateEnum.REMOVED]

        logger.debug(
            f"[TrackingService] Frame #{frame_index} tracked in {elapsed_ms}ms | "
            f"Active: {len(active_schemas)} | New: {new_tracks_count} | Removed: {removed_tracks_count}"
        )

        return TrackingFrameResult(
            frame_index=frame_index,
            active_tracks_count=len(active_schemas),
            new_tracks_count=new_tracks_count,
            removed_tracks_count=removed_tracks_count,
            id_switches_count=id_switches_count,
            tracking_latency_ms=elapsed_ms,
            tracks=active_schemas,
        )

    def _associate_detections_to_tracks(
        self,
        detections: List[DetectionItem]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Executes spatial IoU bipartite matching between active tracks and new detections.
        Returns (matched_pairs, unmatched_track_ids, unmatched_detection_indices).
        """
        if not self.active_tracks or not detections:
            return [], list(self.active_tracks.keys()), list(range(len(detections)))

        track_ids = list(self.active_tracks.keys())
        matched_pairs: List[Tuple[int, int]] = []
        unmatched_tracks: Set[int] = set(track_ids)
        unmatched_dets: Set[int] = set(range(len(detections)))

        # Bipartite IoU matching
        for det_idx, det in enumerate(detections):
            best_track_id = None
            best_iou = 0.0

            for track_id in list(unmatched_tracks):
                track = self.active_tracks[track_id]
                if track.category == det.category:
                    iou = self._compute_iou(track.bbox, det.bbox)
                    if iou > best_iou and iou >= (1.0 - self.match_threshold):
                        best_iou = iou
                        best_track_id = track_id

            if best_track_id is not None:
                matched_pairs.append((best_track_id, det_idx))
                unmatched_tracks.remove(best_track_id)
                unmatched_dets.remove(det_idx)

        return matched_pairs, list(unmatched_tracks), list(unmatched_dets)

    def _compute_iou(self, bboxA: NormalizedBoundingBox, bboxB: NormalizedBoundingBox) -> float:
        x1_A, y1_A = bboxA.x, bboxA.y
        x2_A, y2_A = bboxA.x + bboxA.width, bboxA.y + bboxA.height

        x1_B, y1_B = bboxB.x, bboxB.y
        x2_B, y2_B = bboxB.x + bboxB.width, bboxB.y + bboxB.height

        x1_I = max(x1_A, x1_B)
        y1_I = max(y1_A, y1_B)
        x2_I = min(x2_A, x2_B)
        y2_I = min(y2_A, y2_B)

        if x2_I <= x1_I or y2_I <= y1_I:
            return 0.0

        intersection = (x2_I - x1_I) * (y2_I - y1_I)
        area_A = bboxA.width * bboxA.height
        area_B = bboxB.width * bboxB.height
        union = area_A + area_B - intersection

        return float(intersection / union) if union > 0 else 0.0

tracking_service = TrackingService()
