import math
import time
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from backend.config.ai_settings import ai_settings
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
        # Centre velocity in normalized percent per frame, used ONLY to rank
        # associations. It never moves a mask: redaction uses last observed
        # boxes (see predicted_bbox and BlurService).
        self.velocity = (0.0, 0.0)
        # Redaction-box smoothing. ``bbox`` stays the rendered box: the
        # smoothed trajectory UNION the measured detection, so it always
        # covers the current detection and only damps detector jitter.
        self.smoothing_alpha = ai_settings.TRACK_SMOOTHING_ALPHA
        self.smoothing_snap_ratio = ai_settings.TRACK_SMOOTHING_SNAP_RATIO

    @property
    def center(self) -> Tuple[float, float]:
        return (self.bbox.x + self.bbox.width / 2.0, self.bbox.y + self.bbox.height / 2.0)

    def predicted_bbox(self) -> NormalizedBoundingBox:
        """Last box shifted by the smoothed velocity (association only)."""
        dx, dy = self.velocity
        return NormalizedBoundingBox(x=self.bbox.x + dx, y=self.bbox.y + dy,
                                     width=self.bbox.width, height=self.bbox.height)

    def _smooth_bbox(self, measured: NormalizedBoundingBox) -> NormalizedBoundingBox:
        """Damps detector jitter without ever shrinking below the detection.

        A still face jitters by a few percent of its own size frame to frame,
        which makes a rectangular mask visibly crawl. Blending the previous box
        with the measurement removes most of that. Real motion (relative to the
        face's own size) snaps straight to the measurement so the mask never
        trails a moving face. The result is always expanded to contain the
        measurement: smoothing may add coverage, never remove it.
        """
        previous = self.bbox
        motion = max(
            abs(measured.x - previous.x), abs(measured.y - previous.y),
            abs(measured.width - previous.width), abs(measured.height - previous.height),
        )
        scale = max(1e-6, min(previous.width, previous.height))
        if motion > self.smoothing_snap_ratio * scale:
            smoothed = measured
        else:
            alpha = max(0.05, min(1.0, self.smoothing_alpha))
            smoothed = NormalizedBoundingBox(
                x=alpha * measured.x + (1.0 - alpha) * previous.x,
                y=alpha * measured.y + (1.0 - alpha) * previous.y,
                width=alpha * measured.width + (1.0 - alpha) * previous.width,
                height=alpha * measured.height + (1.0 - alpha) * previous.height,
            )
        return self._cover(smoothed, measured)

    @staticmethod
    def _cover(box: NormalizedBoundingBox, required: NormalizedBoundingBox) -> NormalizedBoundingBox:
        """Smallest box containing both, clamped to the normalized 0-100 space.

        Rounding is outward (origin down, far edge up) so the rounded values
        still contain the required detection.
        """
        x1 = max(0.0, min(box.x, required.x))
        y1 = max(0.0, min(box.y, required.y))
        x2 = min(100.0, max(box.x + box.width, required.x + required.width))
        y2 = min(100.0, max(box.y + box.height, required.y + required.height))
        x1 = math.floor(x1 * 100.0) / 100.0
        y1 = math.floor(y1 * 100.0) / 100.0
        x2 = min(100.0, math.ceil(x2 * 100.0) / 100.0)
        y2 = min(100.0, math.ceil(y2 * 100.0) / 100.0)
        return NormalizedBoundingBox(
            x=x1, y=y1,
            width=round(max(0.1, x2 - x1), 2), height=round(max(0.1, y2 - y1), 2),
        )

    def mark_matched(self, bbox: NormalizedBoundingBox, confidence: float):
        previous_center = self.center
        new_center = (bbox.x + bbox.width / 2.0, bbox.y + bbox.height / 2.0)
        measured = (new_center[0] - previous_center[0], new_center[1] - previous_center[1])
        # Smooth, and clamp to a plausible per-frame face motion. A single noisy
        # box cannot send the prediction far from the face.
        limit = 6.0
        self.velocity = (
            max(-limit, min(limit, 0.6 * self.velocity[0] + 0.4 * measured[0])),
            max(-limit, min(limit, 0.6 * self.velocity[1] + 0.4 * measured[1])),
        )
        self.bbox = self._smooth_bbox(bbox)
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
    Stateful lightweight IoU multi-face tracker for SafeFrame AI.

    Responsibilities:
    - Maintain persistent trajectory IDs across video frames.
    - Associate detections with existing trajectories by IoU (globally greedy).
    - Manage trajectory state transitions: NEW, TRACKED, LOST, REMOVED.

    This is deliberately the only tracker in the system. No BoT-SORT / OSNet
    binding is initialized or claimed: those packages are not installed and
    their association behavior has not been validated here.

    Redaction does not depend on this tracker for coverage: every detection in
    the current frame is masked by BlurService directly. The tracker only
    supplies identity continuity and the short conservative hold for tracks that
    are temporarily missed.
    """

    def __init__(self) -> None:
        self.next_track_id: int = 1
        self.active_tracks: Dict[int, SingleTrackInternal] = {}
        self.max_buffer_frames: int = ai_settings.TRACK_BUFFER_FRAMES
        self.match_iou_threshold: float = ai_settings.MATCH_IOU_THRESHOLD
        self.conf_threshold: float = ai_settings.TRACK_CONF_THRESHOLD
        self.total_new_tracks: int = 0

    def reset(self) -> None:
        """Resets tracker state for a new video task sequence."""
        self.next_track_id = 1
        self.active_tracks.clear()
        self.total_new_tracks = 0
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
            self.total_new_tracks += 1

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
        Globally greedy IoU association between active tracks and detections.

        All candidate (track, detection) pairs are ranked by IoU before any
        assignment. Sequential per-detection matching can steal a track's best
        match during a crossing; ranking first makes that less likely and keeps
        IDs stable while two faces overlap.

        Returns (matched_pairs, unmatched_track_ids, unmatched_detection_indices).
        """
        if not self.active_tracks or not detections:
            return [], list(self.active_tracks.keys()), list(range(len(detections)))

        candidates: List[Tuple[float, float, int, int]] = []
        for track_id, track in self.active_tracks.items():
            predicted = track.predicted_bbox()
            predicted_center = (predicted.x + predicted.width / 2.0,
                                predicted.y + predicted.height / 2.0)
            for det_idx, det in enumerate(detections):
                if track.category != det.category:
                    continue
                # Score against both the last observed box and the constant
                # velocity prediction. During a crossing the prediction keeps
                # each face's own box as the best match instead of handing the
                # id to whichever neighbour happens to overlap more.
                iou = max(self._compute_iou(track.bbox, det.bbox),
                          self._compute_iou(predicted, det.bbox))
                if iou < self.match_iou_threshold:
                    continue
                det_center = (det.bbox.x + det.bbox.width / 2.0,
                              det.bbox.y + det.bbox.height / 2.0)
                distance = abs(det_center[0] - predicted_center[0]) + abs(det_center[1] - predicted_center[1])
                candidates.append((iou, -distance, track_id, det_idx))

        # Best overlap first; equal overlap is broken by proximity to the
        # predicted centre so the tie does not depend on dict/insertion order.
        candidates.sort(reverse=True)

        matched_pairs: List[Tuple[int, int]] = []
        unmatched_tracks: Set[int] = set(self.active_tracks.keys())
        unmatched_dets: Set[int] = set(range(len(detections)))
        for _, _, track_id, det_idx in candidates:
            if track_id in unmatched_tracks and det_idx in unmatched_dets:
                matched_pairs.append((track_id, det_idx))
                unmatched_tracks.discard(track_id)
                unmatched_dets.discard(det_idx)

        return matched_pairs, sorted(unmatched_tracks), sorted(unmatched_dets)

    @staticmethod
    def _compute_iou(bboxA: NormalizedBoundingBox, bboxB: NormalizedBoundingBox) -> float:
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
