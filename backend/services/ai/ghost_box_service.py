import time
import numpy as np
from typing import List, Dict, Any, Optional, Set
from backend.config.ai_settings import ai_settings
from backend.schemas.detection import NormalizedBoundingBox, TargetCategory
from backend.schemas.tracking import TrackedObjectItem, TrackStateEnum
from backend.schemas.ghost_box import GhostBoxItem, GhostBoxFrameResult
from backend.core.logging import logger

class SingleKalmanGhostPredictor:
    """
    2D Constant Velocity Kalman Filter state estimator for a lost trajectory.
    State Vector x = [u, v, s, r, du, dv, ds]^T
    """
    def __init__(self, track_id: int, category: TargetCategory, bbox: NormalizedBoundingBox, initial_conf: float):
        self.track_id = track_id
        self.category = category
        self.initial_bbox = bbox
        self.confidence = initial_conf
        self.ghost_age = 1
        self.max_ghost_frames = ai_settings.GHOST_MAX_FRAMES
        self.base_expansion_factor = ai_settings.GHOST_EXPANSION_FACTOR

        # State initialization [u, v, s, r, du, dv, ds]
        cx = bbox.x + bbox.width / 2.0
        cy = bbox.y + bbox.height / 2.0
        area = bbox.width * bbox.height
        aspect_ratio = bbox.width / max(1e-5, bbox.height)

        self.x = np.array([cx, cy, area, aspect_ratio, 0.0, 0.0, 0.0], dtype=np.float32)
        # Covariance matrix P initialized with higher uncertainty for velocity
        self.P = np.eye(7, dtype=np.float32) * 10.0
        self.P[4:, 4:] *= 100.0

        # State Transition Matrix F (dt = 1)
        self.F = np.eye(7, dtype=np.float32)
        self.F[0, 4] = 1.0  # u += du
        self.F[1, 5] = 1.0  # v += dv
        self.F[2, 6] = 1.0  # s += ds

        # Process Noise Covariance Q
        self.Q = np.eye(7, dtype=np.float32) * 0.01
        self.Q[4:, 4:] *= 0.1

    def predict(self) -> Tuple[NormalizedBoundingBox, float, float]:
        """
        Advances state estimate by 1 frame timestep, updates covariance P,
        and computes covariance-driven spatial expansion.
        """
        # x_k = F * x_{k-1}
        self.x = np.dot(self.F, self.x)
        # P_k = F * P_{k-1} * F^T + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        self.ghost_age += 1

        # Covariance Norm as measure of prediction uncertainty
        covariance_norm = float(np.linalg.norm(self.P))

        # Dynamic expansion factor: combines exponential loss frames with covariance growth
        cov_expansion = 1.0 + (0.01 * np.sqrt(covariance_norm))
        exp_factor = max(pow(self.base_expansion_factor, self.ghost_age), cov_expansion)

        # Unpack predicted bounding box
        cx, cy, s, r = self.x[0], self.x[1], max(1.0, self.x[2]), max(0.1, self.x[3])
        w = np.sqrt(s * r) * exp_factor
        h = np.sqrt(s / r) * exp_factor

        px = max(0.0, min(100.0, cx - w / 2.0))
        py = max(0.0, min(100.0, cy - h / 2.0))
        pw = max(1.0, min(100.0 - px, w))
        ph = max(1.0, min(100.0 - py, h))

        predicted_bbox = NormalizedBoundingBox(x=round(px, 2), y=round(py, 2), width=round(pw, 2), height=round(ph, 2))

        # Decay confidence linearly over ghost age
        decayed_conf = max(0.10, self.confidence - (self.ghost_age * 0.04))

        return predicted_bbox, round(decayed_conf, 2), round(exp_factor, 3)

class GhostBoxService:
    """
    SafeFrame Core Innovation Service: Predictive Ghost Box Preservation Engine.
    
    Responsibilities:
    - Automatically creates dynamic Ghost Boxes when sensitive object tracks transition to LOST state.
    - Uses 2D Constant Velocity Kalman Filter to estimate spatial motion vectors.
    - Computes spatial expansion driven by Kalman error covariance growth P(t) + base expansion factor.
    - Automatically recovers active ghosts when tracks re-appear.
    - Automatically purges expired ghosts exceeding GHOST_MAX_FRAMES.
    """

    def __init__(self) -> None:
        self.enabled: bool = ai_settings.GHOST_BOX_ENABLED
        self.max_ghost_frames: int = ai_settings.GHOST_MAX_FRAMES
        self.active_ghosts: Dict[int, SingleKalmanGhostPredictor] = {}

    def process_ghost_boxes(
        self,
        tracks: List[TrackedObjectItem],
        frame_index: int = 0
    ) -> GhostBoxFrameResult:
        """
        Evaluates input tracks and updates active Ghost Box trajectory predictions.
        Returns strongly-typed GhostBoxFrameResult.
        """
        start_time = time.time()
        new_ghosts_count = 0
        recovered_ghosts_count = 0
        expired_ghosts_count = 0

        if not self.enabled:
            return GhostBoxFrameResult(
                frame_index=frame_index,
                active_ghosts_count=0,
                new_ghosts_count=0,
                recovered_ghosts_count=0,
                expired_ghosts_count=0,
                prediction_latency_ms=0.0,
                ghost_boxes=[],
            )

        active_tracked_ids = {t.track_id for t in tracks if t.track_state in (TrackStateEnum.TRACKED, TrackStateEnum.NEW)}
        lost_tracks = [t for t in tracks if t.track_state == TrackStateEnum.LOST]

        # Step 1: Recover ghosts if track returned to TRACKED/NEW
        recovered_ids = [tid for tid in self.active_ghosts.keys() if tid in active_tracked_ids]
        for tid in recovered_ids:
            del self.active_ghosts[tid]
            recovered_ghosts_count += 1

        # Step 2: Register new ghosts for LOST tracks
        for lost_t in lost_tracks:
            if lost_t.track_id not in self.active_ghosts:
                predictor = SingleKalmanGhostPredictor(
                    track_id=lost_t.track_id,
                    category=lost_t.category,
                    bbox=lost_t.bbox,
                    initial_conf=lost_t.confidence,
                )
                self.active_ghosts[lost_t.track_id] = predictor
                new_ghosts_count += 1

        # Step 3: Advance Kalman prediction for all active ghosts & purge expired
        output_ghost_items: List[GhostBoxItem] = []
        expired_ids = []

        for tid, ghost in list(self.active_ghosts.items()):
            if ghost.ghost_age > self.max_ghost_frames:
                expired_ids.append(tid)
                continue

            pred_bbox, conf, exp_factor = ghost.predict()
            cov_norm = float(np.linalg.norm(ghost.P))

            category_prefix_map = {
                TargetCategory.FACES: "FACE",
                TargetCategory.LICENSE_PLATES: "PLATE",
                TargetCategory.ID_CARDS: "ID",
                TargetCategory.LAPTOP_SCREENS: "SCREEN",
                TargetCategory.PHONE_SCREENS: "PHONE",
            }
            prefix = category_prefix_map.get(ghost.category, "OBJECT")

            output_ghost_items.append(
                GhostBoxItem(
                    original_track_id=ghost.track_id,
                    category=ghost.category,
                    label=f"GHOST {prefix} #{ghost.track_id}",
                    predicted_bbox=pred_bbox,
                    confidence=conf,
                    ghost_age=ghost.ghost_age,
                    covariance_norm=round(cov_norm, 2),
                    expansion_factor=exp_factor,
                    active=True,
                )
            )

        for tid in expired_ids:
            del self.active_ghosts[tid]
            expired_ghosts_count += 1

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        logger.debug(
            f"[GhostBoxService] Frame #{frame_index} predicted in {elapsed_ms}ms | "
            f"Active Ghosts: {len(output_ghost_items)} | New: {new_ghosts_count} | "
            f"Recovered: {recovered_ghosts_count} | Expired: {expired_ghosts_count}"
        )

        return GhostBoxFrameResult(
            frame_index=frame_index,
            active_ghosts_count=len(output_ghost_items),
            new_ghosts_count=new_ghosts_count,
            recovered_ghosts_count=recovered_ghosts_count,
            expired_ghosts_count=expired_ghosts_count,
            prediction_latency_ms=elapsed_ms,
            ghost_boxes=output_ghost_items,
        )

ghost_box_service = GhostBoxService()
