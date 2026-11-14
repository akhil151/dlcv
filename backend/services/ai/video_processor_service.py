import asyncio
import collections
import gc
import json
import os
import subprocess
import threading
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np

from backend.config.ai_settings import ai_settings
from backend.core.logging import logger
from backend.services.ai.blur_service import blur_service
from backend.services.ai.detection_service import detection_service
from backend.services.ai.ghost_box_service import GhostBoxService
from backend.services.ai.tracking_service import TrackingService
from backend.utils.ffmpeg import probe_audio_streams, require_ffmpeg
from backend.utils.storage import storage_manager
from backend.websocket.connection_manager import ws_manager
from backend.workers.task_manager import task_manager

# Broadcast at most this often, so a long video cannot flood the socket.
WS_BROADCAST_MIN_INTERVAL_S = 0.1
ENCODER_WAIT_TIMEOUT_S = 120


class TaskCancelledError(Exception):
    """Raised inside the frame loop when the task was cancelled by a client."""


class _StderrCollector(threading.Thread):
    """Drains the encoder's stderr on a worker thread.

    FFmpeg writes diagnostics to stderr. If that pipe is not consumed while
    frames are pumped into stdin, it can fill and deadlock the encoder. This
    thread keeps the pipe empty and retains the last lines for error reporting.
    """

    def __init__(self, stream: Any) -> None:
        super().__init__(daemon=True)
        self._stream = stream
        self._lines: collections.deque = collections.deque(maxlen=300)

    def run(self) -> None:
        try:
            for line in iter(self._stream.readline, b""):
                self._lines.append(line.decode("utf-8", "replace"))
        except Exception:  # pragma: no cover - stream closed during shutdown
            pass

    def text(self) -> str:
        return "".join(self._lines)[-2000:]

    def close(self) -> None:
        try:
            self._stream.close()
        except Exception:  # pragma: no cover - best effort
            pass


class VideoProcessorService:
    """Fail-closed redaction pipeline.

    Completion means: every input frame was decoded, redacted, written to the
    encoder; the encoder exited successfully; and the produced file was fully
    re-decoded with the expected resolution and frame count.
    """

    def __init__(self) -> None:
        self.detection_service = detection_service
        self.blur_service = blur_service

    # ------------------------------------------------------------------ output

    @staticmethod
    def _validate_output(
        path: str,
        expected_width: Optional[int] = None,
        expected_height: Optional[int] = None,
        expected_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Independently re-decodes the produced file and verifies its integrity."""
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            raise RuntimeError("encoder produced no output file")
        cap = cv2.VideoCapture(path)
        try:
            if not cap.isOpened():
                raise RuntimeError("encoded output cannot be opened independently")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if expected_width is not None and width != int(expected_width):
                raise RuntimeError(
                    f"output width {width} != expected {expected_width}")
            if expected_height is not None and height != int(expected_height):
                raise RuntimeError(
                    f"output height {height} != expected {expected_height}")

            decoded = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame is None or frame.size == 0:
                    raise RuntimeError(f"output frame {decoded + 1} is empty")
                decoded += 1
            if decoded == 0:
                raise RuntimeError("encoded output contains no decodable frames")
            if expected_frames is not None and decoded != int(expected_frames):
                raise RuntimeError(
                    f"output frame count {decoded} != expected {expected_frames}")
            return {
                "frame_count": decoded,
                "width": width,
                "height": height,
                "fps": round(fps, 3),
                "size_bytes": os.path.getsize(path),
            }
        finally:
            cap.release()

    @staticmethod
    def _require_successful_encoder(encoder: subprocess.Popen, stderr_text: Optional[str] = None) -> None:
        if encoder.stdin and not getattr(encoder.stdin, "closed", False):
            encoder.stdin.close()
        if stderr_text is None:
            stream = getattr(encoder, "stderr", None)
            stderr_text = (stream.read() if stream else b"").decode("utf-8", "replace")
        try:
            code = encoder.wait(timeout=ENCODER_WAIT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            encoder.kill()
            encoder.wait(timeout=5)
            raise RuntimeError("ffmpeg did not terminate after stdin closed")
        if code != 0:
            raise RuntimeError(f"ffmpeg exited {code}: {stderr_text[-1000:]}")

    @staticmethod
    def _discard(path: Optional[str]) -> None:
        if not path:
            return
        try:
            if os.path.isfile(path):
                os.remove(path)
                logger.info("Removed incomplete output '%s'", path)
        except OSError as exc:  # pragma: no cover - best effort cleanup
            logger.warning("Could not remove incomplete output '%s': %s", path, exc)

    # ------------------------------------------------------------------ pipeline

    async def process_video_pipeline(self, task_id: str, video_id: str, target_categories: List[str],
                                     blur_style: str = "gaussian", blur_intensity: int = 80,
                                     debug_mode: bool = False) -> Dict[str, Any]:
        task = task_manager.get_task(task_id)
        if task is None:
            raise RuntimeError(f"task {task_id} disappeared before processing")
        raw_path = storage_manager.get_raw_video_path(video_id)
        output_path = os.path.join(storage_manager.base_dir, "redacted_videos", f"{task_id}_protected.mp4")
        cap: Optional[cv2.VideoCapture] = None
        encoder: Optional[subprocess.Popen] = None
        collector: Optional[_StderrCollector] = None
        started = time.time()
        processed = 0
        total_metadata = 0
        redaction_counts: Dict[str, int] = {}
        frames_with_detections = 0
        masks_total = 0
        detection_ms_total = 0.0
        blur_ms_total = 0.0
        last_broadcast = 0.0
        debug_frames_saved = 0
        debug_root: Optional[str] = None
        debug_records: List[Dict[str, Any]] = []

        # Per-invocation state: no tracking/ghost identity can cross tasks.
        tracker = TrackingService()
        ghost_service = GhostBoxService() if ai_settings.GHOST_BOX_ENABLED else None
        save_debug = bool(debug_mode or ai_settings.DEBUG_SAVE_FRAMES)

        try:
            if not raw_path or not os.path.isfile(raw_path):
                raise RuntimeError("input video does not exist")
            cap = cv2.VideoCapture(raw_path)
            if not cap.isOpened():
                raise RuntimeError("input video is unreadable")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            total_metadata = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if width <= 0 or height <= 0 or fps <= 0 or total_metadata <= 0:
                raise RuntimeError("input video has invalid metadata")
            task["metrics"]["total_frames"] = total_metadata
            logger.info(
                "Task %s: %dx%d @ %.3ffps, %d frames from '%s'",
                task_id, width, height, fps, total_metadata, os.path.basename(raw_path))

            if save_debug:
                debug_root = os.path.join(storage_manager.base_dir, "debug", task_id)
                os.makedirs(os.path.join(debug_root, "original"), exist_ok=True)
                os.makedirs(os.path.join(debug_root, "annotated"), exist_ok=True)
                os.makedirs(os.path.join(debug_root, "protected"), exist_ok=True)
                logger.info("Task %s: debug frames -> %s", task_id, debug_root)

            # Resolve and validate the encoder before opening a pipe. A missing
            # or unusable FFmpeg is an explicit failure, never a silent switch.
            ffmpeg_bin = require_ffmpeg()
            out_width, out_height = width, height
            filters: List[str] = []
            if width % 2 or height % 2:
                # H.264/yuv420p requires even dimensions.
                out_width -= width % 2
                out_height -= height % 2
                filters = ["-vf", f"scale={out_width}:{out_height}"]
            command = [
                ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-nostats",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{width}x{height}", "-pix_fmt", "bgr24", "-r", f"{fps:.6f}",
                "-i", "-",
                "-i", raw_path,
                *filters,
                "-map", "0:v:0", "-map", "1:a:0?",
                "-c:v", "libx264", "-preset", ai_settings.FFMPEG_PRESET,
                "-crf", str(ai_settings.FFMPEG_CRF), "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k",
                "-movflags", "+faststart",
                output_path,
            ]
            encoder = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if not encoder.stdin or not encoder.stderr:
                raise RuntimeError("encoder pipes are unavailable")
            collector = _StderrCollector(encoder.stderr)
            collector.start()

            for index, frame in self._frame_generator(cap, width, height):
                if task.get("cancel_requested"):
                    raise TaskCancelledError("cancelled by client")

                det_start = time.time()
                detection_result = self.detection_service.detect_entities(
                    frame, target_categories, frame_index=index)
                detection_ms_total += (time.time() - det_start) * 1000

                tracks_result = tracker.update_tracks(
                    detection_result.detections, frame_index=index)
                ghost_result = (
                    ghost_service.process_ghost_boxes(tracks_result.tracks, frame_index=index)
                    if ghost_service is not None else None)
                ghost_boxes = ghost_result.ghost_boxes if ghost_result is not None else []

                blur_start = time.time()
                anonymized, blur_result = self.blur_service.render_blur(
                    frame, tracks_result.tracks, ghost_boxes,
                    blur_style=blur_style, blur_intensity=blur_intensity,
                    frame_index=index, detections=detection_result.detections)
                blur_ms_total += (time.time() - blur_start) * 1000

                for plan in blur_result.roi_plans:
                    key = plan.category.value
                    redaction_counts[key] = redaction_counts.get(key, 0) + 1
                masks_total += blur_result.total_blurred_regions
                if detection_result.total_detections > 0:
                    frames_with_detections += 1

                if save_debug and debug_frames_saved < ai_settings.DEBUG_MAX_FRAMES:
                    self._save_debug_frames(debug_root, index, frame, anonymized,
                                            detection_result, tracks_result.tracks,
                                            blur_result.roi_plans)
                    debug_records.append({
                        "frame_index": index,
                        "detections": [
                            {"confidence": d.confidence, "bbox": d.bbox.model_dump()}
                            for d in detection_result.detections
                        ],
                        "tracks": [
                            {"id": t.track_id, "state": t.track_state.value,
                             "frames_since_update": t.frames_since_update,
                             "bbox": t.bbox.model_dump()}
                            for t in tracks_result.tracks
                        ],
                        "roi_plans": [
                            {"x1": p.x1, "y1": p.y1, "x2": p.x2, "y2": p.y2,
                             "category": p.category.value, "is_ghost": p.is_ghost}
                            for p in blur_result.roi_plans
                        ],
                    })
                    debug_frames_saved += 1

                try:
                    encoder.stdin.write(anonymized.tobytes())
                except (BrokenPipeError, OSError) as exc:
                    raise RuntimeError(
                        f"encoder pipe failed at frame {index}: {exc}. "
                        f"ffmpeg: {collector.text()[-500:]}") from exc
                processed += 1

                elapsed = max(time.time() - started, 0.001)
                current_fps = processed / elapsed
                task["progress_pct"] = (
                    round(processed / total_metadata * 100, 1) if total_metadata else 0.0)
                task["metrics"].update(
                    processed_frames=processed,
                    throughput_frames=processed,
                    fps=int(current_fps),
                    elapsed_seconds=int(elapsed),
                    remaining_seconds=int(max(0.0, (total_metadata - processed) / max(current_fps, 0.001))),
                    frames_with_detections=frames_with_detections,
                    masks_total=masks_total,
                    new_tracks_total=tracker.total_new_tracks,
                    detection_latency_ms=round(detection_ms_total / processed, 2),
                    blur_latency_ms=round(blur_ms_total / processed, 2),
                )

                now = time.time()
                if now - last_broadcast >= WS_BROADCAST_MIN_INTERVAL_S or processed == total_metadata:
                    last_broadcast = now
                    await ws_manager.broadcast_to_task(task_id, {
                        "task_id": task_id,
                        "frame_index": index,
                        "total_frames": total_metadata,
                        "progress_pct": task["progress_pct"],
                        "fps": task["metrics"]["fps"],
                        "status": "PROCESSING",
                        "current_stage": "Detection + Redaction",
                        "active_detections": detection_result.total_detections,
                        "active_tracks": tracks_result.active_tracks_count,
                        "log_line": (f"Frame {index}/{total_metadata}: "
                                     f"detections={detection_result.total_detections}, "
                                     f"masks={blur_result.total_blurred_regions}"),
                        "detected_objects": [
                            {
                                "id": str(track.track_id),
                                "category": track.category.value,
                                "label": track.label,
                                "confidence": track.confidence,
                                "bbox": track.bbox.model_dump(),
                            }
                            for track in tracks_result.tracks
                            if track.track_state.value in ("NEW", "TRACKED")
                        ],
                    })
                await asyncio.sleep(0)

            if task.get("cancel_requested"):
                raise TaskCancelledError("cancelled by client")
            if processed == 0:
                raise RuntimeError("no frames were decoded from the input video")
            if total_metadata and processed != total_metadata:
                # Container frame counts can be approximate; the decoded count is
                # authoritative, and it is what the output must match.
                logger.warning(
                    "Task %s: container reported %d frames, decoded %d",
                    task_id, total_metadata, processed)
                task["metrics"]["total_frames"] = processed

            self._require_successful_encoder(encoder, collector.text())
            verification = self._validate_output(
                output_path, expected_width=out_width, expected_height=out_height,
                expected_frames=processed)
            input_audio = probe_audio_streams(raw_path)
            output_audio = probe_audio_streams(output_path)

            if debug_root:
                try:
                    with open(os.path.join(debug_root, "detections.json"), "w", encoding="utf-8") as handle:
                        json.dump({"task_id": task_id, "records": debug_records}, handle)
                except OSError as exc:
                    logger.warning("Could not write debug detection log: %s", exc)

            elapsed = round(time.time() - started, 2)
            task["metrics"].update(
                total_frames=processed,
                processed_frames=processed,
                throughput_frames=processed,
                elapsed_seconds=int(elapsed),
                remaining_seconds=0,
                fps=int(processed / max(elapsed, 0.001)),
                avg_processing_fps=round(processed / max(elapsed, 0.001), 2),
                frames_with_detections=frames_with_detections,
                masks_total=masks_total,
                new_tracks_total=tracker.total_new_tracks,
                output_frame_count=verification["frame_count"],
                output_width=verification["width"],
                output_height=verification["height"],
                output_size_bytes=verification["size_bytes"],
                input_audio_streams=input_audio,
                output_audio_streams=output_audio,
                audio_preserved=(bool(output_audio) if input_audio else False),
                debug_frames_saved=debug_frames_saved,
            )
            task.update(status="COMPLETED", progress_pct=100.0, output_verified=True,
                        redaction_counts=redaction_counts,
                        verification=verification,
                        debug_frames_dir=debug_root)
            logger.info(
                "Task %s COMPLETED: %d frames verified at %dx%d, %.2fs",
                task_id, verification["frame_count"], verification["width"],
                verification["height"], elapsed)
            await ws_manager.broadcast_to_task(task_id, {
                "task_id": task_id, "status": "COMPLETED", "progress_pct": 100.0,
                "current_stage": "Completed", "frame_index": processed,
                "total_frames": processed, "fps": task["metrics"]["fps"],
                "log_line": f"Completed: {verification['frame_count']} frames verified",
            })
            return {
                "task_id": task_id, "status": "COMPLETED", "total_frames": processed,
                "output_video_path": output_path, "verification": verification,
            }
        except TaskCancelledError as exc:
            logger.warning("Task %s cancelled: %s", task_id, exc)
            task_manager.cancel_task(task_id, str(exc))
            self._discard(output_path)
            return {"task_id": task_id, "status": "CANCELLED", "reason": str(exc)}
        except asyncio.CancelledError:
            # A cancelled worker must not be left stuck in PROCESSING.
            logger.warning("Task %s cancelled before completion", task_id)
            task_manager.cancel_task(task_id, "processing cancelled")
            self._discard(output_path)
            raise
        except Exception as exc:
            logger.error("Task %s failed: %s", task_id, exc, exc_info=True)
            # Never overwrite a client cancellation with a failure state.
            if task.get("status") != "CANCELLED":
                task_manager.fail_task(task_id, str(exc))
            self._discard(output_path)
            raise
        finally:
            if collector is not None:
                try:
                    collector.join(timeout=2)
                except RuntimeError:  # pragma: no cover
                    pass
                collector.close()
            if cap is not None:
                cap.release()
            if encoder is not None:
                try:
                    if encoder.stdin and not encoder.stdin.closed:
                        encoder.stdin.close()
                except Exception:  # pragma: no cover - best effort
                    pass
                if encoder.poll() is None:
                    try:
                        encoder.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        encoder.kill()
                        encoder.wait(timeout=5)
            gc.collect()

    # --------------------------------------------------------------- helpers

    @staticmethod
    def _frame_generator(cap: cv2.VideoCapture, width: int, height: int) -> Generator[Tuple[int, np.ndarray], None, None]:
        index = 1
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            if frame is None or frame.size == 0:
                raise RuntimeError(f"decoder returned an invalid frame at {index}")
            if frame.shape[1] != width or frame.shape[0] != height:
                raise RuntimeError(
                    f"frame {index} is {frame.shape[1]}x{frame.shape[0]}, "
                    f"expected {width}x{height}; coordinates would be invalid")
            yield index, frame
            index += 1

    @staticmethod
    def _mask_outline(plan: Any) -> Optional[np.ndarray]:
        """Mask silhouette for the debug overlay; never breaks a real job."""
        try:
            return blur_service.mask_outline(plan)
        except Exception:  # pragma: no cover - debug only
            return None

    @staticmethod
    def _save_debug_frames(root: Optional[str], index: int, original: np.ndarray,
                           protected: np.ndarray, detection_result: Any,
                           tracks: List[Any], roi_plans: List[Any]) -> None:
        """Saves annotated detection frames separately from the protected video.

        Annotated images contain detector boxes (green), track IDs (blue) and the
        final redaction ROIs (red). They are never written into the output video.
        """
        if not root:
            return
        try:
            name = f"frame_{index:06d}.jpg"
            jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            cv2.imwrite(os.path.join(root, "original", name), original, jpeg_params)
            annotated = original.copy()
            for detection in detection_result.detections:
                box = detection.bbox
                x1 = int(box.x / 100.0 * annotated.shape[1])
                y1 = int(box.y / 100.0 * annotated.shape[0])
                x2 = int((box.x + box.width) / 100.0 * annotated.shape[1])
                y2 = int((box.y + box.height) / 100.0 * annotated.shape[0])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(annotated, f"FACE {detection.confidence:.2f}", (x1, max(10, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
            for track in tracks:
                if track.track_state.value not in ("NEW", "TRACKED"):
                    continue
                box = track.bbox
                x1 = int(box.x / 100.0 * annotated.shape[1])
                y1 = int(box.y / 100.0 * annotated.shape[0])
                cv2.putText(annotated, f"ID {track.track_id}", (x1, min(annotated.shape[0] - 6, y1 + 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 128, 0), 1, cv2.LINE_AA)
            # Draw the exact mask silhouette that was applied (rounded, not a
            # rectangle) so debug frames never misrepresent the redaction area.
            for plan in roi_plans:
                outline = VideoProcessorService._mask_outline(plan)
                if outline is None:
                    continue
                cv2.polylines(annotated, [outline], True, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(root, "annotated", name), annotated, jpeg_params)
            cv2.imwrite(os.path.join(root, "protected", name), protected, jpeg_params)
        except Exception as exc:  # Debug output must never break a real job.
            logger.warning("Could not save debug frame %s: %s", index, exc)


video_processor_service = VideoProcessorService()
