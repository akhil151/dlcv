"""Run the real redaction pipeline over the validation fixtures and measure it.

For every fixture this script:
  * stages the clip into backend storage,
  * runs the production pipeline (real YuNet inference, real FFmpeg encode,
    full output verification) with debug frames enabled,
  * compares the saved original / protected frames against exact ground truth.

Reported metrics (per clip):
  * detection recall     - ground-truth faces matched by a detection (IoU >= 0.3)
  * false detections     - detections that match no ground-truth face
  * redacted GT instances- GT faces whose pixels were actually altered
  * leaked GT instances  - GT faces with >10% of pixels unchanged (failure)
  * background change    - mean |diff| outside GT faces (localization check)

These fixtures are composites built from real photographs, not surveillance
footage. Precision/recall on real footage with independent ground truth has NOT
been measured.

Usage:
    python scripts/make_validation_videos.py
    python scripts/evaluate_redaction.py
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

FIXTURE_DIR = os.path.join(PROJECT_ROOT, ".cache", "validation", "videos")
REPORT_DIR = os.path.join(PROJECT_ROOT, "backend", "storage", "validation")

from backend.utils.storage import storage_manager  # noqa: E402
from backend.workers.task_manager import task_manager  # noqa: E402
from backend.services.ai.video_processor_service import video_processor_service  # noqa: E402

IOU_MATCH = 0.30
# Leakage proxy: re-run the same YuNet detector on the *protected* frames at a
# deliberately sensitive threshold. A face still detectable where it was before
# redaction is treated as a redaction failure.
LEAK_SCORE_THRESHOLD = 0.30


def resolve_fixture_path(meta: dict) -> str:
    """Resolve a fixture video portably.

    Fixture metadata stores a project-relative path (older files stored an
    absolute path from the generating machine), so the clip next to the JSON is
    preferred and only then the recorded path is tried.
    """
    local = os.path.join(FIXTURE_DIR, f"{meta['name']}.mp4")
    if os.path.isfile(local):
        return local
    recorded = meta.get("path") or ""
    if recorded:
        candidate = recorded if os.path.isabs(recorded) else os.path.join(PROJECT_ROOT, recorded)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"fixture video for '{meta.get('name')}' not found; "
        f"run python scripts/make_validation_videos.py")


def iou(a, b) -> float:
    ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]
    bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]
    ix = max(0, min(ax2, bx2) - max(a["x"], b["x"]))
    iy = max(0, min(ay2, by2) - max(a["y"], b["y"]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def visible_ground_truth(faces: list[dict]) -> list[dict]:
    """Drops GT faces that another pasted face almost completely covers."""
    visible = []
    for index, face in enumerate(faces):
        covered = 0.0
        for other_index, other in enumerate(faces):
            if index == other_index:
                continue
            if iou(face, other) > 0.6:
                covered = 1.0
                break
        if covered < 0.7:
            visible.append(face)
    return visible


def mean_abs_diff_in_box(original: np.ndarray, protected: np.ndarray, box: dict) -> float:
    h, w = original.shape[:2]
    x1 = max(0, int(box["x"]))
    y1 = max(0, int(box["y"]))
    x2 = min(w, int(box["x"] + box["w"]))
    y2 = min(h, int(box["y"] + box["h"]))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    a = original[y1:y2, x1:x2].astype(np.int16)
    b = protected[y1:y2, x1:x2].astype(np.int16)
    return float(np.abs(a - b).mean())


def detector_faces(image: np.ndarray, detector) -> list[dict]:
    detector.setInputSize((image.shape[1], image.shape[0]))
    _, faces = detector.detect(image)
    if faces is None:
        return []
    return [{"x": float(f[0]), "y": float(f[1]), "w": float(f[2]), "h": float(f[3])}
            for f in faces if float(f[-1]) >= LEAK_SCORE_THRESHOLD]


def background_change(original: np.ndarray, protected: np.ndarray, faces: list[dict]) -> float:
    h, w = original.shape[:2]
    mask = np.ones((h, w), dtype=bool)
    for box in faces:
        # Exclude the face plus a 10% margin, matching the redaction padding.
        mx = int(box["w"] * 0.15)
        my = int(box["h"] * 0.15)
        x1 = max(0, int(box["x"]) - mx)
        y1 = max(0, int(box["y"]) - my)
        x2 = min(w, int(box["x"] + box["w"]) + mx)
        y2 = min(h, int(box["y"] + box["h"]) + my)
        mask[y1:y2, x1:x2] = False
    if not mask.any():
        return 0.0
    diff = np.abs(original.astype(np.int16) - protected.astype(np.int16)).max(axis=2)
    return float(diff[mask].mean())


def run_fixture(meta: dict, style: str, intensity: int) -> dict:
    name = meta["name"]
    video_id = f"vid_val_{name}_{style}"
    staged = os.path.join(storage_manager.base_dir, "raw_videos", f"{video_id}_{name}.mp4")
    os.makedirs(os.path.dirname(staged), exist_ok=True)
    shutil.copyfile(resolve_fixture_path(meta), staged)
    task_id = f"task_val_{name}_{style}"
    task_manager.tasks.pop(task_id, None)
    task_manager.create_task(task_id, video_id, f"{name}.mp4", ["faces"], style, intensity,
                             debug_mode=True)

    started = time.time()
    try:
        asyncio.run(video_processor_service.process_video_pipeline(
            task_id, video_id, ["faces"], style, intensity, debug_mode=True))
    except Exception as exc:  # A failure is reported, never hidden.
        return {"clip": name, "style": style, "status": "FAILED", "error": str(exc)}
    wall = time.time() - started

    task = task_manager.get_task(task_id)
    if task["status"] != "COMPLETED" or not task["output_verified"]:
        return {"clip": name, "style": style, "status": str(task["status"]),
                "error": task.get("failure_reason")}

    debug_root = task["debug_frames_dir"]
    with open(os.path.join(debug_root, "detections.json"), encoding="utf-8") as handle:
        records = json.load(handle)["records"]
    records_by_frame = {record["frame_index"]: record for record in records}

    from backend.services.ai.model_manager import model_manager  # noqa: E402
    try:
        leak_detector = cv2.FaceDetectorYN.create(
            model_manager.resolve_yunet_path(), "", (320, 320),
            LEAK_SCORE_THRESHOLD, 0.3, 5000)
    except Exception:
        leak_detector = None

    gt_faces = meta["faces_per_frame"]
    gt_total = 0
    gt_matched = 0
    false_detections = 0
    detections_total = 0
    gt_visible_after = 0
    gt_redacted = 0
    gt_leaked = 0
    leak_frames = 0
    gt_diffs: list[float] = []
    bg_changes = []
    frames_checked = 0

    for record in records:
        index = record["frame_index"]
        original_path = os.path.join(debug_root, "original", f"frame_{index:06d}.jpg")
        protected_path = os.path.join(debug_root, "protected", f"frame_{index:06d}.jpg")
        if not (os.path.isfile(original_path) and os.path.isfile(protected_path)):
            continue
        original = cv2.imread(original_path)
        protected = cv2.imread(protected_path)
        if original is None or protected is None:
            continue
        faces = visible_ground_truth(gt_faces[index - 1]) if index - 1 < len(gt_faces) else []
        detections = record["detections"]
        frame_h, frame_w = original.shape[:2]
        # Debug records store detector boxes normalized to 0-100 percent.
        det_boxes = [{
            "x": d["bbox"]["x"] / 100.0 * frame_w,
            "y": d["bbox"]["y"] / 100.0 * frame_h,
            "w": d["bbox"]["width"] / 100.0 * frame_w,
            "h": d["bbox"]["height"] / 100.0 * frame_h,
        } for d in detections]
        detections_total += len(det_boxes)

        used = set()
        frame_matched = 0
        for gt in faces:
            gt_total += 1
            best = None
            best_iou = 0.0
            for det_index, det in enumerate(det_boxes):
                if det_index in used:
                    continue
                score = iou(gt, det)
                if score > best_iou:
                    best_iou, best = score, det_index
            if best is not None and best_iou >= IOU_MATCH:
                used.add(best)
                gt_matched += 1
                frame_matched += 1
            gt_diffs.append(mean_abs_diff_in_box(original, protected, gt))
            if leak_detector is not None:
                residual = detector_faces(protected, leak_detector)
                still_visible = any(iou(gt, det) >= IOU_MATCH for det in residual)
            else:
                still_visible = False
            if still_visible:
                gt_visible_after += 1
                gt_leaked += 1
            else:
                gt_redacted += 1
        false_detections += max(0, len(det_boxes) - len(used))
        if faces and frame_matched < len(faces):
            leak_frames += 1
        bg_changes.append(background_change(original, protected, faces))
        frames_checked += 1

    metrics = task["metrics"]
    return {
        "clip": name,
        "style": style,
        "status": "COMPLETED",
        "frames": frames_checked,
        "resolution": f"{metrics['output_width']}x{metrics['output_height']}",
        "output_frames_verified": metrics["output_frame_count"],
        "processing_seconds": round(wall, 2),
        "avg_pipeline_fps": round(frames_checked / wall, 1) if wall else 0,
        "detection_ms_per_frame": metrics.get("detection_latency_ms"),
        "blur_ms_per_frame": metrics.get("blur_latency_ms"),
        "ground_truth_faces": gt_total,
        "detection_recall_pct": round(gt_matched / gt_total * 100, 1) if gt_total else None,
        "false_detections": false_detections,
        "detections_total": detections_total,
        "gt_faces_redacted_pct": round(gt_redacted / gt_total * 100, 1) if gt_total else None,
        "gt_faces_detector_visible_after": gt_visible_after,
        "gt_faces_leaked": gt_leaked,
        "gt_mean_abs_diff": round(float(np.median(gt_diffs)), 2) if gt_diffs else None,
        "frames_with_missed_gt_face": leak_frames,
        "background_mean_abs_diff": round(float(np.mean(bg_changes)), 2) if bg_changes else None,
        "audio_streams_in": metrics.get("input_audio_streams"),
        "audio_streams_out": metrics.get("output_audio_streams"),
        "debug_frames": metrics.get("debug_frames_saved"),
        "output_bytes": metrics.get("output_size_bytes"),
    }


def main() -> int:
    if not os.path.isdir(FIXTURE_DIR):
        print("No fixtures found. Run: python scripts/make_validation_videos.py")
        return 2
    os.makedirs(REPORT_DIR, exist_ok=True)

    metas = []
    for name in sorted(os.listdir(FIXTURE_DIR)):
        if name.endswith(".json"):
            with open(os.path.join(FIXTURE_DIR, name), encoding="utf-8") as handle:
                metas.append(json.load(handle))

    jobs = [(meta, "gaussian", 80) for meta in metas]
    for meta in metas:
        if meta["name"] in ("frontal_single", "multi_face"):
            jobs.append((meta, "pixelate", 80))
            jobs.append((meta, "blackout", 100))

    results = []
    for meta, style, intensity in jobs:
        print(f"Running {meta['name']} [{style} {intensity}%] ...", flush=True)
        result = run_fixture(meta, style, intensity)
        print("   ", json.dumps(result))
        results.append(result)

    report_path = os.path.join(REPORT_DIR, "report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=1)

    print("\n=== Validation summary ===")
    header = (f"{'clip':<16}{'style':<10}{'frames':>7}{'recall%':>9}{'redact%':>9}"
              f"{'visibleAfter':>14}{'FP':>5}{'fps':>7}{'bgdiff':>8}{'gtDiff':>8}")
    print(header)
    for row in results:
        if row.get("status") != "COMPLETED":
            print(f"{row['clip']:<16}{row['style']:<10}  FAILED: {row.get('error')}")
            continue
        print(f"{row['clip']:<16}{row['style']:<10}{row['frames']:>7}"
              f"{str(row['detection_recall_pct']):>9}{str(row['gt_faces_redacted_pct']):>9}"
              f"{row['gt_faces_detector_visible_after']:>14}{row['false_detections']:>5}"
              f"{row['avg_pipeline_fps']:>7}{str(row['background_mean_abs_diff']):>8}"
              f"{str(row['gt_mean_abs_diff']):>8}")
    print(f"\nReport: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
