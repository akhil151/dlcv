"""Render a visual comparison of the current face redaction against the old one.

Uses a real frame from a validation fixture, real YuNet detections and the real
BlurService, so the panels show what the pipeline actually produces. The
"previous" panel re-creates the earlier behaviour (a hard, padded *rectangle*
with a fixed 10 %-of-smaller-side margin) for a side-by-side check that privacy
coverage was not traded away for looks.

Usage:
    python scripts/make_validation_videos.py      # once, builds the fixtures
    python scripts/render_redaction_comparison.py
    python scripts/render_redaction_comparison.py --clip multi_face --frame 40 --style pixelate
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

FIXTURE_DIR = os.path.join(PROJECT_ROOT, ".cache", "validation", "videos")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "backend", "storage", "validation")
MODEL_PATH = os.path.join(PROJECT_ROOT, "face_detection_yunet_2023mar.onnx")
PANEL_W, PANEL_H = 640, 480
LABEL_H = 34


def load_detections(frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
    """Real YuNet detections as pixel rectangles (x, y, w, h, confidence)."""
    detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (320, 320), 0.25, 0.3, 5000)
    detector.setInputSize((frame.shape[1], frame.shape[0]))
    _, faces = detector.detect(frame)
    if faces is None:
        return []
    return [(int(f[0]), int(f[1]), int(f[2]), int(f[3]), float(f[-1])) for f in faces
            if float(f[-1]) >= 0.60]


def previous_rectangle_redaction(frame: np.ndarray, style: str) -> np.ndarray:
    """The pre-change behaviour: one hard, padded rectangle per face.

    Padding was 10 % of the box's smaller side clamped to 2..32 px, the blur was
    applied to the whole rectangle and its edge was not feathered. Reproduced
    here only to document the difference; the live pipeline no longer uses it.
    """
    from backend.config.ai_settings import ai_settings

    out = frame.copy()
    h, w = frame.shape[:2]
    for x, y, bw, bh, _ in load_detections(frame):
        padding = int(round(min(bw, bh) * 0.10))
        padding = max(2, min(32, padding))
        x1, y1 = max(0, x - padding), max(0, y - padding)
        x2, y2 = min(w, x + bw + padding), min(h, y + bh + padding)
        roi = out[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        if style == "blackout":
            out[y1:y2, x1:x2] = 0
        elif style == "pixelate":
            block = max(ai_settings.PIXELATE_MIN_BLOCK_PX,
                        int(math.ceil(min(roi.shape[:2]) / 10)))
            small = cv2.resize(roi, (max(1, roi.shape[1] // block), max(1, roi.shape[0] // block)),
                               interpolation=cv2.INTER_NEAREST)
            out[y1:y2, x1:x2] = cv2.resize(small, (roi.shape[1], roi.shape[0]),
                                           interpolation=cv2.INTER_NEAREST)
        else:
            sigma = max(ai_settings.GAUSSIAN_MIN_SIGMA_PX,
                        min(roi.shape[:2]) * (ai_settings.GAUSSIAN_SIGMA_MIN_RATIO
                                              + ai_settings.GAUSSIAN_SIGMA_GAIN * 0.8))
            ksize = int(math.ceil(sigma * 4)) | 1
            ksize = max(3, min(ai_settings.GAUSSIAN_MAX_KERNEL, ksize))
            out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (ksize, ksize), sigmaX=sigma)
    return out


def improved_redaction(frame: np.ndarray, style: str, intensity: int) -> np.ndarray:
    """The live pipeline path: real DetectionService + TrackingService + BlurService."""
    from backend.services.ai.blur_service import blur_service
    from backend.services.ai.detection_service import detection_service
    from backend.services.ai.tracking_service import TrackingService

    detections = detection_service.detect_entities(frame, ["faces"], frame_index=1).detections
    tracks = TrackingService().update_tracks(detections, frame_index=1).tracks
    protected, _ = blur_service.render_blur(
        frame, tracks, [], blur_style=style, blur_intensity=intensity,
        frame_index=1, detections=detections)
    return protected


def panel(image: np.ndarray, label: str) -> np.ndarray:
    canvas = np.full((image.shape[0] + LABEL_H, image.shape[1], 3), 28, np.uint8)
    canvas[LABEL_H:] = image
    cv2.putText(canvas, label, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (235, 235, 235), 1, cv2.LINE_AA)
    return canvas


def annotate(frame: np.ndarray) -> np.ndarray:
    out = frame.copy()
    for x, y, bw, bh, confidence in load_detections(frame):
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 255, 0), 1)
        cv2.putText(out, f"face {confidence:.2f}", (x, max(12, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", default="frontal_single")
    parser.add_argument("--frame", type=int, default=20, help="1-based frame index")
    parser.add_argument("--style", default="gaussian",
                        choices=("gaussian", "pixelate", "blackout"))
    parser.add_argument("--intensity", type=int, default=80)
    args = parser.parse_args()

    video_path = os.path.join(FIXTURE_DIR, f"{args.clip}.mp4")
    if not os.path.isfile(video_path):
        raise SystemExit(f"Fixture not found: {video_path}. "
                         f"Run python scripts/make_validation_videos.py first.")
    if not os.path.isfile(MODEL_PATH):
        raise SystemExit("YuNet model missing; run python scripts/download_yunet.py")

    cap = cv2.VideoCapture(video_path)
    frame = None
    for index in range(1, args.frame + 1):
        ok, candidate = cap.read()
        if not ok:
            break
        if index == args.frame:
            frame = candidate
    cap.release()
    if frame is None:
        raise SystemExit(f"Could not read frame {args.frame} from {video_path}")

    detections = load_detections(frame)
    if not detections:
        raise SystemExit(f"No face detected in {args.clip} frame {args.frame}; pick another frame.")

    original_panel = panel(frame, "1. Original frame")
    annotated_panel = panel(annotate(frame), "2. Detected face box (YuNet, real inference)")
    previous_panel = panel(previous_rectangle_redaction(frame, args.style),
                           f"3. Previous redaction ({args.style}, hard padded rectangle)")
    improved_panel = panel(improved_redaction(frame, args.style, args.intensity),
                           f"4. Improved redaction ({args.style}, face-proportional mask)")

    top = np.hstack([original_panel, annotated_panel])
    bottom = np.hstack([previous_panel, improved_panel])
    comparison = np.vstack([top, bottom])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"redaction_comparison_{args.clip}_{args.style}.png")
    cv2.imwrite(out_path, comparison)

    # Machine-readable record of what is shown, so the report can reference it.
    meta = {
        "clip": args.clip, "frame": args.frame, "style": args.style,
        "intensity": args.intensity, "panels": [
            "original", "detected_box", "previous_rectangle", "improved_mask"],
        "detections": [
            {"x": x, "y": y, "w": bw, "h": bh, "confidence": round(c, 3)}
            for x, y, bw, bh, c in detections],
        "comparison_image": out_path,
    }
    with open(os.path.join(OUTPUT_DIR, f"redaction_comparison_{args.clip}_{args.style}.json"),
              "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=1)
    print(f"Detections: {meta['detections']}")
    print(f"Comparison written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
