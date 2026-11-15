"""Renders laptop-screen redaction comparison images (Phase 4).

For each representative real photo this writes five images into
``docs/validation/laptop_screens/<key>/``:

    01_original.png          source frame
    02_detected_boundary.png device box (green), refined display quad (magenta)
                             and the exact applied mask silhouette (red)
    03_gaussian.png          Gaussian-blur redaction
    04_pixelate.png          pixelation redaction
    05_blackout.png          blackout redaction

The mask silhouette is taken from BlurService itself, so the image shows the
real redacted area including padding. When the display boundary could not be
separated the overlay says so and the redaction covers the device box.

Usage (project root, model downloaded, images cached by the evaluation script):

    python scripts/evaluate_laptop_screen_detection.py
    python scripts/render_laptop_screen_redaction_comparison.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

ANNOTATIONS = os.path.join(PROJECT_ROOT, "scripts", "laptop_screen_annotations.json")
CACHE_DIR = os.path.join(PROJECT_ROOT, ".cache", "validation", "screens")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "docs", "validation", "laptop_screens")
REPRESENTATIVE = ("lenovo", "msi_wiki", "charts_desk", "surface7")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", default=CACHE_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--keys", nargs="*", default=list(REPRESENTATIVE))
    args = parser.parse_args(argv)

    from backend.services.ai.blur_service import blur_service
    from backend.services.ai.detection_service import DetectionService

    with open(ANNOTATIONS, encoding="utf-8") as handle:
        entries = {item["key"]: item for item in json.load(handle)["images"]}

    detector = DetectionService()
    written = []
    for key in args.keys:
        entry = entries.get(key)
        image_path = os.path.join(args.images_dir, key + ".jpg")
        if entry is None or not os.path.isfile(image_path):
            print(f"SKIP {key}: image or annotation unavailable")
            continue
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"SKIP {key}: unreadable image")
            continue

        detections = detector._detect_laptop_screens(frame, 1)
        if not detections:
            print(f"SKIP {key}: no laptop detected (nothing to redact)")
            continue

        target_dir = os.path.join(args.output_dir, key)
        os.makedirs(target_dir, exist_ok=True)
        cv2.imwrite(os.path.join(target_dir, "01_original.png"), frame)

        annotated = frame.copy()
        for detection in detections:
            box = detection.bbox
            height, width = frame.shape[:2]
            x1 = int(box.x / 100.0 * width)
            y1 = int(box.y / 100.0 * height)
            x2 = int((box.x + box.width) / 100.0 * width)
            y2 = int((box.y + box.height) / 100.0 * height)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if detection.quad:
                cv2.polylines(annotated, [np.round(np.asarray(detection.quad)).astype(np.int32)],
                              True, (255, 0, 255), 2)
            cv2.putText(annotated,
                        f"{detection.label} {detection.confidence:.2f} ({detection.region_kind})",
                        (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0), 1, cv2.LINE_AA)
        _, report = blur_service.render_blur(
            frame.copy(), [], [], blur_style="blackout", blur_intensity=100,
            frame_index=1, detections=detections)
        for plan in report.roi_plans:
            outline = blur_service.mask_outline(plan)
            if outline is not None:
                cv2.polylines(annotated, [outline], True, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(target_dir, "02_detected_boundary.png"), annotated)

        for index, style in enumerate(("gaussian", "pixelate", "blackout"), start=3):
            protected, _ = blur_service.render_blur(
                frame.copy(), [], [], blur_style=style, blur_intensity=80,
                frame_index=1, detections=detections)
            path = os.path.join(target_dir, f"{index:02d}_{style}.png")
            cv2.imwrite(path, protected)
            written.append(path)
        written.extend([
            os.path.join(target_dir, "01_original.png"),
            os.path.join(target_dir, "02_detected_boundary.png"),
        ])
        print(f"{key}: wrote comparison images to {target_dir}")

    if not written:
        print("No comparison images written.")
        return 1
    print("\nWritten files:")
    for path in sorted(written):
        print(" ", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
