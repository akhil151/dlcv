"""Render before/after comparison images for license-plate redaction.

Produces five images from one real photo:

    original.jpg          the untouched frame
    detected_boxes.jpg    detector bounding boxes with confidence scores
    gaussian.jpg          Gaussian-blurred plates
    pixelate.jpg          pixelated plates
    blackout.jpg          black plates

The comparisons use the real detector and the real BlurService masks, so they
show exactly what the pipeline does. Images are written outside the source
tree (backend/storage/validation/plate_comparison) and are git-ignored.

Usage:
    python scripts/render_plate_redaction_comparison.py
    python scripts/render_plate_redaction_comparison.py --image car.jpg
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "backend", "storage", "validation",
                              "plate_comparison")
DEFAULT_IMAGE = os.path.join(PROJECT_ROOT, ".cache", "platemodel", "sample.jpeg")


def _annotate(frame: np.ndarray, detections) -> np.ndarray:
    annotated = frame.copy()
    height, width = frame.shape[:2]
    for detection in detections:
        box = detection.bbox
        x1 = int(round(box.x / 100.0 * width))
        y1 = int(round(box.y / 100.0 * height))
        x2 = int(round((box.x + box.width) / 100.0 * width))
        y2 = int(round((box.y + box.height) / 100.0 * height))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"PLATE {detection.confidence:.2f}",
                    (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA)
    return annotated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=DEFAULT_IMAGE,
                        help="Input photo containing a license plate.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from backend.services.ai.blur_service import blur_service
    from backend.services.ai.detection_service import detection_service

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"ERROR: could not read image '{args.image}'", file=sys.stderr)
        return 1
    detection_service.validate_available(["license_plates"])
    result = detection_service.detect_entities(frame, ["license_plates"], 0)
    if not result.detections:
        print("ERROR: no license plate was detected in this image", file=sys.stderr)
        return 1

    os.makedirs(args.output_dir, exist_ok=True)
    jpeg = [int(cv2.IMWRITE_JPEG_QUALITY), 92]
    written = []

    original_path = os.path.join(args.output_dir, "original.jpg")
    cv2.imwrite(original_path, frame, jpeg)
    written.append(original_path)

    annotated_path = os.path.join(args.output_dir, "detected_boxes.jpg")
    cv2.imwrite(annotated_path, _annotate(frame, result.detections), jpeg)
    written.append(annotated_path)

    for mode in ("gaussian", "pixelate", "blackout"):
        # Detections are redacted directly, exactly as the fail-safe first-frame
        # path in the video pipeline does.
        output, report = blur_service.render_blur(
            frame.copy(), [], [], blur_style=mode, blur_intensity=80,
            frame_index=1, detections=result.detections)
        path = os.path.join(args.output_dir, f"{mode}.jpg")
        cv2.imwrite(path, output, jpeg)
        written.append(path)
        print(f"  {mode}: {report.total_blurred_regions} plate region(s) masked")

    with open(os.path.join(args.output_dir, "detections.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "image": os.path.abspath(args.image),
            "detections": [
                {"confidence": detection.confidence, "bbox": detection.bbox.model_dump()}
                for detection in result.detections
            ],
        }, handle, indent=1)

    print(f"Input: {os.path.abspath(args.image)}")
    print(f"Detections: {len(result.detections)} "
          f"(inference {result.inference_time_ms} ms)")
    for path in written:
        print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
