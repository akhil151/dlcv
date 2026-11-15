"""Render before/after comparison images for identity-document redaction.

Produces five images from one real photo:

    <prefix>_original.jpg      the untouched frame
    <prefix>_detected.jpg      detected document quad and bounding box
    <prefix>_gaussian.jpg      Gaussian-blurred document
    <prefix>_pixelate.jpg      pixelated document
    <prefix>_blackout.jpg      black document

The comparisons use the real corner-heatmap detector and the real BlurService
masks, so they show exactly what the pipeline does. Images are written outside
the source tree (backend/storage/validation/document_comparison) and are
git-ignored.

Usage:
    python scripts/render_document_redaction_comparison.py \
        --image .cache/docmodels/midv_val/000_img.jpg --prefix id_card
    python scripts/render_document_redaction_comparison.py \
        --image .cache/docmodels/passport_samples/0.jpg --prefix passport
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
                              "document_comparison")


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
        cv2.putText(annotated, f"DOCUMENT {detection.confidence:.2f}",
                    (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA)
    return annotated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True,
                        help="Input photo containing an ID card or passport.")
    parser.add_argument("--prefix", default="document",
                        help="Filename prefix for the rendered images.")
    parser.add_argument("--pad", action="store_true",
                        help="Composite the input onto a neutral 1280x960 scene. "
                             "Use for flat scans where the document fills the "
                             "whole frame (the detector needs some background).")
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
    if args.pad:
        canvas = np.full((960, 1280, 3), 60, np.uint8)
        scale = min(0.75 * 1280 / frame.shape[1], 0.75 * 960 / frame.shape[0])
        resized = cv2.resize(frame, (int(frame.shape[1] * scale),
                                     int(frame.shape[0] * scale)))
        x = (1280 - resized.shape[1]) // 2
        y = (960 - resized.shape[0]) // 2
        canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        frame = canvas
    detection_service.validate_available(["id_cards"])
    result = detection_service.detect_entities(frame, ["id_cards"], 0)
    if not result.detections:
        print("ERROR: no identity document was detected in this image", file=sys.stderr)
        return 1

    os.makedirs(args.output_dir, exist_ok=True)
    jpeg = [int(cv2.IMWRITE_JPEG_QUALITY), 92]
    written = []

    original_path = os.path.join(args.output_dir, f"{args.prefix}_original.jpg")
    cv2.imwrite(original_path, frame, jpeg)
    written.append(original_path)

    annotated_path = os.path.join(args.output_dir, f"{args.prefix}_detected.jpg")
    cv2.imwrite(annotated_path, _annotate(frame, result.detections), jpeg)
    written.append(annotated_path)

    for mode in ("gaussian", "pixelate", "blackout"):
        # Detections are redacted directly, exactly as the fail-safe first-frame
        # path in the video pipeline does.
        output, report = blur_service.render_blur(
            frame.copy(), [], [], blur_style=mode, blur_intensity=80,
            frame_index=1, detections=result.detections)
        path = os.path.join(args.output_dir, f"{args.prefix}_{mode}.jpg")
        cv2.imwrite(path, output, jpeg)
        written.append(path)
        print(f"  {mode}: {report.total_blurred_regions} document region(s) masked")

    with open(os.path.join(args.output_dir, f"{args.prefix}_detections.json"),
              "w", encoding="utf-8") as handle:
        json.dump({
            "image": os.path.abspath(args.image),
            "detections": [
                {"confidence": detection.confidence,
                 "bbox": detection.bbox.model_dump()}
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
