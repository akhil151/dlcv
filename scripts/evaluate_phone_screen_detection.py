"""Evaluation script for Phase 5 phone-screen detection and redaction.

Runs the phone-screen detector on the annotated validation photos and
reports per-image coverage, quad IoU, fallback rates, and aggregate
statistics. Results live in scripts/eval_results_phone.json.

Dependencies: cv2, numpy, onnxruntime, the RT-DETR phone ONNX model,
and the phone-screen annotations JSON.
"""

import json
import os
import cv2
import numpy as np

# Project root: this script lives at scripts/evaluate_phone_screen_detection.py.
# Going up 1 level from the scripts/ directory reaches the dlcv project root.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
assert os.path.isdir(os.path.join(PROJECT_ROOT, "backend")), (
    f"Not at project root: {PROJECT_ROOT}"
)
ANNOTATIONS = os.path.join(PROJECT_ROOT, "scripts", "phone_screen_annotations.json")
PHOTO_DIR = os.path.join(PROJECT_ROOT, ".cache", "validation", "screens")
OUTPUT = os.path.join(PROJECT_ROOT, "scripts", "eval_results_phone.json")


def load_annotations():
    with open(ANNOTATIONS, encoding="utf-8") as f:
        return json.load(f)


def quad_mask(quad, shape):
    """Rasterize a quadrilateral into a boolean mask."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(np.asarray(quad)).astype(np.int32)], 1)
    return mask.astype(bool)


def iou(mask1, mask2):
    """Pixel intersection over union of two boolean masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(intersection / union) if union else 0.0


def evaluate():
    ann = load_annotations()
    photos = ann["images"]
    from backend.services.ai.phone_screen_detection_service import (
        PhoneScreenDetectionService,
        REGION_SCREEN_QUAD,
        REGION_PHONE_FALLBACK,
    )
    service = PhoneScreenDetectionService()
    results = []

    for entry in photos:
        key = entry["key"]
        filename = entry["file"]
        image_path = os.path.join(PHOTO_DIR, filename)
        if not os.path.isfile(image_path):
            print(f"SKIP: {filename} not found in {PHOTO_DIR}")
            results.append(
                {"key": key, "filename": filename, "skipped": True,
                 "reason": "image file absent"}
            )
            continue

        frame = cv2.imread(image_path)
        if frame is None:
            print(f"SKIP: {filename} could not be read")
            results.append({"key": key, "filename": filename, "skipped": True,
                            "reason": "unreadable image"})
            continue

        scale = frame.shape[1] / float(ann.get("reference_width", 1280))
        screen_quad = np.array(
            [[x * scale, y * scale] for x, y in entry["screen_quad"]],
            dtype=np.float32,
        )
        keyboard_rect = entry.get("keyboard_rect")
        if keyboard_rect:
            keyboard_rect = [v * scale for v in keyboard_rect]

        # Run the phone detector on this frame
        try:
            detections = service.detect(frame, frame_index=0)
        except Exception as exc:
            print(f"ERR: {filename} detector failed: {exc}")
            results.append({"key": key, "filename": filename, "error": str(exc)})
            continue

        if not detections:
            print(f"{key}: detections=0 (no phone found)")
            results.append(
                {"key": key, "filename": filename, "detections": 0,
                 "quad_iou": None, "fallback": None}
            )
            continue

        # Use the first (best) detection
        det = detections[0]
        quad = np.array(det.quad, dtype=np.float32)
        has_quad = bool(det.quad) and len(det.quad) >= 3
        fallback = det.region_kind == REGION_PHONE_FALLBACK

        # Coverage = share of annotated display pixels that the mask covers.
        screen_mask = quad_mask(screen_quad, frame.shape)
        if has_quad:
            detected_mask = quad_mask(quad, frame.shape)
            iou_val = iou(detected_mask, screen_mask)
            annotated_pixels = int(screen_mask.sum())
            covered = int(np.logical_and(detected_mask, screen_mask).sum())
            coverage = (covered / annotated_pixels * 100.0) if annotated_pixels else None
        else:
            iou_val = None
            coverage = None

        results.append(
            {
                "key": key,
                "filename": filename,
                "license": entry.get("license", "unknown"),
                "detections": len(detections),
                "region_kind": det.region_kind,
                "quad_iou": iou_val,
                "fallback": fallback,
                "coverage_pct": round(coverage, 2) if coverage else None,
                "detected_confidence": round(det.confidence, 3),
            }
        )
        print(
            f"{key}: detections={len(detections)}, region_kind={det.region_kind}, "
            f"IoU={iou_val}, coverage={coverage:.1f}%, conf={det.confidence:.3f}"
        )

    # Write aggregate report
    total = len([r for r in results if "error" not in r])
    detected = [r for r in results if r.get("detections", 0) > 0]
    fallbacks = [r for r in detected if r.get("fallback")]
    refined = [r for r in detected
               if r.get("quad_iou") is not None and not r.get("fallback")]
    quads = [r for r in detected if r.get("quad_iou") is not None]
    coverages = [r["coverage_pct"] for r in detected
                 if r.get("coverage_pct") is not None]

    report = {
        "total_images": total,
        "total_annotations": len(photos),
        "images_with_detection": len(detected),
        "detection_rate_pct": round(len(detected) / total * 100, 1) if total else None,
        "images_with_fallback": len(fallbacks),
        "images_with_quad_refinement": len(refined),
        "fallback_rate_pct": round(len(fallbacks) / len(detected) * 100, 1) if detected else None,
        "median_quad_iou": round(
            float(np.median([r["quad_iou"] for r in quads])), 3
        ) if quads else None,
        "mean_quad_iou": round(
            sum(r["quad_iou"] for r in quads) / len(quads), 3
        ) if quads else None,
        "median_coverage_pct": round(
            float(np.median(coverages)), 1
        ) if coverages else None,
        "mean_coverage_pct": round(sum(coverages) / len(coverages), 1) if coverages else None,
        "per_image": results,
    }

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nEvaluation complete. Report written to {OUTPUT}")
    print(f"  Fallback rate: {report['fallback_rate_pct']}%")
    print(f"  Mean quad IoU: {report['mean_quad_iou']}")
    print(f"  Mean coverage: {report['mean_coverage_pct']}%")
    return report


if __name__ == "__main__":
    evaluate()