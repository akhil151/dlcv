"""Reproducible laptop-screen detection evaluation (Phase 4).

Data
----
Real photographs from Wikimedia Commons, listed in
``scripts/laptop_screen_annotations.json`` together with their licence, source
page, a manually annotated display polygon and a keyboard/deck rectangle. The
annotations were made by the implementer on 1280px-wide thumbnails
(approximate, +/- 10-20 px). They are NOT third-party ground truth and the
sample is small, so the numbers are indicative, not benchmark-grade.

What is measured (per image, on the actual production path)
-----------------------------------------------------------
* device detection: does the RT-DETR laptop proposal fire on this photo?
* screen coverage: share of the annotated display polygon that the rendered
  redaction mask actually covers. This is the privacy metric: 100% means no
  annotated display pixel survives.
* screen IoU: IoU between the refined display quad and the annotated polygon,
  reported only for detections whose region kind is ``screen_quad``.
* keyboard over-coverage: share of the annotated keyboard/deck rectangle that
  the redaction mask changes (lower is better; a device-box fallback raises it).
* fallback regions: detections that had to mask the whole device box.
* latency: end-to-end detect time and model-only inference time.

Usage (project root, backend dependencies installed, model downloaded):

    python scripts/download_laptop_screen_model.py
    python scripts/evaluate_laptop_screen_detection.py
    python scripts/evaluate_laptop_screen_detection.py --output report.json
    python scripts/evaluate_laptop_screen_detection.py --images-dir my_images
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.parse
import urllib.request

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

ANNOTATIONS = os.path.join(PROJECT_ROOT, "scripts", "laptop_screen_annotations.json")
CACHE_DIR = os.path.join(PROJECT_ROOT, ".cache", "validation", "screens")


def download_image(file_name: str, destination: str) -> bool:
    """Fetches a 1280px Wikimedia Commons thumbnail without needing an API key."""
    url = ("https://commons.wikimedia.org/wiki/Special:FilePath/"
           + urllib.parse.quote(file_name) + "?width=1280")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "SafeFrameAI/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, \
                open(destination, "wb") as handle:
            handle.write(response.read())
        return os.path.getsize(destination) > 0
    except Exception as error:  # pragma: no cover - network dependent
        print(f"  could not download {file_name}: {error}", file=sys.stderr)
        return False


def resolve_image(entry: dict, images_dir: str) -> str | None:
    path = os.path.join(images_dir, entry["key"] + ".jpg")
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path
    if images_dir == CACHE_DIR:
        if not download_image(entry["file"], path):
            return None
        return path if os.path.isfile(path) else None
    return None


def polygon_mask(shape: tuple[int, int], points: list[list[float]]) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.round(np.asarray(points, np.float32)).astype(np.int32)], 1)
    return mask


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.logical_or(first, second).sum()
    if not union:
        return 0.0
    return float(np.logical_and(first, second).sum()) / float(union)


def redaction_mask(frame: np.ndarray, detections: list, blur_service) -> np.ndarray:
    """Renders the real redaction mask by blacking out a copy of the frame."""
    if not detections:
        return np.zeros(frame.shape[:2], np.uint8)
    black = np.zeros_like(frame)
    _, report = blur_service.render_blur(
        frame.copy(), [], [], blur_style="blackout", blur_intensity=100,
        frame_index=1, detections=detections)
    rendered = np.zeros_like(frame)
    for plan in report.roi_plans:
        mask = blur_service._alpha_mask(plan, soft=False, power=1.0)
        rendered[plan.y1:plan.y2, plan.x1:plan.x2] = (
            mask[:, :, None] * 255).astype(np.uint8)
    del black
    return (rendered[:, :, 0] > 127).astype(np.uint8)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", default=CACHE_DIR,
                        help="Directory holding <key>.jpg images (default: cache).")
    parser.add_argument("--output", default=os.path.join(CACHE_DIR, "evaluation.json"),
                        help="Where to write the JSON report.")
    args = parser.parse_args(argv)

    from backend.services.ai.blur_service import BlurService
    from backend.services.ai.laptop_screen_detection_service import (
        REGION_SCREEN_QUAD,
        LaptopScreenDetectionService,
    )
    from backend.services.ai.model_manager import model_manager
    from backend.services.ai.detection_service import DetectionService

    weights = model_manager.resolve_laptop_path()
    if not os.path.isfile(weights):
        print("ERROR: laptop-screen weights missing; run "
              "python scripts/download_laptop_screen_model.py", file=sys.stderr)
        return 1

    with open(ANNOTATIONS, encoding="utf-8") as handle:
        annotations = json.load(handle)
    reference_width = float(annotations.get("reference_width", 1280))

    service = LaptopScreenDetectionService()
    blur = BlurService()
    ranker = DetectionService()

    rows = []
    model_ms: list[float] = []
    detect_ms: list[float] = []
    coverages: list[float] = []
    keyboard_changes: list[float] = []
    quad_ious: list[float] = []
    detected_with_screen = 0
    screen_images = 0
    total_quad_regions = 0
    total_fallback_regions = 0
    negative_images = 0
    negative_detections = 0

    for entry in annotations["images"]:
        path = resolve_image(entry, args.images_dir)
        if path is None:
            print(f"SKIP {entry['key']}: image unavailable")
            continue
        frame = cv2.imread(path)
        if frame is None:
            print(f"SKIP {entry['key']}: unreadable image")
            continue
        scale = frame.shape[1] / reference_width
        started = time.perf_counter()
        results = service.detect(frame)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        detect_ms.append(elapsed_ms)
        model_ms.append(service.last_inference_ms)

        detections = [
            ranker._detect_laptop_screens(frame, 1)[index]
            for index in range(len(results))
        ]
        mask = redaction_mask(frame, detections, blur)
        total_quad_regions += sum(1 for item in results
                                  if item.region_kind == REGION_SCREEN_QUAD)
        total_fallback_regions += sum(1 for item in results
                                      if item.region_kind != REGION_SCREEN_QUAD)

        row = {
            "key": entry["key"],
            "detections": len(results),
            "region_kinds": [item.region_kind for item in results],
            "model_ms": round(service.last_inference_ms, 1),
            "total_ms": round(elapsed_ms, 1),
        }

        if not entry.get("screen_visible"):
            negative_images += 1
            negative_detections += len(results)
            rows.append(row)
            continue

        screen_images += 1
        screen = polygon_mask(frame.shape[:2], [
            [x * scale, y * scale] for x, y in entry["screen_quad"]])
        screen_pixels = int(screen.sum())
        covered = int(np.logical_and(mask, screen).sum())
        coverage = covered / screen_pixels if screen_pixels else 0.0
        coverages.append(coverage)
        row["screen_coverage_pct"] = round(coverage * 100.0, 1)

        keyboard = polygon_mask(frame.shape[:2], [
            [entry["keyboard_rect"][0] * scale, entry["keyboard_rect"][1] * scale],
            [entry["keyboard_rect"][2] * scale, entry["keyboard_rect"][1] * scale],
            [entry["keyboard_rect"][2] * scale, entry["keyboard_rect"][3] * scale],
            [entry["keyboard_rect"][0] * scale, entry["keyboard_rect"][3] * scale],
        ])
        keyboard_pixels = int(keyboard.sum())
        if keyboard_pixels:
            keyboard_change = float(np.logical_and(mask, keyboard).sum()) / keyboard_pixels
            keyboard_changes.append(keyboard_change)
            row["keyboard_changed_pct"] = round(keyboard_change * 100.0, 1)

        if coverage >= 0.25:
            detected_with_screen += 1
        best_iou = 0.0
        for item, detection in zip(results, detections):
            if item.region_kind != REGION_SCREEN_QUAD:
                continue
            predicted = polygon_mask(frame.shape[:2], item.quad)
            best_iou = max(best_iou, mask_iou(predicted, screen))
        if best_iou:
            quad_ious.append(best_iou)
            row["screen_quad_iou"] = round(best_iou, 3)
        rows.append(row)
        print(f"{entry['key']:<18} detections={len(results)} coverage="
              f"{coverage * 100:5.1f}% keyboard="
              f"{row.get('keyboard_changed_pct', 'n/a')}% kinds={row['region_kinds']}")

    report = {
        "dataset": ANNOTATIONS,
        "annotation_method": annotations["description"],
        "images_evaluated": len(rows),
        "images_with_visible_screen": screen_images,
        "images_with_screen_detected": detected_with_screen,
        "device_detection_recall_pct": (round(detected_with_screen / screen_images * 100.0, 1)
                                        if screen_images else None),
        "negative_images": negative_images,
        "negative_images_with_any_detection": negative_detections,
        "screen_quad_regions": total_quad_regions,
        "device_fallback_regions": total_fallback_regions,
        "screen_coverage_median_pct": (round(statistics.median(coverages) * 100.0, 1)
                                       if coverages else None),
        "screen_coverage_min_pct": (round(min(coverages) * 100.0, 1)
                                    if coverages else None),
        "screen_coverage_mean_pct": (round(statistics.mean(coverages) * 100.0, 1)
                                     if coverages else None),
        "keyboard_changed_median_pct": (round(statistics.median(keyboard_changes) * 100.0, 1)
                                        if keyboard_changes else None),
        "screen_quad_iou_median": (round(statistics.median(quad_ious), 3)
                                   if quad_ious else None),
        "model_inference_ms_median": (round(statistics.median(model_ms), 1)
                                      if model_ms else None),
        "end_to_end_detect_ms_median": (round(statistics.median(detect_ms), 1)
                                        if detect_ms else None),
        "device": "CPUExecutionProvider (onnxruntime), no GPU execution path active",
        "per_image": rows,
        "notes": [
            "Coverage is measured against manually annotated display polygons, not "
            "third-party ground truth.",
            "A device fallback deliberately over-covers: it raises keyboard "
            "over-coverage but never leaves annotated display content visible.",
            "Screen-quad IoU is only defined for refined quads; fallback regions "
            "have no display boundary to score.",
            "No precision/recall is claimed for screen boundaries beyond the "
            "coverage figures above.",
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)

    print("\n=== Summary ===")
    for key in ("images_evaluated", "device_detection_recall_pct",
                "screen_coverage_median_pct", "screen_coverage_min_pct",
                "keyboard_changed_median_pct", "screen_quad_iou_median",
                "screen_quad_regions", "device_fallback_regions",
                "model_inference_ms_median", "end_to_end_detect_ms_median"):
        print(f"{key}: {report[key]}")
    print(f"\nReport: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
