"""Reproducible license-plate detection evaluation.

Dataset: ``keremberke/license-plate-object-detection`` on Hugging Face
("Vehicle Registration Plates Dataset", Augmented Startups / Roboflow
Universe), licensed CC BY 4.0. Images and ground-truth plate boxes are fetched
through the Hugging Face datasets server; nothing is bundled with the repo.

The detector is the same pinned RT-DETR graph the backend uses. Matching is
greedy by score at IoU >= 0.50. The script prints and stores real precision,
recall, mean IoU and small-plate recall for the evaluated sample. It never
estimates or extrapolates: the JSON records exactly how many images and boxes
were used.

Usage (project root, backend dependencies installed, model downloaded):

    python scripts/evaluate_plate_detection.py
    python scripts/evaluate_plate_detection.py --count 60 --output report.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_ROOT = os.path.join(PROJECT_ROOT, ".cache", "validation", "plates")
DATASET = "keremberke/license-plate-object-detection"
SPLIT = "validation"
ROWS_URL = (
    "https://datasets-server.huggingface.co/rows?"
    f"dataset=keremberke%2Flicense-plate-object-detection&config=full&split={SPLIT}"
    "&offset={offset}&length={length}"
)
ATTRIBUTION = (
    "Dataset: keremberke/license-plate-object-detection (CC BY 4.0), "
    "from Roboflow Universe / Augmented Startups."
)


def _detail(message: str) -> None:
    print(f"  {message}")


def fetch_rows(count: int) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while len(rows) < count:
        length = min(20, count - len(rows))
        url = ROWS_URL.format(offset=offset, length=length)
        payload = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=90) as response:
                    payload = json.load(response)
                break
            except Exception as error:  # transient dataset-server 5xx
                if attempt == 3:
                    raise RuntimeError(f"could not fetch validation rows: {error}") from error
                time.sleep(2 * (attempt + 1))
        batch = payload.get("rows", [])
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
    return rows


def prepare_dataset(count: int) -> list[dict]:
    os.makedirs(CACHE_ROOT, exist_ok=True)
    annotations_path = os.path.join(CACHE_ROOT, "annotations.json")
    images: list[dict] = []
    if os.path.isfile(annotations_path):
        with open(annotations_path, encoding="utf-8") as handle:
            images = json.load(handle)
        # Re-point stale relative paths (the cache may have been produced from a
        # different working directory) at the files next to this manifest.
        for item in images:
            resolved = os.path.join(CACHE_ROOT, os.path.basename(item["file"]))
            if not os.path.isfile(item["file"]) and os.path.isfile(resolved):
                item["file"] = resolved
    if len(images) >= count:
        return images[:count]

    print(f"Fetching {count} annotated images from Hugging Face...")
    rows = fetch_rows(count)
    images = []
    for index, item in enumerate(rows):
        row = item["row"]
        path = os.path.join(CACHE_ROOT, f"{index:03d}.jpg")
        if not os.path.isfile(path):
            urllib.request.urlretrieve(row["image"]["src"], path)
        images.append({
            "file": path,
            "width": int(row["width"]),
            "height": int(row["height"]),
            "boxes": [[float(value) for value in box] for box in row["objects"]["bbox"]],
        })
    with open(annotations_path, "w", encoding="utf-8") as handle:
        json.dump(images, handle, indent=1)
    return images


def iou(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    intersection = ix * iy
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def match(predictions: list[tuple[float, list[float]]], ground_truth: list[list[float]],
          iou_threshold: float = 0.5) -> tuple[int, int, int, list[float]]:
    used: set[int] = set()
    true_positives = 0
    matched_ious: list[float] = []
    for _, box in sorted(predictions, key=lambda item: -item[0]):
        best, best_index = -1.0, -1
        for index, target in enumerate(ground_truth):
            if index in used:
                continue
            value = iou(box, target)
            if value > best:
                best, best_index = value, index
        if best >= iou_threshold:
            used.add(best_index)
            true_positives += 1
            matched_ious.append(best)
    return (true_positives, len(predictions) - true_positives,
            len(ground_truth) - true_positives, matched_ious)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=40,
                        help="Number of annotated validation images to evaluate.")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="IoU threshold for a true positive.")
    parser.add_argument("--small-width", type=float, default=40.0,
                        help="Ground-truth plates narrower than this are 'small/distant'.")
    parser.add_argument("--output",
                        default=os.path.join(CACHE_ROOT, "evaluation.json"),
                        help="Where to write the JSON report.")
    parser.add_argument("--save-predictions", action="store_true",
                        help="Also store every predicted box in the JSON report.")
    args = parser.parse_args(argv)

    # The script is launched as ``python scripts/...py``; make the project root
    # importable for the backend package.
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from backend.services.ai.model_manager import model_manager
    from backend.services.ai.detection_service import detection_service

    if not os.path.isfile(model_manager.resolve_plate_path()):
        print("ERROR: license-plate weights missing; run "
              "python scripts/download_plate_model.py", file=sys.stderr)
        return 1
    detection_service.validate_available(["license_plates"])

    images = prepare_dataset(args.count)
    if not images:
        print("ERROR: no validation images could be fetched", file=sys.stderr)
        return 1

    true_positives = false_positives = false_negatives = 0
    all_ious: list[float] = []
    small_tp = small_total = 0
    per_image: list[dict] = []
    inference_ms: list[float] = []

    unreadable = 0
    for item in images:
        frame = cv2.imread(item["file"])
        if frame is None:
            unreadable += 1
            continue
        result = detection_service.detect_entities(frame, ["license_plates"], 0)
        inference_ms.append(result.inference_time_ms)
        height, width = frame.shape[:2]
        predictions = [
            (detection.confidence,
             [detection.bbox.x / 100.0 * width, detection.bbox.y / 100.0 * height,
              detection.bbox.width / 100.0 * width,
              detection.bbox.height / 100.0 * height])
            for detection in result.detections
        ]
        tp, fp, fn, ious = match(predictions, item["boxes"], args.iou)
        true_positives += tp
        false_positives += fp
        false_negatives += fn
        all_ious.extend(ious)

        small = [box for box in item["boxes"] if box[2] < args.small_width]
        if small:
            small_total += len(small)
            small_tp += match(predictions, small, args.iou)[0]

        record = {
            "file": os.path.basename(item["file"]),
            "ground_truth": len(item["boxes"]),
            "predicted": len(predictions),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }
        if args.save_predictions:
            record["predictions"] = [
                {"score": round(score, 4),
                 "bbox": [round(value, 2) for value in box]}
                for score, box in predictions
            ]
        per_image.append(record)

    if unreadable:
        print(f"WARNING: {unreadable} cached image(s) could not be read", file=sys.stderr)
    if not per_image:
        print("ERROR: no validation image could be read", file=sys.stderr)
        return 1

    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    report = {
        "attribution": ATTRIBUTION,
        "model": "RT-DETRv2 (R18), Open Images plate class, Apache-2.0",
        "confidence_threshold": detection_service.plate_conf_threshold,
        "iou_threshold": args.iou,
        "images_evaluated": len(per_image),
        "ground_truth_boxes": true_positives + false_negatives,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "mean_iou_of_true_positives": round(float(np.mean(all_ious)), 4) if all_ious else None,
        "small_plate_recall": (round(small_tp / small_total, 4) if small_total else None),
        "small_plate_ground_truth": small_total,
        "mean_inference_ms": round(float(np.mean(inference_ms)), 1) if inference_ms else None,
        "per_image": per_image,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)

    print(ATTRIBUTION)
    print(f"Images: {report['images_evaluated']}  GT boxes: {report['ground_truth_boxes']}")
    print(f"TP={true_positives} FP={false_positives} FN={false_negatives}")
    print(f"Precision={precision:.3f}  Recall={recall:.3f}  "
          f"mean IoU={report['mean_iou_of_true_positives']}")
    print(f"Small/distant plate recall (<{args.small_width:.0f}px GT width): "
          f"{report['small_plate_recall']} over {small_total} boxes")
    print(f"Mean inference: {report['mean_inference_ms']} ms/image (CPU)")
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
