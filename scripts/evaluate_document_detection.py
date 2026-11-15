"""Reproducible identity-document detection evaluation.

Dataset: ``Noaman/midv500`` on Hugging Face (MIDV-500 frames and binary
document masks, MIT-tagged). Images and masks are fetched from the Hugging
Face datasets server; nothing is bundled with the repository.

The detector is the pinned DocAligner FastViT-SA24 corner-heatmap graph the
backend uses. A prediction is a true positive when the IoU between its
quadrilateral mask and the ground-truth document mask is >= 0.50. Metrics are
computed only from the frames actually evaluated: nothing is estimated.

The graph is class-agnostic (it finds a document, not its type), so this
script measures document-localisation quality, not ID-card/passport
classification accuracy.

Usage (project root, backend dependencies installed, model downloaded):

    python scripts/evaluate_document_detection.py
    python scripts/evaluate_document_detection.py --count 60 --output report.json
    python scripts/evaluate_document_detection.py \
        --negatives .cache/validation/plates
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
CACHE_ROOT = os.path.join(PROJECT_ROOT, ".cache", "validation", "documents")
DATASET = "Noaman/midv500"
SPLIT = "validation"
ROWS_URL = (
    "https://datasets-server.huggingface.co/rows?"
    f"dataset={DATASET.replace('/', '%2F')}&config=default&split={SPLIT}"
    "&offset={offset}&length={length}"
)
MODEL_NAME = "DocAligner FastViT-SA24 (corner heatmap, 256 input)"
MODEL_URL = (
    "https://huggingface.co/7rplus/pagescan-weights/resolve/main/"
    "fastvit_sa24_h_e_bifpn_256_fp32.onnx"
)
MODEL_SHA256 = "7f9f5a8935b2eb22b3ee0245d34996063f54562df390d34714af2d76928695bc"
MODEL_LICENSE = "Apache-2.0 (DocsaidLab DocAligner; weights mirrored by 7rplus/pagescan-weights)"
ATTRIBUTION = (
    "Dataset: Noaman/midv500 (MIDV-500 frames with document masks, MIT tag). "
    "Detector: DocAligner FastViT-SA24, Apache-2.0."
)


def _detail(message: str) -> None:
    print(f"  {message}")


def _download(url: str, path: str) -> None:
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            urllib.request.urlretrieve(url, path)
            return
        except Exception as error:  # transient asset-server failure
            last_error = error
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"could not download {url}: {last_error}")


def fetch_rows(count: int) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while len(rows) < count:
        length = min(20, count - len(rows))
        url = ROWS_URL.format(offset=offset, length=length)
        payload = None
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=90) as response:
                    payload = json.load(response)
                break
            except Exception as error:
                last_error = error
                time.sleep(2 * (attempt + 1))
        if payload is None:
            raise RuntimeError(f"could not fetch validation rows: {last_error}")
        batch = payload.get("rows", [])
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
    return rows


def prepare_dataset(count: int) -> list[dict]:
    """Returns ``{'image': path, 'mask': path}`` items for the evaluated frames."""
    root = CACHE_ROOT
    os.makedirs(root, exist_ok=True)

    cached = []
    for index in range(count):
        image = os.path.join(root, f"{index:03d}_img.jpg")
        mask = os.path.join(root, f"{index:03d}_mask.jpg")
        if not (os.path.isfile(image) and os.path.isfile(mask)):
            cached = []
            break
        cached.append({"image": image, "mask": mask})
    if cached:
        print(f"Using {len(cached)} cached MIDV validation frames from {root}")
        return cached

    print(f"Fetching {count} annotated MIDV frames from Hugging Face...")
    rows = fetch_rows(count)
    items = []
    for index, item in enumerate(rows):
        row = item["row"]
        image = os.path.join(root, f"{index:03d}_img.jpg")
        mask = os.path.join(root, f"{index:03d}_mask.jpg")
        _download(row["pixel_values"]["src"], image)
        _download(row["label"]["src"], mask)
        items.append({"image": image, "mask": mask})
        if (index + 1) % 10 == 0:
            _detail(f"downloaded {index + 1}/{len(rows)}")
    with open(os.path.join(root, "annotations.json"), "w", encoding="utf-8") as handle:
        json.dump(items, handle, indent=1)
    return items


def mask_iou(quad: list[tuple[float, float]], mask: np.ndarray,
             shape: tuple[int, int]) -> float:
    height, width = shape
    prediction = np.zeros((height, width), np.uint8)
    points = np.round(np.asarray(quad, dtype=np.float32)).astype(np.int32)
    cv2.fillPoly(prediction, [points], 1)
    predicted = prediction > 0
    intersection = np.logical_and(predicted, mask).sum()
    union = np.logical_or(predicted, mask).sum()
    return float(intersection) / float(union) if union else 0.0


def evaluate_negatives(path: str) -> dict:
    """Counts non-document images that still produce a document detection."""
    if not os.path.isdir(path):
        return {"directory": path, "error": "directory does not exist"}
    from backend.services.ai.document_detection_service import (
        document_detection_service,
    )

    images = 0
    flagged = 0
    detections = 0
    flagged_files = []
    for name in sorted(os.listdir(path)):
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        frame = cv2.imread(os.path.join(path, name))
        if frame is None:
            continue
        images += 1
        found = document_detection_service.detect(frame)
        if found:
            flagged += 1
            detections += len(found)
            flagged_files.append({
                "file": name,
                "detections": [
                    {"confidence": item.confidence,
                     "bbox": item.bbox.model_dump()}
                    for item in found
                ],
            })
    return {
        "directory": os.path.abspath(path),
        "images": images,
        "images_with_detections": flagged,
        "detections": detections,
        "flagged_files": flagged_files,
        "note": ("These vehicle photos come from the license-plate evaluation "
                 "cache and contain no identity documents. Any detection is a "
                 "false positive; for a redaction tool it fails closed "
                 "(over-redaction, never under-redaction)."),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=60,
                        help="Number of annotated validation frames to evaluate.")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="Mask-IoU threshold for a true positive.")
    parser.add_argument("--output",
                        default=os.path.join(CACHE_ROOT, "evaluation.json"),
                        help="Where to write the JSON report.")
    parser.add_argument("--negatives", default=None,
                        help="Optional directory of non-document photos used to "
                             "count false positives.")
    parser.add_argument("--save-predictions", action="store_true",
                        help="Also store every predicted quad in the report.")
    args = parser.parse_args(argv)

    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from backend.services.ai.document_detection_service import (
        document_detection_service,
    )
    from backend.services.ai.model_manager import model_manager

    if not os.path.isfile(model_manager.resolve_document_path()):
        print("ERROR: identity-document weights missing; run "
              "python scripts/download_document_model.py", file=sys.stderr)
        return 1

    items = prepare_dataset(args.count)
    if not items:
        print("ERROR: no validation frames could be prepared", file=sys.stderr)
        return 1

    true_positives = false_negatives = extra_regions = 0
    all_ious: list[float] = []
    per_image: list[dict] = []
    inference_ms: list[float] = []
    unreadable = 0

    for item in items:
        frame = cv2.imread(item["image"])
        mask_image = cv2.imread(item["mask"], cv2.IMREAD_GRAYSCALE)
        if frame is None or mask_image is None:
            unreadable += 1
            continue
        mask = mask_image > 127
        started = time.perf_counter()
        detections = document_detection_service.detect(frame)
        inference_ms.append((time.perf_counter() - started) * 1000.0)

        shape = frame.shape[:2]
        scored = [(mask_iou(detection.quad, mask, shape), detection)
                  for detection in detections]
        scored.sort(key=lambda entry: -entry[0])
        best_iou = scored[0][0] if scored else 0.0
        matched = best_iou >= args.iou
        if matched:
            true_positives += 1
            all_ious.append(best_iou)
            extra_regions += len(scored) - 1
        else:
            false_negatives += 1
            extra_regions += len(scored)

        record = {
            "file": os.path.basename(item["image"]),
            "predicted": len(detections),
            "best_mask_iou": round(best_iou, 4),
            "matched": matched,
        }
        if args.save_predictions:
            record["predictions"] = [
                {"confidence": detection.confidence, "quad": detection.quad,
                 "bbox": detection.bbox.model_dump(), "mask_iou": round(value, 4)}
                for value, detection in scored
            ]
        per_image.append(record)

    if unreadable:
        print(f"WARNING: {unreadable} cached frame(s) could not be read", file=sys.stderr)
    if not per_image:
        print("ERROR: no validation frame could be read", file=sys.stderr)
        return 1

    recall = true_positives / max(1, true_positives + false_negatives)
    report = {
        "attribution": ATTRIBUTION,
        "model": MODEL_NAME,
        "model_url": MODEL_URL,
        "model_sha256": MODEL_SHA256,
        "model_license": MODEL_LICENSE,
        "confidence_threshold": document_detection_service.conf_threshold,
        "heatmap_threshold": document_detection_service.heatmap_threshold,
        "iou_threshold": args.iou,
        "images_evaluated": len(per_image),
        "ground_truth_documents": true_positives + false_negatives,
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "extra_predicted_regions": extra_regions,
        "recall": round(recall, 4),
        "mean_mask_iou": round(float(np.mean(all_ious)), 4) if all_ious else 0.0,
        "mean_inference_ms": round(float(np.mean(inference_ms)), 1),
        "note": ("The graph is class-agnostic: it localises documents but does "
                 "not distinguish an ID card from a passport."),
        "per_image": per_image,
    }
    if args.negatives:
        report["negatives"] = evaluate_negatives(args.negatives)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)

    print(f"Model: {MODEL_NAME} ({MODEL_LICENSE})")
    print(f"Frames evaluated: {len(per_image)}")
    print(f"Recall (mask IoU >= {args.iou}): "
          f"{true_positives}/{true_positives + false_negatives} = {recall:.1%}")
    if all_ious:
        print(f"Mean matched mask IoU: {float(np.mean(all_ious)):.4f}")
    print(f"Extra predicted regions: {extra_regions}")
    print(f"Mean inference time: {float(np.mean(inference_ms)):.1f} ms/frame")
    if "negatives" in report:
        negatives = report["negatives"]
        if "error" in negatives:
            print(f"WARNING: negatives not evaluated: {negatives['error']}",
                  file=sys.stderr)
        else:
            print(f"Negatives: {negatives['images_with_detections']}/"
                  f"{negatives['images']} non-document images produced a detection")
    print(f"Wrote {os.path.abspath(args.output)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
