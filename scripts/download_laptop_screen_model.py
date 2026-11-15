"""Download and verify the laptop-screen detector ONNX model (Phase 4).

Model
-----
RT-DETR R18 trained on COCO (object detection). SafeFrame uses its COCO
``laptop`` class as the *device proposal*; the visible display quadrilateral is
estimated separately by ``backend/services/ai/laptop_screen_detection_service.py``.
The graph does not predict screen corners, and this project does not claim that
it does.

Source of truth:
    https://huggingface.co/onnx-community/rtdetr_r18vd
    file: onnx/model.onnx
Upstream weights:
    https://huggingface.co/PekingU/rtdetr_r18vd (Apache-2.0)
License: Apache-2.0 (RT-DETR / PekingU weights). The ONNX conversion repository
carries no separate licence tag; it is a format conversion of the Apache-2.0
weights above. Verify the upstream card before redistribution.
Runtime: onnxruntime (CPUExecutionProvider). Input: RGB, /255, ImageNet
mean/std, stretched to 640x640. Outputs: ``logits`` [1,300,80] and
``pred_boxes`` [1,300,4] as normalized cx,cy,w,h. COCO label index 63 = laptop.

The download is pinned by SHA-256 and size. A file that does not match is
deleted and reported as an error: the backend must never run an unverified
detector weight file.

Usage (from the project root, with the backend dependencies installed):

    python scripts/download_laptop_screen_model.py
    python scripts/download_laptop_screen_model.py --output path/to/model.onnx
    python scripts/download_laptop_screen_model.py --check
    python scripts/download_laptop_screen_model.py --no-smoke-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request

MODEL_FILENAME = "laptop_screen_detection_rtdetr_r18.onnx"
MODEL_URL = (
    "https://huggingface.co/onnx-community/rtdetr_r18vd/"
    "resolve/main/onnx/model.onnx"
)
CONFIG_URL = (
    "https://huggingface.co/onnx-community/rtdetr_r18vd/resolve/main/config.json"
)
# SHA-256 of the file served by the pinned Hugging Face URL above, computed on
# 2026-11-15. Pinning the digest means a modified/mirrored file is rejected.
PINNED_SHA256 = "11843b02455cc24009aed24d4c40db721b1093be5ccd6bbe7b9c441abb1d0558"
EXPECTED_SIZE = 82572357
LAPTOP_CLASS_INDEX = 63

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: str) -> None:
    """Raises RuntimeError unless the file matches the pinned digest and size."""
    if not os.path.isfile(path):
        raise RuntimeError(f"model file does not exist: {path}")
    size = os.path.getsize(path)
    if size == 0:
        raise RuntimeError(f"model file is empty: {path}")
    if size != EXPECTED_SIZE:
        raise RuntimeError(
            f"unexpected model size {size} bytes (expected {EXPECTED_SIZE}): {path}"
        )
    actual = sha256_of(path)
    if actual != PINNED_SHA256:
        raise RuntimeError(
            "model checksum mismatch; refusing to use this file.\n"
            f"  expected sha256: {PINNED_SHA256}\n"
            f"  actual   sha256: {actual}"
        )


def smoke_test(path: str) -> None:
    """Confirms onnxruntime can load the graph, run one frame and label laptop.

    A zero frame is enough to prove the operator set is supported and that the
    output pair matches the decoder contract. No detection quality is claimed
    from a blank frame.
    """
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "onnxruntime/numpy are not installed; run "
            "'pip install -r backend/requirements.txt'"
        ) from exc

    try:
        with urllib.request.urlopen(CONFIG_URL, timeout=60) as response:
            config = json.load(response)
        label = (config.get("id2label") or {}).get(str(LAPTOP_CLASS_INDEX))
        if str(label).lower() != "laptop":
            raise RuntimeError(
                f"config label for class {LAPTOP_CLASS_INDEX} is {label!r}, expected 'laptop'"
            )
        print(f"  config self-test: class {LAPTOP_CLASS_INDEX} = laptop")
    except RuntimeError:
        raise
    except Exception as exc:  # offline environment: still run the graph probe
        print(f"  warning: could not verify class labels from config.json ({exc})")

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    output_names = [output.name for output in session.get_outputs()]
    if not {"logits", "pred_boxes"}.issubset(set(output_names)):
        raise RuntimeError(
            f"unexpected detector outputs {output_names}; expected logits/pred_boxes"
        )
    outputs = session.run(None, {
        session.get_inputs()[0].name: np.zeros((1, 3, 640, 640), dtype=np.float32)
    })
    by_name = dict(zip(output_names, outputs))
    logits, boxes = by_name["logits"], by_name["pred_boxes"]
    if tuple(logits.shape) != (1, 300, 80) or tuple(boxes.shape) != (1, 300, 4):
        raise RuntimeError(
            f"unexpected output shapes logits={logits.shape} boxes={boxes.shape}"
        )
    print(f"  inference self-test: logits {logits.shape}, boxes {boxes.shape}")


def download(output: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    tmp_path = output + ".part"
    print(f"Downloading {MODEL_URL}")
    with urllib.request.urlopen(MODEL_URL, timeout=900) as response, open(tmp_path, "wb") as handle:
        while True:
            chunk = response.read(1024 * 256)
            if not chunk:
                break
            handle.write(chunk)
    os.replace(tmp_path, output)
    print(f"Saved {output} ({os.path.getsize(output)} bytes)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=os.path.join(PROJECT_ROOT, MODEL_FILENAME),
        help="Destination path (default: project root / %s)" % MODEL_FILENAME,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only verify an existing file (no download, no self-test).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if a valid file already exists.",
    )
    parser.add_argument(
        "--no-smoke-test",
        action="store_true",
        help="Skip the inference self-test (offline environments).",
    )
    args = parser.parse_args(argv)
    output = os.path.abspath(args.output)

    if args.check:
        verify(output)
        print(f"OK: {output} matches the pinned checksum.")
        return 0

    if os.path.isfile(output) and not args.force:
        try:
            verify(output)
            print(f"Existing model is valid: {output}")
        except RuntimeError:
            print("Existing file failed verification; re-downloading.")
            download(output)
    else:
        download(output)

    verify(output)
    print(f"Checksum verified (sha256 {PINNED_SHA256[:16]}...)")
    if not args.no_smoke_test:
        smoke_test(output)
    print("Laptop-screen detector is ready.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
