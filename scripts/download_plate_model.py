"""Download and verify the license-plate detection ONNX model.

Model: RT-DETRv2 (R18) fine-tuned on the Open Images V7 "Vehicle registration
plate" class, exported to a fixed 1x3x640x640 ONNX graph.

Source of truth:
    https://huggingface.co/Topurrra/rtdetr-license-plate-detection-onnx
License: Apache-2.0, inherited from the base model
    (PekingU/rtdetr_v2_r18vd, Apache-2.0).
Training data: Open Images V7 plate class (images CC-BY-2.0, annotations
CC-BY-4.0).

The download is pinned by SHA-256 and size. A file that does not match is
deleted and reported as an error: the backend must never run an unverified
detector weight file.

Usage (from the project root, with the backend dependencies installed):

    python scripts/download_plate_model.py
    python scripts/download_plate_model.py --output path/to/model.onnx
    python scripts/download_plate_model.py --check
    python scripts/download_plate_model.py --no-smoke-test
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request

MODEL_FILENAME = "plate_detection_rtdetr_r18.onnx"
MODEL_URL = (
    "https://huggingface.co/Topurrra/rtdetr-license-plate-detection-onnx/"
    "resolve/main/plate_rtdetr.onnx"
)
# SHA-256 of the file served by the pinned Hugging Face URL above, computed on
# 2026-11-14. Pinning the digest means a modified/mirrored file is rejected.
PINNED_SHA256 = "50f9bf9d7eaa97ade59063ba608107203d0575a84c0fb27e0ebf3bab6f594366"
EXPECTED_SIZE = 80671879

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
    """Confirms OpenCV DNN can load the graph and run one inference.

    A zero frame is enough to prove the operator set is supported and that the
    expected ``logits`` / ``pred_boxes`` outputs exist. No detection quality is
    claimed from a blank frame.
    """
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "OpenCV/numpy are not installed; run 'pip install -r backend/requirements.txt'"
        ) from exc

    net = cv2.dnn.readNetFromONNX(path)
    if net is None:
        raise RuntimeError("cv2.dnn.readNetFromONNX() returned None")

    output_names = list(net.getUnconnectedOutLayersNames())
    expected = {"logits", "pred_boxes"}
    if not expected.issubset(set(output_names)):
        raise RuntimeError(
            f"unexpected detector outputs {output_names}; expected {sorted(expected)}"
        )

    net.setInput(np.zeros((1, 3, 640, 640), dtype=np.float32))
    outputs = net.forward(output_names)
    shapes = [tuple(output.shape) for output in outputs]
    if not any(shape == (1, 300, 1) for shape in shapes) or \
            not any(shape == (1, 300, 4) for shape in shapes):
        raise RuntimeError(f"unexpected detector output shapes: {shapes}")
    print(f"  inference self-test: outputs {shapes}")


def download(output: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    tmp_path = output + ".part"
    print(f"Downloading {MODEL_URL}")
    with urllib.request.urlopen(MODEL_URL, timeout=300) as response, open(tmp_path, "wb") as handle:
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
    print("License-plate detector is ready.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
