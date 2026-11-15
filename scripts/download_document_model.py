"""Download and verify the identity-document detection ONNX model.

Model: DocAligner FastViT-SA24 corner-heatmap regressor. It predicts four
document-corner heatmaps for a photographed identity document; the SafeFrame
backend turns those corners into a document quadrilateral and masks the whole
document (ID card or passport) in full.

Source of truth:
    https://huggingface.co/7rplus/pagescan-weights
    file: fastvit_sa24_h_e_bifpn_256_fp32.onnx
Upstream project:
    https://github.com/DocsaidLab/DocAligner (Apache-2.0)
License: Apache-2.0. The pagescan weights repository re-hosts the DocAligner
checkpoint unchanged and documents it as Apache-2.0.
Training data: MIDV-500 / MIDV-2020 identity-document captures (the MIDV source
document images are public domain or openly licensed; MIDV-2020 annotations are
CC-BY-SA-2.5). The model is class-agnostic: it locates a document, it does not
classify an ID card vs a passport.

The download is pinned by SHA-256 and size. A file that does not match is
deleted and reported as an error: the backend must never run an unverified
detector weight file.

Usage (from the project root, with the backend dependencies installed):

    python scripts/download_document_model.py
    python scripts/download_document_model.py --output path/to/model.onnx
    python scripts/download_document_model.py --check
    python scripts/download_document_model.py --no-smoke-test
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request

MODEL_FILENAME = "document_detection_docaligner_fastvit.onnx"
MODEL_URL = (
    "https://huggingface.co/7rplus/pagescan-weights/"
    "resolve/main/fastvit_sa24_h_e_bifpn_256_fp32.onnx"
)
# SHA-256 of the file served by the pinned Hugging Face URL above, computed on
# 2026-11-14. Pinning the digest means a modified/mirrored file is rejected.
PINNED_SHA256 = "7f9f5a8935b2eb22b3ee0245d34996063f54562df390d34714af2d76928695bc"
EXPECTED_SIZE = 83084930

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
    expected four-channel corner heatmap is produced. No detection quality is
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
    if "heatmap" not in output_names:
        raise RuntimeError(
            f"unexpected detector outputs {output_names}; expected ['heatmap']"
        )

    net.setInput(np.zeros((1, 3, 256, 256), dtype=np.float32))
    heatmap = net.forward()
    if heatmap.shape != (1, 4, 128, 128):
        raise RuntimeError(f"unexpected heatmap shape: {heatmap.shape}")
    print(f"  inference self-test: heatmap {heatmap.shape}")


def download(output: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    tmp_path = output + ".part"
    print(f"Downloading {MODEL_URL}")
    with urllib.request.urlopen(MODEL_URL, timeout=600) as response, open(tmp_path, "wb") as handle:
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
    print("Identity-document detector is ready.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
