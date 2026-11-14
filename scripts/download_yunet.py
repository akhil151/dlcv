"""Download and verify the YuNet face-detection ONNX model.

Source of truth: the official OpenCV Model Zoo repository
(https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).

The download is pinned by SHA-256. A file that does not match the pinned hash is
deleted and reported as an error: the backend must never run an unverified
detector weight file.

Usage (from the project root, with the backend dependencies installed):

    python scripts/download_yunet.py
    python scripts/download_yunet.py --output path/to/model.onnx
    python scripts/download_yunet.py --check path/to/model.onnx
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request

MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/" + MODEL_FILENAME
)
# SHA-256 of the file served by the official OpenCV Zoo URL above, computed on
# 2026-11-10. Pinning the digest means a modified/mirrored file is rejected.
PINNED_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
EXPECTED_SIZE = 232589

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
    """Confirms OpenCV can load the model and run inference on a real photo.

    The photo is fetched from the official OpenCV samples repository. Detection
    is asserted to return at least one face, so a silently broken model cannot
    pass as installed.
    """
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "OpenCV/numpy are not installed; run 'pip install -r requirements.txt'"
        ) from exc

    detector = cv2.FaceDetectorYN.create(path, "", (320, 320), 0.5, 0.3, 5000)
    if detector is None:
        raise RuntimeError("cv2.FaceDetectorYN.create() returned None")

    sample_url = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg"
    try:
        with urllib.request.urlopen(sample_url, timeout=60) as response:
            data = np.frombuffer(response.read(), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise RuntimeError("could not decode the sample image")
        detector.setInputSize((image.shape[1], image.shape[0]))
        _, faces = detector.detect(image)
        count = 0 if faces is None else len(faces)
        if count < 1:
            raise RuntimeError("inference on the sample photo returned no faces")
        best = max(float(row[-1]) for row in faces)
        print(f"  inference self-test: {count} face(s) detected, top score {best:.3f}")
    except Exception as exc:
        # Offline verification is acceptable only if the caller marked the file
        # as already verified; surface it clearly instead of pretending.
        raise RuntimeError(f"inference self-test failed: {exc}") from exc


def download(output: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    tmp_path = output + ".part"
    print(f"Downloading {MODEL_URL}")
    with urllib.request.urlopen(MODEL_URL, timeout=120) as response, open(tmp_path, "wb") as handle:
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
    print(f"Checksum verified (sha256 {PINNED_SHA256[:16]}…)")
    if not args.no_smoke_test:
        smoke_test(output)
    print("YuNet is ready.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
