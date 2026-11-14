"""Build deterministic validation videos from real photographs.

Why composites: the project has no surveillance footage that can be committed,
and synthetic drawings are not valid detector input. These fixtures paste real
face photographs (OpenCV samples, MediaPipe assets) onto a plain canvas with
known geometry, so every frame has an exact ground-truth face rectangle.

They are NOT a substitute for evaluation on real footage; they exercise detector
scale/rotation/low-light/occlusion behavior and redaction localization with
known truth. Sources are downloaded to ``.cache/validation`` on first run.

Usage:
    python scripts/make_validation_videos.py
    python scripts/make_validation_videos.py --force
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(PROJECT_ROOT, ".cache", "validation")
SOURCES = os.path.join(CACHE, "sources")
OUT_DIR = os.path.join(CACHE, "videos")

SOURCE_URLS = {
    "portrait": "https://storage.googleapis.com/mediapipe-assets/portrait.jpg",
    "lena": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg",
}
MODEL_PATH = os.path.join(PROJECT_ROOT, "face_detection_yunet_2023mar.onnx")

W, H, FPS = 640, 480, 24


def fetch_sources() -> dict[str, np.ndarray]:
    os.makedirs(SOURCES, exist_ok=True)
    images: dict[str, np.ndarray] = {}
    for name, url in SOURCE_URLS.items():
        path = os.path.join(SOURCES, f"{name}.jpg")
        if not os.path.isfile(path):
            print(f"Downloading {url}")
            with urllib.request.urlopen(url, timeout=120) as response, open(path, "wb") as handle:
                handle.write(response.read())
        image = cv2.imread(path)
        if image is None:
            raise RuntimeError(f"could not read source image {path}")
        images[name] = image
    return images


def detect_face(image: np.ndarray) -> tuple[int, int, int, int]:
    detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (320, 320), 0.6, 0.3, 5000)
    detector.setInputSize((image.shape[1], image.shape[0]))
    _, faces = detector.detect(image)
    if faces is None or len(faces) == 0:
        raise RuntimeError("source photograph has no detectable face")
    best = max(faces, key=lambda row: float(row[2]) * float(row[3]))
    x, y, w, h = (float(v) for v in best[:4])
    return int(x), int(y), int(w), int(h)


def face_crop(image: np.ndarray, margin: float = 0.55) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Returns a square crop around the face and the face box inside the crop."""
    fx, fy, fw, fh = detect_face(image)
    cx, cy = fx + fw / 2.0, fy + fh / 2.0
    side = max(fw, fh) * (1.0 + 2.0 * margin)
    x1 = int(round(cx - side / 2))
    y1 = int(round(cy - side / 2))
    x2, y2 = x1 + int(round(side)), y1 + int(round(side))
    crop = np.zeros((int(round(side)), int(round(side)), 3), np.uint8)
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(image.shape[1], x2), min(image.shape[0], y2)
    crop[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1] = image[sy1:sy2, sx1:sx2]
    # Face rectangle relative to the (unclipped) crop origin.
    return crop, (fx - x1, fy - y1, fw, fh)


def background(index: int, label: str) -> np.ndarray:
    frame = np.full((H, W, 3), 40, np.uint8)
    # A moving vertical gradient keeps the background non-flat so we can prove
    # that pixels outside the face are preserved.
    shift = (index * 3) % W
    ramp = np.tile(np.linspace(45, 95, W, dtype=np.float32), (H, 1))
    ramp = np.roll(ramp, shift, axis=1)
    for channel in range(3):
        frame[:, :, channel] = np.clip(ramp + channel * 6, 0, 255).astype(np.uint8)
    cv2.putText(frame, label, (14, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1, cv2.LINE_AA)
    return frame


def paste(canvas: np.ndarray, crop: np.ndarray, x: int, y: int,
          size: int, alpha: float = 1.0) -> tuple[int, int, int, int]:
    resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    x2, y2 = min(W, x + size), min(H, y + size)
    x1, y1 = max(0, x), max(0, y)
    if x2 <= x1 or y2 <= y1:
        return (0, 0, 0, 0)
    patch = resized[y1 - y:y2 - y, x1 - x:x2 - x]
    canvas[y1:y2, x1:x2] = patch
    if alpha < 1.0:
        canvas[y1:y2, x1:x2] = cv2.addWeighted(
            canvas[y1:y2, x1:x2], alpha, np.zeros_like(patch), 1 - alpha, 0)
    return (x1, y1, x2 - x1, y2 - y1)


def scaled_face_box(crop_box, x: int, y: int, size: int, crop_size: int) -> tuple[int, int, int, int]:
    """Maps the source face box into canvas pixels for the pasted crop."""
    scale = size / crop_size
    fx, fy, fw, fh = crop_box
    return (
        int(round(x + fx * scale)), int(round(y + fy * scale)),
        max(1, int(round(fw * scale))), max(1, int(round(fh * scale))),
    )


def write_video(name: str, frames: list[np.ndarray], ground_truth: list[list[dict]],
                notes: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError("OpenCV VideoWriter backend unavailable")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    meta = {
        "name": name,
        # Project-relative POSIX path: the fixture set is portable across
        # machines instead of pointing at the machine that generated it.
        "path": os.path.relpath(path, PROJECT_ROOT).replace(os.sep, "/"),
        "width": W, "height": H, "fps": FPS,
        "frame_count": len(frames), "notes": notes,
        "source": "Composited from real photographs (OpenCV samples / MediaPipe assets)",
        "faces_per_frame": ground_truth,
    }
    with open(os.path.join(OUT_DIR, f"{name}.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=1)
    print(f"  {name}: {len(frames)} frames -> {path}")


def box_to_dict(box) -> dict:
    return {"x": box[0], "y": box[1], "w": box[2], "h": box[3]}


def build_single(portrait_crop, portrait_box) -> None:
    frames, gt = [], []
    for i in range(72):
        frame = background(i, "single frontal face")
        x = 60 + int(220 * (i / 71))
        y = 150 + int(10 * np.sin(i / 6))
        paste(frame, portrait_crop, x, y, 210)
        frames.append(frame)
        gt.append([box_to_dict(scaled_face_box(portrait_box, x, y, 210, portrait_crop.shape[0]))])
    write_video("frontal_single", frames, gt, "One frontal face moving left to right.")


def build_multi(lena_crop, lena_box, portrait_crop, portrait_box) -> None:
    frames, gt = [], []
    for i in range(96):
        frame = background(i, "multiple faces")
        specs = [
            (60 + 2 * i, 90, 150, lena_crop, lena_box),          # crosses the other
            (500 - 2 * i, 120, 150, portrait_crop, portrait_box),
            (250, 250 + int(8 * np.sin(i / 8)), 110, portrait_crop, portrait_box),
        ]
        boxes = []
        for x, y, size, crop, box in specs:
            paste(frame, crop, x, y, size)
            boxes.append(scaled_face_box(box, x, y, size, crop.shape[0]))
        frames.append(frame)
        gt.append([box_to_dict(b) for b in boxes])
    write_video("multi_face", frames, gt,
                "Three faces; two cross around the middle of the clip.")


def build_small(portrait_crop, portrait_box) -> None:
    frames, gt = [], []
    for i in range(60):
        frame = background(i, "small distant faces")
        boxes = []
        for k, (x, y, size) in enumerate(((90, 90, 46), (300, 120, 58), (500, 80, 40))):
            paste(frame, portrait_crop, x + i // 6, y, size)
            boxes.append(scaled_face_box(portrait_box, x + i // 6, y, size, portrait_crop.shape[0]))
        frames.append(frame)
        gt.append([box_to_dict(b) for b in boxes])
    write_video("small_faces", frames, gt, "Faces between 40 and 58 pixels wide.")


def build_occluded(lena_crop, lena_box, portrait_crop, portrait_box) -> None:
    frames, gt = [], []
    for i in range(72):
        frame = background(i, "partial occlusion")
        paste(frame, lena_crop, 120, 130, 200)
        paste(frame, portrait_crop, 400, 140, 160)
        # A moving opaque bar covers the left face for ~24 frames.
        bar_x = 60 + i * 6 if 20 <= i < 44 else None
        if bar_x is not None:
            cv2.rectangle(frame, (bar_x, 120), (bar_x + 90, 350), (120, 120, 120), -1)
        boxes = [scaled_face_box(lena_box, 120, 130, 200, lena_crop.shape[0]),
                 scaled_face_box(portrait_box, 400, 140, 160, portrait_crop.shape[0])]
        frames.append(frame)
        gt.append([box_to_dict(b) for b in boxes])
    write_video("occluded", frames, gt,
                "Two faces; a bar occludes the left face for 24 frames.")


def build_low_light(portrait_crop, portrait_box) -> None:
    frames, gt = [], []
    for i in range(60):
        frame = background(i, "low light")
        x = 100 + i
        paste(frame, portrait_crop, x, 140, 220)
        dark = (frame.astype(np.float32) * 0.32 + 8).clip(0, 255).astype(np.uint8)
        frames.append(dark)
        gt.append([box_to_dict(
            scaled_face_box(portrait_box, x, 140, 220, portrait_crop.shape[0]))])
    write_video("low_light", frames, gt,
                "Frontal face darkened to ~32% brightness.")


def build_rotated(portrait_crop, portrait_box) -> None:
    frames, gt = [], []
    rotated = cv2.warpAffine(
        portrait_crop,
        cv2.getRotationMatrix2D((portrait_crop.shape[1] / 2, portrait_crop.shape[0] / 2), 40, 1.0),
        (portrait_crop.shape[1], portrait_crop.shape[0]))
    # The rotated crop's face rectangle is only approximate; keep the original
    # box grown by the rotation envelope for a conservative ground truth.
    for i in range(48):
        frame = background(i, "rotated face")
        x = 170 + i
        y = 100
        paste(frame, rotated, x, y, 220)
        fx, fy, fw, fh = scaled_face_box(portrait_box, x, y, 220, portrait_crop.shape[0])
        cx, cy = fx + fw / 2, fy + fh / 2
        env = int((fw + fh) * 0.42)
        frames.append(frame)
        gt.append([{"x": int(cx - env / 2), "y": int(cy - env / 2), "w": env, "h": env}])
    write_video("rotated", frames, gt,
                "Face rotated ~40 degrees; ground truth is a conservative envelope.")


def build_no_faces() -> None:
    frames, gt = [], []
    for i in range(48):
        frame = background(i, "no faces")
        cv2.circle(frame, (100 + i * 8, 240), 60, (90, 140, 190), -1)
        frames.append(frame)
        gt.append([])
    write_video("no_faces", frames, gt, "Control clip with no human faces.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not os.path.isfile(MODEL_PATH):
        raise SystemExit("YuNet model missing; run python scripts/download_yunet.py first")

    if args.force and os.path.isdir(OUT_DIR):
        for name in os.listdir(OUT_DIR):
            os.remove(os.path.join(OUT_DIR, name))

    images = fetch_sources()
    lena_crop, lena_box = face_crop(images["lena"])
    portrait_crop, portrait_box = face_crop(images["portrait"])
    print(f"Fixtures -> {OUT_DIR}")
    build_single(portrait_crop, portrait_box)
    build_multi(lena_crop, lena_box, portrait_crop, portrait_box)
    build_small(portrait_crop, portrait_box)
    build_occluded(lena_crop, lena_box, portrait_crop, portrait_box)
    build_low_light(portrait_crop, portrait_box)
    build_rotated(portrait_crop, portrait_box)
    build_no_faces()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
