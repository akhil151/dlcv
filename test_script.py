"""End-to-end HTTP smoke test for a running SafeFrame AI backend.

This drives the real public API exactly as the frontend does:

    liveness -> readiness -> upload -> start -> poll -> summary -> download
    -> decode verification

It also exercises the negative paths that must stay negative:

    * an unsupported target category is rejected (never silently accepted)
    * a non-video upload is rejected at upload time
    * the protected download stays gated while the task is incomplete
    * an unknown task id returns 404

Prerequisites (both terminals from the project root):
    uvicorn backend.main:app --host 127.0.0.1 --port 8000
    python test_script.py

Exit code 0 means every assertion passed. Any failure prints the stage and the
server's own error message, then exits non-zero.
"""
from __future__ import annotations

import io
import os
import sys
import time

import cv2
import requests

BASE_URL = os.environ.get("SAFEFRAME_API", "http://127.0.0.1:8000/api/v1")
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FIXTURE_DIR = os.path.join(PROJECT_ROOT, ".cache", "validation", "videos")

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"   [{status}] {message}")
    if not condition:
        FAILURES.append(message)


def pick_fixture() -> str:
    for name in ("frontal_single.mp4", "multi_face.mp4", "low_light.mp4"):
        path = os.path.join(FIXTURE_DIR, name)
        if os.path.isfile(path):
            return path
    raise SystemExit(
        "No validation fixture found. Generate them first:\n"
        "    python scripts/make_validation_videos.py"
    )


def main() -> int:
    fixture = pick_fixture()
    print(f"SafeFrame AI end-to-end smoke test -> {BASE_URL}")
    print(f"Fixture: {fixture}\n")

    # 1. Liveness
    print("1. Liveness /api/v1/health")
    res = requests.get(f"{BASE_URL}/health", timeout=10)
    check(res.status_code == 200, f"health returned 200 (got {res.status_code})")

    # 2. Readiness (real detector + ffmpeg probe)
    print("2. Readiness /api/v1/health/readiness")
    res = requests.get(f"{BASE_URL}/health/readiness", timeout=30)
    readiness = res.json() if res.headers.get("content-type", "").startswith("application/json") else {}
    check(res.status_code == 200, f"readiness returned 200 (got {res.status_code})")
    check(bool(readiness.get("processing_ready")), f"processing_ready is true ({readiness.get('reasons')})")
    check(bool(readiness.get("detector", {}).get("available")), "YuNet detector is available")
    check(bool(readiness.get("ffmpeg", {}).get("available")), "ffmpeg is available")
    plate_ready = bool(readiness.get("plate_detector", {}).get("available"))
    documents_ready = bool(readiness.get("documents_ready"))
    laptop_ready = bool(readiness.get("laptop_screens_ready"))
    phone_ready = bool(readiness.get("phone_screens_ready"))
    print(f"        license-plate detector available: {plate_ready}")
    print(f"        identity-document detector available: {documents_ready}")
    print(f"        laptop-screen detector available: {laptop_ready}")
    print(f"        phone-screen detector available: {phone_ready}")

    # 3a. Negative: an unvalidated category must be refused, not ignored.
    # Laptop and phone screens are validated; smartwatches are not.
    print("3a. Unsupported category is rejected")
    res = requests.post(f"{BASE_URL}/videos/upload",
                        files={"file": ("probe.mp4", open(fixture, "rb"), "video/mp4")}, timeout=120)
    check(res.status_code == 200, f"probe upload returned 200 (got {res.status_code})")
    probe_video_id = res.json().get("video_id", "")
    res = requests.post(f"{BASE_URL}/redact/start", json={
        "video_id": probe_video_id,
        "target_categories": ["faces", "smartwatches"],
        "blur_style": "gaussian",
        "blur_intensity": 80,
    }, timeout=60)
    check(res.status_code >= 400, f"smartwatches request was refused (got {res.status_code})")
    print(f"        server said: {res.json().get('message')}")

    # 3b. License plates are accepted, or refused with a clear missing-model
    # error. They must never be silently dropped from the job.
    print("3b. License-plate category is accepted or explicitly unavailable")
    res = requests.post(f"{BASE_URL}/redact/start", json={
        "video_id": probe_video_id,
        "target_categories": ["license_plates"],
        "blur_style": "gaussian",
        "blur_intensity": 80,
    }, timeout=60)
    if plate_ready:
        check(res.status_code == 200, f"license_plates start returned 200 (got {res.status_code})")
        if res.status_code == 200:
            plate_task_id = res.json().get("task_id", "")
            requests.post(f"{BASE_URL}/redact/cancel/{plate_task_id}", timeout=30)
    else:
        check(res.status_code == 503,
              f"license_plates refused with 503 when weights are missing (got {res.status_code})")
        print(f"        server said: {res.json().get('message')}")

    # 3b2. Laptop screens are accepted, or refused with a clear missing-model
    # error. They must never be silently dropped from the job.
    print("3b2. Laptop-screen category is accepted or explicitly unavailable")
    res = requests.post(f"{BASE_URL}/redact/start", json={
        "video_id": probe_video_id,
        "target_categories": ["laptop_screens"],
        "blur_style": "gaussian",
        "blur_intensity": 80,
    }, timeout=60)
    if laptop_ready:
        check(res.status_code == 200, f"laptop_screens start returned 200 (got {res.status_code})")
        if res.status_code == 200:
            laptop_task_id = res.json().get("task_id", "")
            requests.post(f"{BASE_URL}/redact/cancel/{laptop_task_id}", timeout=30)
    else:
        check(res.status_code == 503,
              f"laptop_screens refused with 503 when weights are missing (got {res.status_code})")
        print(f"        server said: {res.json().get('message')}")

    # 3c. ID cards / passports are accepted, or refused with a clear
    # missing-model error. The document detector is class-agnostic, so both
    # categories run the same validated graph.
    print("3c. Identity-document categories are accepted or explicitly unavailable")
    for category in ("id_cards", "passports"):
        res = requests.post(f"{BASE_URL}/redact/start", json={
            "video_id": probe_video_id,
            "target_categories": [category],
            "blur_style": "gaussian",
            "blur_intensity": 80,
        }, timeout=60)
        if documents_ready:
            check(res.status_code == 200,
                  f"{category} start returned 200 (got {res.status_code})")
            if res.status_code == 200:
                doc_task_id = res.json().get("task_id", "")
                requests.post(f"{BASE_URL}/redact/cancel/{doc_task_id}", timeout=30)
        else:
            check(res.status_code == 503,
                  f"{category} refused with 503 when weights are missing "
                  f"(got {res.status_code})")
            print(f"        server said: {res.json().get('message')}")

    # 4. Negative: a non-video upload is rejected.
    print("4. Corrupt upload is rejected")
    res = requests.post(f"{BASE_URL}/videos/upload",
                        files={"file": ("not_a_video.mp4", io.BytesIO(b"this is not a video"), "video/mp4")},
                        timeout=60)
    check(res.status_code == 400, f"corrupt upload returned 400 (got {res.status_code})")

    # 5. Real upload
    print("5. Upload a real video")
    with open(fixture, "rb") as handle:
        res = requests.post(f"{BASE_URL}/videos/upload",
                            files={"file": (os.path.basename(fixture), handle, "video/mp4")}, timeout=300)
    check(res.status_code == 200, f"upload returned 200 (got {res.status_code})")
    if res.status_code != 200:
        print(f"        server said: {res.text[:400]}")
        return 1
    meta = res.json()
    video_id = meta["video_id"]
    print(f"        {meta['filename']} {meta['resolution']} {meta['fps']} fps "
          f"{meta['frame_count']} frames")
    check(meta["frame_count"] > 0, "upload reported a real frame count")
    check(meta["resolution"] not in ("", "unknown"), "upload reported a real resolution")

    # 6. Start redaction (faces only)
    print("6. Start face redaction")
    res = requests.post(f"{BASE_URL}/redact/start", json={
        "video_id": video_id,
        "target_categories": ["faces"],
        "blur_style": "gaussian",
        "blur_intensity": 80,
        "debug_mode": True,
    }, timeout=60)
    check(res.status_code == 200, f"start returned 200 (got {res.status_code})")
    if res.status_code != 200:
        print(f"        server said: {res.text[:400]}")
        return 1
    task_id = res.json()["task_id"]
    print(f"        task_id = {task_id}")

    # 7. Download must stay gated while the task runs.
    print("7. Download gate before completion")
    res = requests.get(f"{BASE_URL}/videos/download/{task_id}", timeout=30)
    check(res.status_code == 409, f"download refused with 409 while running (got {res.status_code})")

    # 8. Poll real progress
    print("8. Poll task status")
    status_data: dict = {}
    progress_seen: list[float] = []
    deadline = time.time() + 300
    while time.time() < deadline:
        res = requests.get(f"{BASE_URL}/redact/status/{task_id}", timeout=30)
        if res.status_code != 200:
            check(False, f"status returned {res.status_code}: {res.text[:200]}")
            return 1
        status_data = res.json()
        progress_seen.append(float(status_data["progress_pct"]))
        if status_data["status"] in ("COMPLETED", "FAILED", "CANCELLED"):
            break
        time.sleep(0.5)
    print(f"        status={status_data['status']} progress={status_data['progress_pct']}% "
          f"frames={status_data['metrics']['processed_frames']}/{status_data['metrics']['total_frames']}")
    check(status_data.get("status") == "COMPLETED", f"task completed ({status_data.get('status')})")
    check(bool(status_data.get("output_verified")), "task reports output_verified = true")
    check(status_data["metrics"]["processed_frames"] == status_data["metrics"]["total_frames"],
          "every frame was processed")
    check(any(0 < value < 100 for value in progress_seen), "real intermediate progress was observed")
    check(status_data["metrics"]["masks_total"] > 0, "face regions were actually masked")

    if status_data.get("status") != "COMPLETED":
        print(f"        failure_reason: {status_data.get('failure_reason')}")
        return 1

    # 9. Summary
    print("9. Task summary")
    res = requests.get(f"{BASE_URL}/redact/results/{task_id}", timeout=30)
    check(res.status_code == 200, f"summary returned 200 (got {res.status_code})")
    summary = res.json() if res.status_code == 200 else {}
    if summary:
        print(f"        faces_blurred={summary.get('faces_blurred')} "
              f"documents_redacted={summary.get('documents_redacted')} "
              f"frames={summary.get('frames_processed')} "
              f"detection_frames={summary.get('detection_frame_pct')}% "
              f"resolution={summary.get('output_resolution')} "
              f"audio_preserved={summary.get('audio_preserved')}")
        check(int(summary.get("verified_frame_count") or 0) == meta["frame_count"],
              "summary verified frame count matches the input")

    # 10. Download and independently verify the artefact
    print("10. Download and decode the protected video")
    res = requests.get(f"{BASE_URL}/videos/download/{task_id}", timeout=300)
    check(res.status_code == 200, f"download returned 200 (got {res.status_code})")
    if res.status_code != 200:
        return 1
    out_dir = os.path.join(PROJECT_ROOT, "backend", "storage", "validation")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"e2e_{task_id}_protected.mp4")
    with open(out_path, "wb") as handle:
        handle.write(res.content)
    size = os.path.getsize(out_path)
    check(size > 0, f"downloaded file is non-empty ({size} bytes)")

    cap = cv2.VideoCapture(out_path)
    check(cap.isOpened(), "downloaded file opens with OpenCV")
    decoded = 0
    while True:
        ok, _frame = cap.read()
        if not ok:
            break
        decoded += 1
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"        decoded {decoded} frames at {width}x{height}")
    check(decoded == meta["frame_count"], f"decoded {decoded} frames, expected {meta['frame_count']}")
    expected_w, expected_h = (int(part) for part in meta["resolution"].lower().split("x"))
    check((width, height) == (expected_w, expected_h), "output resolution matches the input")

    # 11. Original (before/after review) and unknown task behaviour.
    print("11. Original video endpoint and 404 handling")
    res = requests.get(f"{BASE_URL}/videos/original/{task_id}", timeout=300)
    check(res.status_code == 200, f"original video returned 200 (got {res.status_code})")
    check(len(res.content) > 0, "original video body is non-empty")
    res = requests.get(f"{BASE_URL}/redact/status/task_does_not_exist", timeout=30)
    check(res.status_code == 404, f"unknown task returns 404 (got {res.status_code})")

    print("\n=== Result ===")
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED:")
        for failure in FAILURES:
            print(f"  - {failure}")
        print(f"Protected artefact kept for inspection: {out_path}")
        return 1
    print(f"All checks passed. Protected artefact: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
