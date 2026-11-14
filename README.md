# SafeFrame AI

A deep-learning video privacy redaction system built as a college project.
SafeFrame AI detects faces with **YuNet (ONNX)** through OpenCV DNN, tracks
them with a lightweight IoU tracker, masks each face region, re-encodes the
video with **FFmpeg (H.264)**, and only offers the result for download after the
encoded output has been independently re-decoded and verified.

**Scope: face redaction only.** License plates, ID cards, laptop screens and
phone screens are visible in the UI but disabled with a "Coming Later" label and
are rejected by the backend. They are never silently ignored.

This is a demonstration/engineering-review project, not a certified privacy
product. See [Known limitations](#known-limitations).

---

## 1. What actually works

| Capability | State |
| --- | --- |
| YuNet face detection (real inference, ONNX) | Implemented, validated |
| Per-face Gaussian blur / pixelation / blackout | Implemented, validated |
| Mask on the first frame a face is detected | Implemented, unit-tested |
| IoU face tracking with constant-velocity association | Implemented, regression-tested |
| 2-frame conservative hold for missed detections | Implemented, regression-tested |
| Face-proportional padding, rounded mask, union merging, bounds clamping | Implemented, unit-tested |
| Jitter smoothing that never shrinks below the detection | Implemented, regression-tested |
| FFmpeg H.264 encode with audio passthrough | Implemented, verified on an audio clip |
| Full-output decode verification (size, frames, resolution) | Implemented |
| Real progress, telemetry over WebSocket, cooperative cancel | Implemented |
| Debug frames (detector boxes, confidences, track IDs, final ROIs) | Implemented, off by default |
| License plates, ID cards, laptop/phone screens | **Not implemented** (disabled in UI, refused by API) |
| Re-ID (OSNet), BoT-SORT, ghost-box prediction | **Not active** (no validated weights/model) |
| CUDA / TensorRT acceleration | **Not used**; inference runs on CPU |
| Metrics such as mAP, MOTA, precision/recall on real footage | **Not measured** (no ground truth) |

---

## 2. Architecture

```
React 19 + TypeScript + Vite (src/)
  pages: Landing, Dashboard, Upload, Processing, Results
  services/api.ts        REST client (VITE_API_BASE_URL)
  services/websocket.ts  live telemetry (reconnect, terminal state from REST)
        │  HTTP / WebSocket
        ▼
FastAPI (backend/main.py)
  api/v1/endpoints/      health, videos, redact, dashboard
  services/              upload validation, task status, readiness, dashboard
  workers/task_manager   process-local task registry (single source of truth)
        │
        ▼
Pipeline (backend/services/ai/video_processor_service.py)
  cv2.VideoCapture ─► DetectionService (YuNet ONNX)
                   ─► TrackingService (IoU + velocity association)
                   ─► BlurService (ROI plan + blur/pixelate/blackout)
                   ─► FFmpeg stdin (rawvideo) ─► H.264 MP4 + AAC audio
                   ─► output verification (decode every frame)
```

Request flow for a redaction job:

1. `POST /api/v1/videos/upload` — the file is saved, opened with OpenCV, and
   rejected with `400` if it has no decodable frames. Real metadata is returned.
2. `POST /api/v1/redact/start` — the detector and FFmpeg are preflighted before
   a task exists; unsupported categories are refused with `503`.
3. The worker processes frames, writing real progress into the task registry and
   broadcasting throttled telemetry on `/ws/redact/{task_id}`.
4. On completion the encoder is joined, then the output is reopened and every
   frame is decoded. Only a verified output sets `output_verified: true`.
5. `GET /api/v1/videos/download/{task_id}` returns `409` until that flag is set.

---

## 3. Requirements

- Windows 11 (developed and validated here), macOS/Linux should work too
- Python 3.11+
- Node.js 20+ and npm
- **FFmpeg** on `PATH` (or set `FFMPEG_PATH`). Windows build:
  <https://www.gyan.dev/ffmpeg/builds/> — extract and add the `bin` folder to
  `PATH`, then check `ffmpeg -version`.
- YuNet weights: `face_detection_yunet_2023mar.onnx` in the project root,
  downloaded and SHA-256 verified by `scripts/download_yunet.py`.
- No GPU is required. Inference is CPU-only (`cv2.dnn`, single thread per
  frame); an NVIDIA GPU is detected nowhere and claimed nowhere.

Python packages are in `backend/requirements.txt`: FastAPI, Uvicorn, Pydantic,
python-multipart, websockets, python-dotenv, OpenCV, NumPy, pytest, requests.

---

## 4. Setup (Windows)

```powershell
# from the project root
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r backend/requirements.txt
python scripts/download_yunet.py          # downloads + verifies the ONNX model
ffmpeg -version                           # must print a version

npm install
```

If PowerShell blocks activation: `Set-ExecutionPolicy -Scope Process Bypass`.

### Run

Terminal 1 — backend:

```powershell
.venv\Scripts\Activate.ps1
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Terminal 2 — frontend:

```powershell
npm run dev
```

Open <http://localhost:5173>. The upload page reads
`GET /api/v1/health/readiness`; processing stays disabled until the detector
weights and FFmpeg are both reported available.

Optional environment overrides:

| Variable | Default | Purpose |
| --- | --- | --- |
| `VITE_API_BASE_URL` | `http://localhost:8000/api/v1` | REST base URL |
| `VITE_WS_BASE_URL` | `ws://localhost:8000/ws/redact` | WebSocket base URL |
| `FFMPEG_PATH` | resolved from `PATH` | explicit FFmpeg executable |

---

## 5. How redaction is done

- **Detection**: YuNet is created once (`cv2.FaceDetectorYN`) with an internal
  candidate threshold of 0.25; detections below
  `DETECTION_CONF_THRESHOLD = 0.60` are not masked. The threshold was chosen from
  measurements on this checkout: real faces scored 0.86–0.95 from 36 px to
  270 px, rotated 45°, and in low light, while spurious boxes on a crowded photo
  scored 0.36–0.51.
- **Coordinates**: YuNet returns pixel coordinates; the detector normalizes them
  to 0–100 % once, and BlurService converts back to pixel ROIs against the actual
  decoded frame size. No double scaling.
- **Masking**: every detection in the current frame is masked immediately, so a
  face is covered on its first frame; the tracker is not required for coverage.
  The mask is built from the detected box itself: padding is **12 % of its own
  width and 14 % of its own height** (2–64 px, clipped to the frame), and the
  painted silhouette is a **superellipse (rounded rectangle)** inscribed in the
  padded region rather than a raw rectangle. The whole detected face rectangle
  is always at 100 % filter strength; the rounded outer third of the padding ring
  fades into the background. Overlapping plans for the same face are merged by
  union, so a shifted smoothed box can never cut a corner off the detection and
  the same pixels are never filtered twice.
- **Intensity**: the Gaussian sigma scales with the ROI (0.18–0.30 × smaller
  side), with a measured floor that keeps a redacted face undetectable even at
  the lowest setting. Pixelation keeps a face-relative grid (at most 8 blocks
  across the smaller face side, 8 px minimum block) and uses a hard rounded rim
  — a blended mosaic rim both looks muddy and leaks recognizable edge pixels.
  Blackout fills the same rounded silhouette opaquely.
- **Tracking**: IoU association with a smoothed constant-velocity prediction used
  only to rank matches (never to move a mask). Redaction boxes are smoothed only
  below `TRACK_SMOOTHING_SNAP_RATIO` of the face size and always expanded to
  contain the current detection, so jitter is damped but a moving face is never
  trailed and coverage can only grow. A confirmed track missed by the detector
  keeps its last box for `REDACTION_HOLD_FRAMES = 2` frames, without moving or
  growing, then stops being masked.
- **Encoding**: frames are piped as raw BGR to FFmpeg, which maps `0:v:0` and
  `1:a:0?`, encodes H.264 (`libx264`, CRF 23, `yuv420p`, faststart) and copies
  audio as AAC when the input has an audio stream. Resolution and frame rate are
  preserved; odd dimensions are rounded down to even for yuv420p.
- **Verification**: the encoded file must exist, be non-empty, open with OpenCV,
  decode to the expected frame count and report the expected dimensions. Failures
  mark the task `FAILED`, delete the partial output, and are never downloadable.

---

## 6. Validation: measured results

### 6.1 Test suite

```powershell
python -m pytest backend/tests -q
```

Last run in this checkout: **79 passed, 0 failed** (a few seconds of warnings
only). The suite covers:

- fail-closed detector behavior (no fabricated detections, explicit errors),
- blur geometry: exact ROI, face-proportional padding on both axes, bounds,
  fractional targets, overlaps, no modification outside planned ROIs, rounded
  silhouette, full coverage of small/large/rotated/multiple faces, all modes,
- privacy strength floors (Gaussian, pixelation, blackout, feathered edges),
- tracking: crossing faces keep their IDs, short occlusion hold, expiry,
  no duplicate track after a one-frame gap, held boxes do not drift or grow,
  mask IoU across jittering frames, coverage restored after a detection gap,
- pipeline on real fixtures: no-face clip masks nothing, all three modes
  produce a verified output at the input resolution/frame count, audio survives,
  smoothing never shrinks a box below its detection,
- model manager: missing/empty/invalid YuNet weights, no heavy loading on
  health checks, FFmpeg discovery failures,
- pipeline: encoder failures, invalid output, cancellation, no fabricated
  progress, failed tasks cannot be downloaded,
- operational: upload validation and real-video acceptance.

The earlier `SAFEFRAME_ENGINEERING_AUDIT.md` records the pre-fix audit. It is
history, not a description of the current build.

### 6.2 Real pipeline on validation clips

Generated fixtures: `python scripts/make_validation_videos.py` (downloads two
public face photographs on first run, then builds the clips under `.cache/`).
Full run with metrics: `python scripts/evaluate_redaction.py`
(Report: `backend/storage/validation/report.json`.) Fixture metadata stores a
**project-relative** path and the evaluator resolves clips next to their JSON, so
validation works on any machine instead of depending on the absolute path of the
machine that generated the fixtures.

The fixtures are composites built from real photographs (single frontal face,
three faces, small/distant faces, rotated face, partially occluded face,
low-light, no faces). They are **not** surveillance footage, and no independent
ground-truth annotation exists for real footage.

Run on 2026-11-10, 640×480, CPU inference:

| clip | style | frames | detection recall* | GT faces redacted* | still detectable after* | background change† | pipeline FPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frontal_single | gaussian | 72 | 100.0 % | 100.0 % | 0 | 0.06 | 48.2 |
| low_light | gaussian | 60 | 100.0 % | 100.0 % | 0 | 0.03 | 45.3 |
| multi_face | gaussian | 96 | 97.9 % | 100.0 % | 0 | 0.17 | 45.3 |
| no_faces | gaussian | 48 | n/a | n/a | 0 | 0.00 | 52.5 |
| occluded | gaussian | 72 | 94.4 % | 99.3 % | 1 | 0.14 | 43.2 |
| rotated | gaussian | 48 | 100.0 % | 100.0 % | 0 | 1.35 | 43.3 |
| small_faces | gaussian | 60 | 96.7 % | 97.8 % | 4 | 0.07 | 41.9 |
| frontal_single | pixelate | 72 | 100.0 % | 100.0 % | 0 | 0.05 | 50.2 |
| frontal_single | blackout | 72 | 100.0 % | 100.0 % | 0 | 0.07 | 50.9 |
| multi_face | pixelate | 96 | 97.9 % | 100.0 % | 0 | 0.19 | 47.4 |
| multi_face | blackout | 96 | 97.9 % | 100.0 % | 0 | 0.34 | 46.6 |

\* Measured against the generator's own face placements, with a detection match
at IoU ≥ 0.30. "Still detectable after" re-runs YuNet on the protected frame at a
sensitive 0.30 threshold; a face still found there is counted as a leak.
† Mean absolute pixel difference outside the ground-truth faces. Near-zero means
blur stayed inside the face regions instead of smearing the scene.

What the numbers say:

- Redaction is localized: background change is 0.02–0.15 on every clip except
  `rotated` (1.29), where the rotating face legitimately moves between frames.
- Exact frame count and resolution are preserved on every clip
  (`output_frames_verified` equals the input frame count in the JSON report).
- No false detections were produced on any fixture, including `no_faces`.
- Tightening the mosaic grid (8 blocks instead of 10 across the face) removed
  the multi-face pixelation leaks on this fixture: 7 of 288 → 0 of 288, with the
  same background change and a stronger mosaic.
- Weak spots, reported rather than hidden: 4 of 180 small-face instances and 1
  of 144 occluded-face instances remained detectable at the sensitive
  re-detection threshold. Gaussian and blackout covered every multi-face
  instance. For a presentation, Gaussian is the safest visual choice.

### 6.3 Redaction comparison (before/after this change)

```powershell
python scripts/render_redaction_comparison.py --clip frontal_single --frame 20 --style gaussian
python scripts/render_redaction_comparison.py --clip multi_face --frame 40 --style gaussian
```

Writes `backend/storage/validation/redaction_comparison_<clip>_<style>.png`: four
panels on one real fixture frame — original, real detected boxes, the previous
hard padded rectangle, and the current face-proportional mask. The previous
rectangle is re-created by the script itself, so the difference is visible
rather than described.

### 6.4 End-to-end HTTP test (real server, real upload, real download)

With the backend running:

```powershell
python test_script.py
```

Last run: **all checks passed**. It verifies liveness, readiness, refusal of an
unsupported category, rejection of a corrupt upload, real upload metadata,
starting a face-only job, the download gate returning `409` before completion,
real intermediate progress, completion with `output_verified`, the summary
frame count, a downloaded file that decodes to the exact frame count and
resolution, the original-video endpoint, and `404` for an unknown task.

Audio was checked separately on a clip with an AAC track: the protected output
contained an H.264 video stream and an AAC audio stream at the original
resolution and frame rate, and the summary reported `audio_preserved: true`.

---

## 7. Debug mode

Tick "Save debug frames" before starting a job (or pass `"debug_mode": true`).
For each of the first 300 frames the backend writes, under
`backend/storage/debug/<task_id>/`:

- `original/frame_XXXXXX.jpg` — decoded input frame,
- `annotated/frame_XXXXXX.jpg` — detector boxes with confidence, track IDs and
  the exact mask silhouette that was applied (rounded, not a rectangle),
- `protected/frame_XXXXXX.jpg` — the frame that was encoded,
- `detections.json` — machine-readable detections, tracks, and ROI plans.

Debug overlays are written to separate JPEGs only; the protected video never
contains boxes.

---

## 8. Presentation demo (≈5 minutes)

1. Start the backend and frontend (section 4). Open the dashboard: the status
   line should read `YUNET CPU — FFmpeg ready` with a green dot.
2. Upload a short clip (5–10 s, 720p is plenty) with 1–3 visible faces. Confirm
   the metadata card shows the real resolution, FPS and frame count.
3. Keep "Faces" selected — the other targets show "Coming Later" and cannot be
   enabled. Choose **Gaussian**, intensity ~80, optionally enable debug frames.
4. Start processing. Point out that progress, FPS, masked-region count and the
   log stream come from the worker, and that a Cancel button actually stops the
   job and deletes the partial file.
5. When it completes, the results page opens automatically with **Original and
   Protected side by side** at the same size, the selected method badge and a
   verified badge. Show the summary (faces masked, frames, detection coverage,
   output resolution, audio preserved).
6. Download the protected MP4 and play it: faces are masked, the rest of the
   scene is intact, and the file plays in any player.
7. Optional evidence: open the debug folder to show detector boxes plus the
   exact mask silhouette hugging each face, or open
   `backend/storage/validation/redaction_comparison_frontal_single_gaussian.png`
   for the before/after comparison. Run `python test_script.py` or
   `python -m pytest backend/tests -q` live.

Screenshots are not committed. Capture the dashboard, upload, processing and
results pages from this flow if the report needs them.

---

## 9. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| Upload page says detector unavailable | Model missing or invalid. Run `python scripts/download_yunet.py`; check `GET /api/v1/health/readiness` → `reasons`. |
| "ffmpeg executable is not available" | Install FFmpeg, add `bin` to `PATH`, restart the backend; or set `FFMPEG_PATH`. |
| `Start` returns 503 for license plates | Expected: no validated detector exists for those categories. |
| `Download` returns 409 | The task is not completed and verified yet, or it failed. Check the status endpoint / Processing page. |
| Video plays in the browser but not in Windows Media Player | Output is H.264 + AAC in MP4; use a current player. Browser playback needs H.264 support. |
| `WebSocket telemetry: DISCONNECTED` | The page still polls REST status; check the backend is on port 8000 and CORS allows `http://localhost:5173`. |
| Port already in use | `uvicorn ... --port 8001` plus `VITE_API_BASE_URL=http://localhost:8001/api/v1 npm run dev`. |
| Task stuck at 0 % | Look at the Processing page log stream and the backend console; a decode or encoder failure is reported there and in `failure_reason`. |
| `python download_yunet.py` checksum error | The download was modified. Delete the `.onnx` file and re-run; the script only accepts the pinned SHA-256. |

---

## 10. Known limitations

- **Face redaction only.** Plates, ID cards and screens are UI placeholders and
  are rejected by the API.
- **Detection is not perfect.** Small, heavily occluded, extreme-profile or
  motion-blurred faces can be missed. A few small/occluded faces still pass the
  deliberately sensitive 0.30 re-detection check after redaction (see 6.2).
  There is no guarantee of complete coverage, and the project is not
  privacy-certified.
- **Masks are not face-shaped exactly.** The rounded silhouette is derived from
  the detector's rectangle, not from a face segmentation model: it is tight to
  the detected face and leaves most hair/shoulder pixels untouched, but it is
  still a geometric mask, and the padding ring fades into the background.
- **No ground-truth benchmark on real footage.** The metrics above are measured
  on generated composites. mAP/MOTA/precision/recall on real surveillance video
  are not measured and are not claimed.
- **CPU-only inference.** No CUDA/TensorRT path is implemented; adding one would
  require a validated alternative backend, not just a flag.
- **Tasks are process-local.** Restarting the backend loses task history (the
  files remain on disk). There is no database, auth, or multi-user separation.
- **No Re-ID or BoT-SORT.** The IoU + velocity tracker is sufficient for face
  masking, which does not depend on IDs; it can still swap IDs during a full
  overlap, though both faces remain masked.
- **Audio** is copied when present, re-encoded to AAC; a source with no audio
  yields a video-only output, reported as `audio_preserved: false`.
- **Debug frames** are capped at 300 per task to bound disk usage.
- **Debug/storage growth** is not automatically pruned; delete
  `backend/storage/debug` and processed videos manually between demos if needed.

---

## 11. Project layout

```
backend/
  main.py                     FastAPI app, lifespan warmup, WebSocket route
  api/v1/endpoints/           health, videos, redact, dashboard
  config/                     app settings, ai_settings (thresholds, blur, FFmpeg)
  core/                       logging, domain exceptions/handlers
  schemas/                    Pydantic request/response contracts
  services/                   upload, redaction orchestration, readiness, dashboard
  services/ai/                model manager, detection, tracking, blur, pipeline
  utils/                      FFmpeg/ffprobe discovery, storage paths
  workers/task_manager.py     in-process task registry
  tests/                      pytest suites
scripts/
  download_yunet.py           pinned + checksum-verified model download
  make_validation_videos.py   builds the composite validation fixtures
  evaluate_redaction.py       runs the real pipeline and measures it
  render_redaction_comparison.py  renders before/after redaction panels
src/
  pages/                      Landing, Dashboard, Upload, Processing, Results
  components/                 layout, upload, processing, results, dashboard, ui
  services/                   REST + WebSocket clients
  store/                      Zustand app state
  types/                      shared TypeScript contracts
test_script.py                HTTP end-to-end smoke test
SAFEFRAME_ENGINEERING_AUDIT.md  original audit (pre-fix historical record)
```

---

## 12. Responsible-use note

Face redaction is a privacy tool, but this implementation is a student project:
it can miss faces and it has no audit trail, authentication or encryption at
rest. Do not present its output as certified anonymization, and do not process
recordings of people who have not consented to the demonstration.
