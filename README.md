# SafeFrame AI

A deep-learning video privacy redaction system built as a college project.
SafeFrame AI detects faces with **YuNet (ONNX)**, license plates with an
**RT-DETRv2 plate model (ONNX)**, identity documents (ID cards and passports)
with a **DocAligner FastViT document-corner model (ONNX)**, and laptop screens
with an **RT-DETR R18 laptop detector (ONNX) plus a display-quad refinement
stage**. The first three run through OpenCV DNN; the laptop graph runs through
onnxruntime. It tracks every category with a lightweight IoU tracker, masks each
region, re-encodes the video with **FFmpeg (H.264)**, and only offers the result
for download after the encoded output has been independently re-decoded and
verified.

**Scope: faces, license plates, ID cards, passports, laptop screens and phone
screens.** Phone screens are their own independent category, masked from the
estimated display polygon or, when that boundary cannot be separated, from the
whole device box with the fallback reported. The COCO phone proposal can miss
extreme close-ups and glare-heavy displays; that limitation is measured and
documented (see `docs/PHASE5_PHONE_SCREENS.md`). The document detector
is class-agnostic: it localises an ID card or passport but does not claim which
of the two it is. The laptop model localises the **device**, not the display:
the visible screen is estimated from image edges, and when that boundary cannot
be separated the whole device box is masked and reported as a fallback (see
[Known limitations](#known-limitations) and
[docs/PHASE4_LAPTOP_SCREENS.md](docs/PHASE4_LAPTOP_SCREENS.md)).

This is a demonstration/engineering-review project, not a certified privacy
product. See [Known limitations](#known-limitations).

---

## 1. What actually works

| Capability | State |
| --- | --- |
| YuNet face detection (real inference, ONNX) | Implemented, validated |
| RT-DETRv2 license-plate detection (real inference, ONNX) | Implemented, validated on 40 annotated photos |
| DocAligner ID-card / passport detection (real inference, ONNX) | Implemented, validated on 60 annotated MIDV frames |
| RT-DETR laptop detection (real inference, ONNX) + display-quad refinement | Implemented, validated on 8 real annotated photos (section 6.8) |
| Per-target Gaussian blur / pixelation / blackout | Implemented, validated |
| Mask on the first frame a target is detected | Implemented, unit-tested |
| IoU tracking with constant-velocity association (faces, plates, documents, laptop screens) | Implemented, regression-tested |
| Conservative hold for missed detections (2 frames faces/plates/screens, 3 documents) | Implemented, regression-tested |
| Proportional padding; rounded face masks, rectangular plate/document masks; union merging | Implemented, unit-tested |
| Jitter smoothing that never shrinks below the detection | Implemented, regression-tested |
| Face-only, plate-only, ID-card-only, passport-only and combined jobs | Implemented, pipeline-tested |
| FFmpeg H.264 encode with audio passthrough | Implemented, verified on an audio clip |
| Full-output decode verification (size, frames, resolution) | Implemented |
| Real progress, telemetry over WebSocket, cooperative cancel | Implemented, cancel-shutdown regression fixed |
| Debug frames (detector boxes, confidences, track IDs, final ROIs) | Implemented, off by default |
| Phone screens | **Implemented**; display quad estimated from edges, unrefined cases fall back to the device box and say so |
| Laptop-screen boundary detection | **Estimated, not model-predicted**; unrefined cases fall back to the device box and say so |
| Phone-screen boundary detection | **Estimated, not model-predicted**; 3/5 annotated photos detected, 1/3 of those fell back to the device box |
| Re-ID (OSNet), BoT-SORT, ghost-box prediction | **Not active** (no validated weights/model) |
| CUDA / TensorRT acceleration | **Not used**; inference runs on CPU |
| mAP/MOTA on real surveillance footage | **Not measured**; plate precision/recall and document mask recall measured on annotated photo sets |

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
  cv2.VideoCapture ─► DetectionService (YuNet faces + RT-DETR plates + DocAligner documents, one decoded frame)
                   ─► TrackingService (category-aware IoU + velocity association)
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
- License-plate weights: `plate_detection_rtdetr_r18.onnx` in the project root
  (80 MB), downloaded and SHA-256 verified by `scripts/download_plate_model.py`.
  A plate-only job fails with a clear `503` when this file is missing; it is
  never substituted with another detector.
- Identity-document weights: `document_detection_docaligner_fastvit.onnx` in the
  project root (83 MB), downloaded and SHA-256 verified by
  `scripts/download_document_model.py`. A document job fails with a clear `503`
  when this file is missing; there is no fallback detector.
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
python scripts/download_yunet.py          # downloads + verifies the face model
python scripts/download_plate_model.py    # downloads + verifies the plate model
python scripts/download_document_model.py # downloads + verifies the document model
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

- **Face detection**: YuNet is created once (`cv2.FaceDetectorYN`) with an
  internal candidate threshold of 0.25; detections below
  `DETECTION_CONF_THRESHOLD = 0.60` are not masked. The threshold was chosen from
  measurements on this checkout: real faces scored 0.86–0.95 from 36 px to
  270 px, rotated 45°, and in low light, while spurious boxes on a crowded photo
  scored 0.36–0.51.
- **Plate detection**: an RT-DETRv2 (R18) ONNX graph fine-tuned on the Open
  Images V7 "Vehicle registration plate" class, run through `cv2.dnn`. Input is
  RGB `/255` at 640×640; the two outputs (`logits` [1,300,1], `pred_boxes`
  [1,300,4]) are sigmoid-scored, thresholded at `PLATE_CONF_THRESHOLD = 0.10`
  and NMS-filtered (IoU 0.45). The low-looking threshold is correct for this
  single-class RT-DETR export: confident plates score ~0.10–0.20, and the value
  was measured (section 6.2), not guessed. All detectors consume the same
  decoded frame; a face-only job never loads the 80 MB plate graph.
- **Document detection**: a DocAligner **FastViT-SA24** corner-heatmap graph
  (Apache-2.0, DocsaidLab, trained on MIDV-500/MIDV-2020), run through the same
  `cv2.dnn` backend. Input is BGR `/255` at 256×256; the named output `heatmap`
  is `[1,4,128,128]`, one channel per document corner. Each channel's peak
  (sub-pixel centroid, threshold 0.30) becomes a corner; a missing fourth corner
  is completed by the parallelogram implied by the other three, which raised
  mask recall from 44/60 to 56/60 on the MIDV sample. Detections are filtered by
  frame area (0.2–90 %) and aspect ratio (0.30–3.30), then converted to a
  quadrilateral and an axis-aligned box. The graph is single-document per
  inference, so a frame is also scanned as four overlapping half-strips;
  results are deduplicated by mask IoU ≥ 0.35 and capped at 3 documents per
  frame. Confidence is the minimum of the four corner peaks, so a weak corner
  lowers the score instead of being hidden. The graph is class-agnostic: both
  `id_cards` and `passports` run this one detector once, and the result is
  attributed internally to `ID_CARDS` with the label `DOCUMENT` (see the
  limitations).
- **Coordinates**: YuNet returns pixel coordinates; the detector normalizes them
  to 0–100 % once, and BlurService converts back to pixel ROIs against the actual
  decoded frame size. No double scaling.
- **Masking**: every detection in the current frame is masked immediately, so a
  target is covered on its first frame; the tracker is not required for coverage.
  The mask is built from the detected box itself. Faces use **12 % of their own
  width and 14 % of their own height** padding (2–64 px) and a **superellipse
  (rounded rectangle)** silhouette; the whole detected face rectangle is always
  at 100 % filter strength, and only the outer part of the padding ring fades.
  Plates use **6 % width and 20 % height** padding (2–48 px) and a **full
  rectangle**: a plate is a flat high-contrast rectangle, and any rounded or soft
  rim could leave a character fragment readable. Documents use **6 % width and
  6 % height** padding (4–64 px) and a **full rectangle** covering the whole
  card or passport, including the photograph, printed fields and the MRZ, so a
  rotated document cannot expose a corner. Overlapping plans for the same
  category are merged by union, so a shifted smoothed box can never cut a corner
  off the detection; faces, plates and documents are never merged with each
  other.
- **Intensity**: the Gaussian sigma scales with the ROI (0.18–0.30 × smaller
  side), with a measured floor that keeps a redacted face undetectable even at
  the lowest setting. Pixelation keeps a face-relative grid (at most 8 blocks
  across the smaller face side, 8 px minimum block) and uses a hard rounded rim
  — a blended mosaic rim both looks muddy and leaks recognizable edge pixels.
  Plates use a stricter **4-block** cap: the face-tuned 8-block grid left a
  plate-shaped mosaic still detected by the plate model on the comparison photo
  and a panning clip, while 4 blocks produced zero re-detections. Documents use
  the same 4-block cap. Blackout fills the same mask opaquely.
- **Tracking**: IoU association with a smoothed constant-velocity prediction used
  only to rank matches (never to move a mask). Redaction boxes are smoothed only
  below `TRACK_SMOOTHING_SNAP_RATIO` of the target size and always expanded to
  contain the current detection, so jitter is damped but a moving target is never
  trailed and coverage can only grow. Association is category-aware, and plates
  use  their own confidence floor (`PLATE_CONF_THRESHOLD`) instead of the face
  floor, which would otherwise discard every RT-DETR score. Documents use a
  `DOCUMENT_CONF_THRESHOLD` association floor and the `DOC` label. A confirmed
  track missed by the detector keeps its last box for 2 frames (3 for a
  document) without moving or growing, then stops being masked; stale boxes
  cannot drift onto other objects.
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

Last run in this checkout: **152 passed, 98 subtests passed, 0 failed**.
(One earlier environmental failure, Starlette's `TestClient` requiring the
`httpx2` package, was fixed by installing that test dependency; the same
endpoints are also verified over real HTTP in section 6.5.) The suite covers:

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
- operational: upload validation and real-video acceptance,
- license plates: single/multiple/small/large/angled decoding, threshold and NMS,
  rectangular full-coverage masks in all three modes, proportional padding,
  aspect ratios, independent multi-plate masks, plate tracking (jitter, motion,
  short hold, expiry), API acceptance/rejection, missing weights, and real
  plate-only / combined / face-only pipeline runs with audio and re-detection.
- identity documents: corner-heatmap decoding (peak, parallelogram completion,
  low-confidence rejection, implausible geometry), multi-document dedup, the
  shared detector running once for ID cards and passports, whole-document
  rectangle masks in all three modes, document padding and three-frame hold,
  document tracking, API acceptance of both new categories, rejection of
  laptop/phone screens, missing document weights, combined face + plate +
  document jobs, real-pixel rotated, small/distant and two-document frames, and
  real document-only pipeline runs whose protected output is re-checked for
  recognizable (correlated) document content in every mode.
- laptop screens (Phase 4): RT-DETR decoding, class filtering, threshold, NMS,
  multiple devices, unavailable/inference failures, display-quad gates on
  synthetic front-facing/perspective/dark/glare/inner-window scenes, the
  privacy invariant that no annotated display pixel survives in any mode, a
  refined quad leaving the keyboard untouched, device-box fallback reporting,
  independent multi-screen redaction, detection/track polygon merging, the
  two-frame hold, API acceptance and refusal of phone screens, and real
  detector + FFmpeg runs (all modes, audio/resolution, combined categories,
  face-only regression) plus a real annotated-photo evaluation script.
- phone screens (Phase 5): RT-DETR class-67 decoding, display-quad gates that
  reject laptop/TV-like rectangles, all three mask modes, outward padding, the
  two-frame hold, category independence (laptop-only jobs never run the phone
  detector), category-specific `phone_fallback` reporting, API/readiness/503
  preflight, summary counts, and real detector + FFmpeg runs on an annotated
  phone photo, plus a real five-photo evaluation script.

The earlier `SAFEFRAME_ENGINEERING_AUDIT.md` records the pre-fix audit. It is
history, not a description of the current build.

### 6.2 License-plate detection validation (real annotated photos)

```powershell
python scripts/download_plate_model.py
python scripts/evaluate_plate_detection.py --count 40
```

The evaluator fetches the `keremberke/license-plate-object-detection` validation
split (CC BY 4.0; Roboflow Universe / Augmented Startups) through the Hugging
Face datasets server and matches detections to ground-truth plate boxes at
IoU ≥ 0.50. It writes every per-image result to
`.cache/validation/plates/evaluation.json`, so the procedure is reproducible and
the sample size is recorded. It does not extrapolate.

Last run on 2026-11-14, RT-DETR CPU, 40 images / 44 ground-truth boxes,
confidence threshold 0.10:

| Metric | Result |
| --- | ---: |
| True positives / false positives / false negatives | 40 / 4 / 4 |
| Precision | 0.909 |
| Recall | 0.909 |
| Mean IoU of true positives | 0.788 |
| Small/distant plate recall (GT width < 40 px) | 0.80 (4 of 5) |
| Mean inference time | 298.8 ms/image (CPU) |

Redaction re-check on a real 12-frame photo clip: after processing, the plate
detector was re-run on input and protected frames in lockstep. Zero plate
instances that were present in the input remained detectable at the same
location in the output, in **all three modes** (Gaussian, pixelate, blackout).
This is a privacy proxy on one clip, not a guarantee for every plate.

For comparison, the MIT-declared YOLOv8 plate export evaluated on the same 40
images reached precision 0.947 / recall 0.818 and only 0.20 small-plate recall at
52 ms/image. It was not selected: the Apache-2.0 RT-DETR model is better
documented, has cleaner license provenance, and finds small/distant plates far
more reliably, which matters more for privacy than raw speed. These numbers are
for that 40-image sample only, not a benchmark of all plate types or regions.

### 6.3 Real pipeline on validation clips

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

### 6.4 Redaction comparison (before/after this change)

```powershell
python scripts/render_redaction_comparison.py --clip frontal_single --frame 20 --style gaussian
python scripts/render_redaction_comparison.py --clip multi_face --frame 40 --style gaussian
```

Writes `backend/storage/validation/redaction_comparison_<clip>_<style>.png`: four
panels on one real fixture frame — original, real detected boxes, the previous
hard padded rectangle, and the current face-proportional mask. The previous
rectangle is re-created by the script itself, so the difference is visible
rather than described.

### 6.5 End-to-end HTTP test (real server, real upload, real download)

With the backend running:

```powershell
python test_script.py
```

Last run: **all checks passed**. It verifies liveness, readiness (including the
plate-, document-, laptop- and phone-detector status), refusal of an unsupported
category (`smartwatches`), acceptance of `laptop_screens` and `phone_screens`
(or an explicit missing-model error), acceptance of `id_cards` and `passports`
(or an explicit missing-model error), rejection of a corrupt upload, real upload
metadata, starting a face-only job, the download gate returning `409` before
completion, real intermediate progress, completion with `output_verified`, the
summary frame count, a downloaded file that decodes to the exact frame count and
resolution, the original-video endpoint, and `404` for an unknown task.

### 6.6 License-plate comparison images

```powershell
python scripts/render_plate_redaction_comparison.py
```

Writes, into `backend/storage/validation/plate_comparison/`: `original.jpg`,
`detected_boxes.jpg` (real detector boxes and scores), `gaussian.jpg`,
`pixelate.jpg`, `blackout.jpg`, plus the machine-readable `detections.json`.
The images use the real detector and the real mask compositor, so they show the
actual pipeline output rather than a mock-up. They are git-ignored.

Audio was checked separately on a clip with an AAC track: the protected output
contained an H.264 video stream and an AAC audio stream at the original
resolution and frame rate, and the summary reported `audio_preserved: true`.

### 6.7 Identity-document detection validation (real annotated frames)

```powershell
python scripts/download_document_model.py
python scripts/evaluate_document_detection.py \
    --negatives .cache/validation/plates
```

The evaluator fetches the `Noaman/midv500` validation split through the Hugging
Face datasets server: 60 real 1080×1920 phone photos of identity documents, each
with a binary document mask. It compares the detector's quadrilateral mask with
the ground-truth mask; a match needs IoU ≥ 0.50. Every per-image result is
written to `.cache/validation/documents/evaluation.json`, and the report records
the exact sample size and thresholds.

Model provenance:

| Property | Value |
| --- | --- |
| Graph | DocAligner FastViT-SA24, corner-heatmap ONNX |
| Upstream | DocsaidLab/DocAligner (Apache-2.0), trained on MIDV-500/MIDV-2020 |
| Weights host | `7rplus/pagescan-weights` on Hugging Face |
| File / size | `fastvit_sa24_h_e_bifpn_256_fp32.onnx`, 83,084,930 bytes |
| SHA-256 | `7f9f5a8935b2eb22b3ee0245d34996063f54562df390d34714af2d76928695bc` |
| Backend | `cv2.dnn`, 256×256 BGR input, `[1,4,128,128]` heatmap |

Last run on 2026-11-14, CPU, confidence threshold 0.30:

| Metric | Result |
| --- | ---: |
| Documents detected / ground truth | 56 / 60 |
| Document recall (mask IoU ≥ 0.50) | 0.933 |
| Mean mask IoU of true positives | 0.9826 |
| Extra predicted regions on document frames | 0 |
| Non-document vehicle photos with a detection (40 images) | 5 |
| Mean inference time (5-scan frame) | 867.8 ms/frame (CPU) |

The 40 negative images are the license-plate evaluator's vehicle photos, which
contain no identity documents. The 5 flags are plate-like rectangles (one is a
motorcycle plate scoring 0.92); for a redaction tool this fails closed — it
over-redacts a vehicle region rather than exposing private text. It is still a
precision limitation, not a success claim.

The graph is **class-agnostic**. It finds a document and its corners, but does
not decide “ID card” vs “passport”, so no per-type accuracy is claimed and the
processing summary reports `documents_redacted` while `id_cards_found` stays 0
(see the limitations). Both frontend categories are real, independently
selectable, and run the same validated detector once per frame.

Before/after images:

```powershell
python scripts/render_document_redaction_comparison.py \
    --image .cache/docmodels/midv_val/000_img.jpg --prefix id_card
python scripts/render_document_redaction_comparison.py \
    --image .cache/docmodels/passport_samples/0.jpg --prefix passport --pad
```

Writes, into `backend/storage/validation/document_comparison/`:
`id_card_original.jpg`, `id_card_detected.jpg`, `id_card_gaussian.jpg`,
`id_card_pixelate.jpg`, `id_card_blackout.jpg` and the same five files for the
passport, plus `*_detections.json`. The passport sample is a public specimen,
not a real person's document; `--pad` places it on a neutral scene because the
corner model needs some background around a flat scan.

---

### 6.8 Laptop-screen detection validation (real annotated photos)

```powershell
python scripts/download_laptop_screen_model.py
python scripts/evaluate_laptop_screen_detection.py
python scripts/render_laptop_screen_redaction_comparison.py
```

The evaluator uses 8 openly licensed Wikimedia Commons photos listed in
`scripts/laptop_screen_annotations.json` (source page and licence per image).
Screen polygons and keyboard/deck rectangles were **manually annotated by the
implementer** on 1280-px thumbnails, so they are approximate (±10–20 px) and the
sample is small. They are not third-party ground truth, and no screen-boundary
precision/recall is claimed beyond the coverage figures below.

Measured on 2026-11-15, CPU, laptop score ≥ 0.35, production detect + mask path:

| Metric | Result |
| --- | ---: |
| Device-detection recall on photos with a visible display | 85.7 % (6 of 7) |
| Median display coverage by the rendered mask | 100.0 % |
| Coverage on detected screens (min / median) | 96.5 % / 100.0 % |
| Refined display quads / device-box fallbacks | 3 / 4 |
| Median quad IoU vs annotated polygon | 0.877 |
| Median keyboard/deck pixels changed | 0.2 % |
| Model inference (median) | 213 ms/frame |
| End-to-end detect + refine (median) | 252 ms/frame |

Four detected screens could not have their display separated reliably and were
masked as whole device boxes: privacy coverage is complete, but the keyboard is
over-covered and the summary says so (`laptop_screen_boundary`). An extreme
close-up of a display was not detected by the COCO `laptop` class at all — that
frame is not redacted, which is the main known limitation. Synthetic geometry
and privacy-invariant tests are in
`backend/tests/test_phase4_laptop_screens.py`; full model provenance, thresholds
and reproducibility commands are in
[docs/PHASE4_LAPTOP_SCREENS.md](docs/PHASE4_LAPTOP_SCREENS.md).

Comparison images are written to `docs/validation/laptop_screens/<photo>/`:
original, detected device box + refined display quad + applied mask silhouette,
and Gaussian, pixelate and blackout outputs. They are git-ignored and
regenerated in seconds.

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
3. Keep "Faces" selected and optionally enable **License Plates**, **ID Cards**,
   **Passports**, **Laptop Screens** or **Phone Screens** independently. Choose
   **Gaussian**, intensity ~80, optionally enable debug frames.
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
| `Start` returns 503 for license plates | Plate weights are missing or invalid. Run `python scripts/download_plate_model.py`; check `GET /api/v1/health/readiness` → `plate_detector`. |
| `Start` returns 503 for ID cards / passports | Document weights are missing or invalid. Run `python scripts/download_document_model.py`; check `GET /api/v1/health/readiness` → `document_detector`. |
| `Start` returns 503 for laptop screens | Laptop weights are missing/invalid, or onnxruntime is not installed. Run `python scripts/download_laptop_screen_model.py`; check `GET /api/v1/health/readiness` → `laptop_detector`. |
| `Start` returns 503 for phone screens | Phone weights are missing/invalid, or onnxruntime is not installed. Run `python scripts/download_phone_screen_model.py`; check `GET /api/v1/health/readiness` → `phone_detector`. |
| `Download` returns 409 | The task is not completed and verified yet, or it failed. Check the status endpoint / Processing page. |
| Video plays in the browser but not in Windows Media Player | Output is H.264 + AAC in MP4; use a current player. Browser playback needs H.264 support. |
| `WebSocket telemetry: DISCONNECTED` | The page still polls REST status; check the backend is on port 8000 and CORS allows `http://localhost:5173`. |
| Port already in use | `uvicorn ... --port 8001` plus `VITE_API_BASE_URL=http://localhost:8001/api/v1 npm run dev`. |
| Task stuck at 0 % | Look at the Processing page log stream and the backend console; a decode or encoder failure is reported there and in `failure_reason`. |
| `python download_yunet.py` checksum error | The download was modified. Delete the `.onnx` file and re-run; the script only accepts the pinned SHA-256. |
| `download_plate_model.py` checksum error | Same policy for the plate graph: delete `plate_detection_rtdetr_r18.onnx` and re-run; only the pinned SHA-256 is accepted. |
| Plate jobs are slow | RT-DETR plate inference is ~0.3 s/frame on CPU (section 6.2), so a plate-enabled job is several times slower than face-only. That is measured, not a bug. |
| Document jobs are slow | Document inference runs a full frame plus four half-strips: ~0.87 s/frame for 1080×1920 on CPU (section 6.7). A document-enabled job is the slowest configuration. |
| Laptop-screen jobs are slow | RT-DETR R18 plus quad refinement is ~0.25 s/frame on CPU (section 6.8); there is no GPU path in this build. |
| Laptop summary says `device_box_fallback` / `mixed` | The display boundary could not be separated from the keyboard deck for some frames, so those frames masked the whole device box. Privacy is preserved; this is over-coverage, not a silent failure. |
| A flat scan with no background is not detected | The corner model expects some scene around the document, and boxes covering >90 % of the frame are filtered to avoid redacting the entire frame on a bad prediction. Place the document on a surface with visible edges, as in normal video. |
| `download_document_model.py` checksum error | Delete `document_detection_docaligner_fastvit.onnx` and re-run; only the pinned SHA-256 is accepted. |

---

## 10. Known limitations

- **The laptop model detects devices, not displays.** RT-DETR boxes the whole
  laptop; the display quadrilateral is estimated from edges. On 4 of 6 detected
  screens in the validation set the boundary was not separable and the whole
  device box was masked, which preserves privacy but also covers the keyboard.
  The summary reports `laptop_screen_boundary` so this is visible.
- **Undetected laptops are not redacted.** A display filling the frame (extreme
  close-up), a heavily occluded device or an unusual form factor can be missed by
  the COCO `laptop` class; the extreme close-up frame in the validation set was
  missed. Dark/low-contrast or hand-occluded displays usually fall back to the
  device box rather than producing a tight quad.
- **Documents are localised but not classified.** The DocAligner graph is
  class-agnostic, so the backend cannot truthfully report “3 ID cards, 1
  passport”. Both categories are independently selectable and redact the whole
  document, and the summary reports `documents_redacted`; `id_cards_found`
  remains 0 rather than inventing type counts. OCR-based field-level redaction
  is deliberately not implemented, so the whole document is always covered.
- **Document detection is not perfect.** On 60 real MIDV frames, mask recall at
  IoU ≥ 0.50 was 93.3 % (56/60) with mean IoU 0.98. Four frames were missed
  entirely (all low-quality/edge cases), and 5 of 40 non-document vehicle
  photos produced a plate-like false detection. False positives over-redact,
  which is the safe direction, but recall is not 100 %. Inference is ~0.87 s per
  1080×1920 frame because of the multi-strip scan.
- **Full-frame documents are filtered.** A quad covering more than 90 % of the
  frame is rejected to avoid redacting an entire video on one bad prediction;
  a document held edge-to-edge to the camera can therefore be missed. The
  document mask is an axis-aligned rectangle over the quadrilateral, so a very
  steep perspective can leave a thin corner sliver; 6 % padding covers the
  material tested, but this is geometry, not a guarantee.
- **Partially visible documents are not guaranteed.** The decoder can rebuild
  one corner that falls outside the frame, and a real-pixel rotation/small-scale
  test passes, but a document cropped by roughly a third in a real photo was not
  detected in manual testing. In that case the exposed part is not masked; this
  is a recall limitation, not a silent failure (the task still completes).
- **Plate detection is not perfect and is slower.** On the 40-photo sample it
  missed 4 of 44 plates and produced 4 false positives; small/distant recall was
  0.80. The model is trained on the Open Images plate class and has no region
  guarantee (it may miss unusual formats or heavily angled/occluded plates).
  False positives only over-redact vehicle regions, which is the safe direction,
  but recall is not 100 %. Inference is ~0.3 s/frame on CPU.
- **Plate confidence scores are compressed.** The RT-DETR export emits ~0.10–0.20
  for confident plates, so the threshold is 0.10. Raising it to face-like values
  would silently drop plates; do not tune it without re-running the evaluator.
- **Plate masks are rectangular.** They cover the padded plate box completely,
  but they are not character segmentation: the mask intentionally covers the
  whole plate, and a heavily rotated plate may need slightly more padding than
  an axis-aligned rectangle provides.
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
  services/ai/                model manager, detection, document detection,
                              tracking, blur, pipeline
  utils/                      FFmpeg/ffprobe discovery, storage paths
  workers/task_manager.py     in-process task registry
  tests/                      pytest suites
scripts/
  download_yunet.py           pinned + checksum-verified face model download
  download_plate_model.py     pinned + checksum-verified plate model download
  make_validation_videos.py   builds the composite validation fixtures
  evaluate_redaction.py       runs the real face pipeline and measures it
  evaluate_plate_detection.py measures plate precision/recall on annotated photos
  download_document_model.py  pinned + checksum-verified document model download
  evaluate_document_detection.py measures document mask recall/IoU on MIDV frames
  render_redaction_comparison.py  renders face before/after panels
  render_plate_redaction_comparison.py  renders plate before/after images
  render_document_redaction_comparison.py  renders ID-card/passport before/after images
  download_laptop_screen_model.py  pinned + checksum-verified laptop detector download
  evaluate_laptop_screen_detection.py measures laptop detection and display coverage
  render_laptop_screen_redaction_comparison.py  renders laptop-screen comparison images
  laptop_screen_annotations.json  annotated real-photo validation set + licences
docs/
  PHASE4_LAPTOP_SCREENS.md    Phase 4 model provenance, methodology, results
  validation/laptop_screens/  generated comparison images (git-ignored)
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

Face, license-plate, identity-document and laptop-screen redaction are privacy
tools, but this
implementation is a student project: it can miss targets and it has no audit
trail, authentication or encryption at rest. Do not present its output as
certified anonymization, and do not process recordings of people who have not
consented to the demonstration.
