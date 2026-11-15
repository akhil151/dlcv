# Phase 4 — Laptop Screen Detection & Redaction

Status: **implemented, independently testable, validated on real annotated photos and a real FFmpeg pipeline run.**

This phase adds a new, independent `laptop_screens` category. It does not change
the face, license-plate, ID-card or passport paths. Phone screens remain
**not implemented** and are still refused by the API.

---

## 1. What the feature does

1. Proposes laptop devices in a frame with a real pretrained detector.
2. Estimates the visible **display region** inside each device box from image
   edges, producing a four-corner polygon where the evidence is reliable.
3. Masks that polygon with a hard, outward-dilated mask in Gaussian, pixelation
   or blackout mode.
4. Tracks each screen with the existing category-aware IoU tracker and a short,
   configurable hold for temporary detection loss.
5. Reports, in the task summary, how many masked screen regions came from a
   refined display quad and how many fell back to the whole device box.

When the display boundary cannot be separated reliably, the pipeline **masks the
whole device box** (fail closed) and reports `device_box_fallback`. It never
claims a screen boundary it did not find. This is the required safe behaviour for
a bounding-box-only detector.

---

## 2. Models

### 2.1 Device proposal — RT-DETR R18 (COCO)

| Property | Value |
| --- | --- |
| Model | RT-DETR R18 (COCO object detection) |
| Upstream weights | `PekingU/rtdetr_r18vd` — https://huggingface.co/PekingU/rtdetr_r18vd |
| ONNX export used | `onnx-community/rtdetr_r18vd` → `onnx/model.onnx` — https://huggingface.co/onnx-community/rtdetr_r18vd |
| Licence | Apache-2.0 (upstream RT-DETR / PekingU weights). The ONNX conversion repo declares no separate licence; it is a format conversion of the Apache-2.0 weights. Verify the upstream card before redistribution. |
| Local file | `laptop_screen_detection_rtdetr_r18.onnx` |
| Size | 82,572,357 bytes |
| SHA-256 | `11843b02455cc24009aed24d4c40db721b1093be5ccd6bbe7b9c441abb1d0558` |
| Runtime | `onnxruntime` CPUExecutionProvider (added to `backend/requirements.txt`) |
| Input | `pixel_values` float32 `[1,3,640,640]`, RGB, `/255`, ImageNet mean/std, stretched (no letterbox) |
| Outputs | `logits` `[1,300,80]` raw class logits, `pred_boxes` `[1,300,4]` normalized `cx,cy,w,h` |
| Class used | COCO index 63 = `laptop` (verified against the pinned `config.json`) |

Download and verify:

```bash
python scripts/download_laptop_screen_model.py
python scripts/download_laptop_screen_model.py --check
```

The script rejects any file whose size or SHA-256 differs from the pinned
values, and runs a constructor-level inference probe plus a label check against
the pinned `config.json`.

### 2.2 What the model does and does not provide

RT-DETR predicts a **box around the whole laptop** (lid + keyboard deck). It does
**not** predict the display quadrilateral. Phase 4 therefore does not claim
model-based screen-corner detection. The display polygon comes from the
refinement stage below and is reported separately.

No model was trained from scratch, and no dataset was created for this phase.

---

## 3. Display-region estimation

Implemented in `backend/services/ai/laptop_screen_detection_service.py`.

For each device box:

1. Crop the box with an 8 % margin and rescale the longest side to 480 px, so
   thresholds are resolution independent.
2. Build a Canny edge map with median-derived thresholds and close 1-px gaps.
3. Generate quad candidates two ways:
   - contour approximation (`approxPolyDP`) of large contours;
   - intersections of near-horizontal and near-vertical Hough segments, which
     can represent a perspective-distorted display that is not a rectangle.
4. Reject candidates that fail **all** of these gates:
   - convex, 4 corners, fully inside the crop;
   - coverage of the device box between 18 % and 105 %;
   - aspect ratio between 1.00 and 2.75;
   - top edge within 13 % of the device-box top;
   - bottom edge between 30 % and 70 % of the device-box height (the lid region,
     not the keyboard deck);
   - the bottom edge is corroborated by a detected line spanning at least 55 %
     of that edge (this is what rejects an inner window/dialog border);
   - perimeter edge support ≥ 0.70 and border luminance contrast ≥ 0.18.
5. Score survivors as `0.52·edge_support + 0.28·contrast + 0.20·aspect_fit` and
   accept only the best if it scores ≥ 0.80.

If no candidate passes, `region_kind = "laptop_fallback"` and the device box is
masked instead. Every threshold lives in `backend/config/ai_settings.py`
(`LAPTOP_SCREEN_*`), so it is configurable without code changes.

---

## 4. Tracking

Reuses the single existing tracker (`backend/services/ai/tracking_service.py`):

- category-aware association on the screen box, unchanged IoU + velocity logic;
- its own confidence floor `LAPTOP_CONF_THRESHOLD = 0.35` (COCO sigmoid score),
  which cannot affect the face (0.60), plate (0.10) or document (0.30) floors;
- label `SCREEN #<id>`;
- the four polygon corners travel with the trajectory and are only updated on a
  real detection match, so a temporarily lost screen keeps its last observed
  geometry;
- `LAPTOP_REDACTION_HOLD_FRAMES = 2` bridges a short detector gap without moving
  or growing the box; after the hold the mask is dropped;
- multiple laptops are associated independently.

---

## 5. Redaction

`BlurService` gains a polygon mask path:

- the ROI is the polygon's bounding box plus padding;
- padding is 4 % of the screen's own size (min 4 px, max 48 px) and is applied by
  **dilating the polygon outward only** — the mask can never shrink below the
  detected display;
- the mask is hard (alpha 1.0) in every mode, including Gaussian: a feathered
  screen edge could leave readable content just inside the border;
- a detection and its own track are merged into the convex hull of both
  polygons, so the same screen is never filtered twice and a smoothed box cannot
  cut a corner off the detection;
- pixelation reuses the 4-block cap used for plates/documents, because screen
  content is high-contrast text/UI;
- the keyboard, trackpad and body are untouched whenever a refined quad is used;
  a device-box fallback covers them by design and is reported.

---

## 6. API and summary

- Category identifier: `laptop_screens` (`RedactionCategoryEnum`, `TargetCategory`).
- Detector source: `rtdetr_laptop`.
- `POST /api/v1/redact/start` accepts it; a missing/invalid model returns `503`
  with an actionable message before a task is created.
- `GET /api/v1/health/readiness` adds `laptop_detector` and
  `laptop_screens_ready`.
- `GET /api/v1/redact/results/{task_id}` adds:
  - `laptop_screens_redacted` — masked screen regions across frames;
  - `laptop_screen_quad_regions` — regions masked with a refined display quad;
  - `laptop_screen_fallback_regions` — regions masked with the whole device box;
  - `laptop_screen_quad_pct` — share of regions that used a refined quad;
  - `laptop_screen_boundary` — `refined_quad` | `device_box_fallback` | `mixed` |
    `not_applicable`.
- Existing fields are unchanged; `screens_hidden` continues to mean
  laptop + phone region-frames (phone stays 0).

Frontend: the **Laptop Screens** toggle is independent, shows a “Weights Missing”
badge and disables itself when readiness reports the detector unavailable, and
the results grid shows a Laptop Screen Regions card with the quad/fallback split.
**Phone Screens stays “Coming Later”** and is never sent to the backend.

---

## 7. Validation

### 7.1 Real, annotated photographs (not synthetic)

Dataset: 8 openly licensed Wikimedia Commons photos listed in
`scripts/laptop_screen_annotations.json` (CC0 / CC BY 2.0 / CC BY-SA 4.0), with
the source page and licence recorded per image. Screen polygons and
keyboard/deck rectangles were **manually annotated by the implementer** on
1280-px thumbnails; they are approximate (±10–20 px), small-sample, and not
third-party ground truth. This is stated in the JSON and the report.

Run:

```bash
python scripts/download_laptop_screen_model.py
python scripts/evaluate_laptop_screen_detection.py
```

Measured on 2026-11-15, RT-DETR CPU, `laptop` score ≥ 0.35, full production
detection + mask path:

| Metric | Result |
| --- | ---: |
| Images evaluated | 8 (7 with a visible display, 1 with the lid facing away) |
| Device-detection recall on visible-display photos | 85.7 % (6/7) |
| Median annotated-display coverage by the rendered mask | **100.0 %** |
| Mean annotated-display coverage | 84.9 % |
| Minimum annotated-display coverage | 0.0 % (the extreme close-up frame, undetected) |
| Coverage on detected screens (min / median) | 96.5 % / 100.0 % |
| Refined display quads / device-box fallbacks | 3 / 4 |
| Median IoU of refined quad vs annotated polygon | 0.877 |
| Median keyboard/deck pixels changed by the mask | 0.2 % |
| Model-only inference time (median) | 213.4 ms/frame (CPU) |
| End-to-end detect + refine time (median) | 252.1 ms/frame (CPU) |

Per image:

| Image | Detections | Region kind | Display coverage | Keyboard changed |
| --- | ---: | --- | ---: | ---: |
| lenovo | 1 | screen_quad | 97.8 % | 0.2 % |
| msi_wiki | 1 | screen_quad | 96.5 % | 0.0 % |
| charts_desk | 2 | fallback ×2 | 100 % | 100 % (fallback) |
| surface7 | 1 | screen_quad | 100 % | 0.1 % |
| wikicamp_screen | 1 | fallback | 100 % | 85.3 % (fallback) |
| dual_touch | 1 | fallback | 100 % | 100 % (fallback) |
| oscillogram (close-up) | 0 | — | 0 % | 0 % |
| cat_laptop (lid away) | 0 | — | n/a | n/a |

Interpretation, stated plainly:

- Refined quads are **accurate but conservative**: two photos are 96–98 %
  covered against a hand annotation, not 100 %.
- Four of the six detected screens could not have their display separated
  reliably and were masked as whole device boxes. Privacy coverage is then
  complete, but the keyboard/deck is over-covered. The summary reports this as
  `mixed` / `device_box_fallback`.
- The extreme close-up of a display fills the frame and is not detected by the
  COCO laptop class; that frame is **not** redacted. This is a false negative and
  is the most important known limitation.
- No precision/recall for screen *boundaries* is claimed beyond these coverage
  figures; the detector does not output screen corners.

### 7.2 Synthetic geometry tests (clearly synthetic)

`backend/tests/test_phase4_laptop_screens.py` draws laptop scenes (front-facing,
perspective, dark, glare, inner-window) and asserts:

- a clear front-facing and a perspective display are refined with IoU ≥ 0.5;
- a dark/off display either refines correctly or is reported as a fallback — it
  is still covered;
- an inner window is never accepted as the full display (covered by the
  privacy invariant);
- a refined quad never escapes the device box;
- zero annotated display pixels survive redaction in all three modes;
- the keyboard/deck stays unchanged under a refined quad;
- multiple screens are independent, detection + track merge once, and no stale
  quad masks after the hold window.

These synthetic tests verify geometry and privacy invariants. They are **not**
accuracy measurements and are labelled as such.

### 7.3 Video / pipeline tests (real detector, real FFmpeg)

Integration tests stage a 10-frame H.264/AAC clip built from the annotated
`lenovo` photo and run the production `process_video_pipeline`:

- task completes with `output_verified = true` and the exact frame count;
- display content is darkened in the decoded output (≥ 95 % of the annotated
  display polygon);
- all three redaction modes complete and verify;
- audio stream and resolution are preserved;
- face + plate + ID-card + laptop combined jobs still complete and still mask
  laptop screens;
- face-only jobs produce no laptop counts (no regression).

Video-behaviour coverage in the unit tests: moving screen, changing size,
temporary detection loss, multiple independent screens, and a 60 fps
frame-index sequence through the tracker.

---

## 8. Performance

Measured on this checkout (CPU only; no CUDA/TensorRT path is active):

| Measurement | Value |
| --- | --- |
| Model load (onnxruntime session + constructor probe) | 601 ms measured cold (495 ms recorded inside the loader); once per process, only when a laptop job runs |
| Model inference | 187–240 ms/frame at 640×640 (median 213.4 ms over 8 photos) |
| Display refinement | median 38.6 ms/frame, measured 0–546 ms across the 8 validation photos (busy crops generate more line pairs) |
| End-to-end detect + refine | median 252–264 ms/frame over the validation photos (range 194–989 ms) |
| Production pipeline detection stage | 386.6 ms/frame on the 960×720 end-to-end clip (includes tracker-independent per-frame detection bookkeeping) |
| Memory | not instrumented; an `onnxruntime` CPU session for an 82.6 MB fp32 graph, and no extra full-frame decode or copy is introduced |

The detector is lazily initialized and cached by `ModelManager`, exactly like the
plate and document detectors; it is never loaded for face/plate/document jobs and
never re-initialized per frame.

---

## 9. Known limitations

1. **The model detects devices, not displays.** Screen corners are estimated
   geometrically. Where the estimate is not confident the whole device box is
   masked and reported; that over-covers the keyboard and deck.
2. **Undetected devices are not redacted.** A close-up display that fills the
   frame, a heavily occluded laptop, or an unusual form factor can be missed by
   the COCO `laptop` class. The task still completes; no mask is applied to the
   missed device.
3. **Dark, low-contrast displays often fall back.** A switched-off screen with a
   dark bezel frequently cannot be separated from the body, so the device box is
   used (privacy-safe, over-covering).
4. **Occluded displays may be under-covered.** The estimate only sees visible
   edges; a hand or mug across the display can truncate the accepted quad. The
   gated fallback reduces but cannot eliminate this.
5. **No screen-boundary precision/recall is claimed.** The 0.877 median IoU and
   96–100 % coverage are measured against small, hand-made annotations.
6. **CPU only.** ~0.25 s/frame makes laptop jobs several times slower than
   face-only; no GPU path is implemented or claimed.
7. **Tablets and phones are not included.** Phone screens remain unimplemented
   (Phase 5 is explicitly out of scope). A tablet may be detected as a laptop;
   that over-redacts rather than leaking.
8. **Fallback masks are device-shaped.** They are privacy-safe but not minimal;
   the summary exposes the fallback counts so this is visible to the user.

---

## 10. Reproduce

```bash
# 0. Dependencies (from the project root; activate .venv first)
pip install -r backend/requirements.txt

# 1. Pinned model
python scripts/download_laptop_screen_model.py
python scripts/download_laptop_screen_model.py --check

# 2. Response/geometry/redaction/tracking/API/pipeline tests
python -m pytest backend/tests/test_phase4_laptop_screens.py -q

# 3. Full backend regression suite
python -m pytest backend/tests -q

# 4. Real annotated-photo evaluation (downloads CC-licensed photos on first run)
python scripts/evaluate_laptop_screen_detection.py

# 5. Comparison images
python scripts/render_laptop_screen_redaction_comparison.py

# 6. Frontend
npm run lint
npm run build

# 7. Optional HTTP smoke test against a running backend
uvicorn backend.main:app --port 8000
python test_script.py
```

---

## 11. Comparison images

Generated by step 5 above (git-ignored PNGs, regenerated in seconds):

```
docs/validation/laptop_screens/lenovo/01_original.png
docs/validation/laptop_screens/lenovo/02_detected_boundary.png
docs/validation/laptop_screens/lenovo/03_gaussian.png
docs/validation/laptop_screens/lenovo/04_pixelate.png
docs/validation/laptop_screens/lenovo/05_blackout.png
docs/validation/laptop_screens/msi_wiki/01_original.png
docs/validation/laptop_screens/msi_wiki/02_detected_boundary.png
docs/validation/laptop_screens/msi_wiki/03_gaussian.png
docs/validation/laptop_screens/msi_wiki/04_pixelate.png
docs/validation/laptop_screens/msi_wiki/05_blackout.png
docs/validation/laptop_screens/charts_desk/01_original.png
docs/validation/laptop_screens/charts_desk/02_detected_boundary.png
docs/validation/laptop_screens/charts_desk/03_gaussian.png
docs/validation/laptop_screens/charts_desk/04_pixelate.png
docs/validation/laptop_screens/charts_desk/05_blackout.png
docs/validation/laptop_screens/surface7/01_original.png
docs/validation/laptop_screens/surface7/02_detected_boundary.png
docs/validation/laptop_screens/surface7/03_gaussian.png
docs/validation/laptop_screens/surface7/04_pixelate.png
docs/validation/laptop_screens/surface7/05_blackout.png
```

`02_detected_boundary.png` shows the device box (green), the refined display quad
(magenta) and the exact applied mask silhouette (red). `charts_desk` is a
device-box fallback and is labelled as such in the image.

---

## 12. Files

Created:

- `backend/services/ai/laptop_screen_detection_service.py`
- `backend/tests/test_phase4_laptop_screens.py`
- `scripts/download_laptop_screen_model.py`
- `scripts/evaluate_laptop_screen_detection.py`
- `scripts/render_laptop_screen_redaction_comparison.py`
- `scripts/laptop_screen_annotations.json`
- `docs/PHASE4_LAPTOP_SCREENS.md`

Modified:

- `backend/config/ai_settings.py` — laptop model/threshold/padding settings
- `backend/services/ai/model_manager.py` — lazy ONNX load, status, readiness
- `backend/services/ai/detection_service.py` — category wiring and preflight
- `backend/services/ai/tracking_service.py` — per-category floor, quad carry
- `backend/services/ai/blur_service.py` — polygon mask, merge, padding
- `backend/services/ai/video_processor_service.py` — region accounting, debug quad
- `backend/schemas/detection.py`, `tracking.py`, `blur.py`, `redact.py`
- `backend/services/redact_service.py` — screen summary fields
- `backend/services/readiness_service.py` — laptop readiness
- `backend/requirements.txt` — `onnxruntime`
- `src/types/index.ts`, `src/services/api.ts`,
  `src/components/upload/RedactionTargetPicker.tsx`, `src/pages/UploadPage.tsx`,
  `src/components/results/SummaryBentoGrid.tsx`
- `backend/tests/test_phase7a_privacy.py`, `backend/tests/test_phase9_license_plates.py`
  — these asserted laptop screens were unsupported; they now assert phone
  screens remain refused (laptop screens are validated in this phase)
- `test_script.py` — refusal check moved to `phone_screens`, laptop acceptance added
- `README.md`
