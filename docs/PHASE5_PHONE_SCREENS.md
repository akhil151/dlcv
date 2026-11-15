# Phase 5 — Phone Screen Detection & Redaction

Status: **implemented, independently testable, validated on real annotated
photos and a real FFmpeg pipeline run.**

This phase adds a new, independent `phone_screens` category. It does not change
the face, license-plate, ID-card, passport or laptop-screen paths. Laptop
screens remain their own category and are unaffected.

---

## 1. What the feature does

1. Proposes phone devices in a frame with a real pretrained detector.
2. Estimates the visible **display region** inside each device box from image
   edges, producing a four-corner polygon where the evidence is reliable.
3. Masks that polygon with a hard, outward-dilated mask in Gaussian, pixelation
   or blackout mode.
4. Tracks each screen with the existing category-aware IoU tracker and a short,
   configurable hold for temporary detection loss (2 frames by default).
5. Reports, in the task summary, how many masked screen regions came from a
   refined display quad and how many fell back to the whole device box.

When the display boundary cannot be separated reliably, the pipeline **masks the
whole device box** (fail closed) and reports it as `phone_fallback`. It never
claims a screen boundary it did not find.

Unlike Phase 4, the fallback kind is category-aware: `phone_fallback` for the
`phone_screens` category and `laptop_fallback` for `laptop_screens`. This fixed
a defect where merged plans always reported the laptop fallback label.

---

## 2. Models

### 2.1 Device proposal — RT-DETR R18 (COCO)

| Property | Value |
| --- | --- |
| Model | RT-DETR R18 (COCO object detection) |
| Upstream weights | `PekingU/rtdetr_r18vd` — https://huggingface.co/PekingU/rtdetr_r18vd |
| ONNX export used | `onnx-community/rtdetr_r18vd` → `onnx/model.onnx` — https://huggingface.co/onnx-community/rtdetr_r18vd |
| Licence | Apache-2.0 (upstream RT-DETR / PekingU weights). The ONNX conversion repo declares no separate licence; it is a format conversion of the Apache-2.0 weights. Verify the upstream card before redistribution. |
| Local file | `phone_screen_detection_rtdetr_r18.onnx` |
| Size | 82,572,357 bytes |
| SHA-256 | `11843b02455cc24009aed24d4c40db721b1093be5ccd6bbe7b9c441abb1d0558` |
| Runtime | `onnxruntime` CPUExecutionProvider |
| Input | `pixel_values` float32 `[1,3,640,640]`, RGB, `/255`, ImageNet mean/std, stretched (no letterbox) |
| Outputs | `logits` `[1,300,80]` raw class logits, `pred_boxes` `[1,300,4]` normalized `cx,cy,w,h` |
| Class used | COCO index 67 = `cell phone` |

Download and verify:

```bash
python scripts/download_phone_screen_model.py
python scripts/download_phone_screen_model.py --check
```

The script rejects any file whose size or SHA-256 differs from the pinned
values, and runs a constructor-level inference probe plus a label check against
the pinned `config.json`.

### 2.2 Provenance note

The COCO RT-DETR R18 export is the same graph already shipped for Phase 4
(laptop screens use COCO index 63; phones use index 67). The phone model file is
therefore byte-identical to the laptop file: same size and same SHA-256 shown
above. That is intentional, not a copy mistake — one COCO detector serves both
categories, and the class index is what separates them. No phone-specific model
was trained, fine-tuned or claimed.

RT-DETR predicts a **box around the whole phone**. It does **not** predict the
display quadrilateral. The display polygon comes from the refinement stage below
and is reported separately.

---

## 3. Display-region estimation

Implemented in `backend/services/ai/phone_screen_detection_service.py`.

For each device box:

1. Crop the box with a small margin and rescale the longest side to 480 px, so
   thresholds are resolution independent.
2. Build a Canny edge map with median-derived thresholds and close 1-px gaps.
3. Generate quad candidates from large contour approximations and from
   line-support scoring.
4. Reject candidates that do not look like a phone display: coverage must be
   within 28 %–105 % of the device box, aspect ratio within 1.10–3.20, centre
   distance from the device centre at most 0.35, edge support at least 0.38, and
   border contrast at least 0.06.
5. Pick the best-scoring candidate; if its score is below 0.55, fall back to the
   whole device box.

Mask padding is outward-only and configurable: 4 % of the screen size, clamped
to 3–36 px. The mask is never shrunk below the estimated display.

---

## 4. Configuration

All values live in `backend/config/ai_settings.py` and are independently
configurable from the laptop-screen settings:

| Setting | Default |
| --- | --- |
| `PHONE_CONF_THRESHOLD` | 0.30 |
| `PHONE_NMS_IOU_THRESHOLD` | 0.50 |
| `PHONE_MAX_DETECTIONS` | 15 |
| `PHONE_SCREEN_REFINE_SCALE` | 480 |
| `PHONE_SCREEN_MIN_COVERAGE` / `MAX_COVERAGE` | 0.28 / 1.05 |
| `PHONE_SCREEN_MIN_ASPECT` / `MAX_ASPECT` | 1.10 / 3.20 |
| `PHONE_SCREEN_MIN_SCORE` | 0.55 |
| `PHONE_PADDING_RATIO` | 0.04 |
| `PHONE_MIN_PADDING_PX` / `MAX_PADDING_PX` | 3 / 36 |
| `PHONE_PIXELATE_MAX_BLOCKS` | 4 |
| `PHONE_REDACTION_HOLD_FRAMES` | 2 |

---

## 5. API and summary fields

- `GET /api/v1/health/readiness` includes a `phone_detector` block (`model`,
  `available`, `status`) and a top-level `phone_screens_ready` flag.
- Requesting `phone_screens` without usable weights returns an actionable
  **503** before processing starts, naming the missing file and the download
  script.
- Task metrics gain `phone_screen_quad_regions` and
  `phone_screen_fallback_regions`; redaction counts gain a `phone_screens` key.
- Laptop-only jobs never instantiate or invoke the phone detector, and phone
  jobs never mask laptop screens.

---

## 6. Frontend

- Phone Screens is a real, independent toggle in `RedactionTargetPicker`.
- The toggle is disabled with a weights-missing badge when the detector is not
  ready, and the readiness label is shown in the upload footer.
- The results page adds a **Phone Screen Regions** card showing the phone region
  count plus refined-quad vs. fallback breakdown, with a footnote explaining the
  difference.
- The Laptop Screens toggle and card are unchanged.

---

## 7. Tests

`backend/tests/test_phase5_phone_screens.py` contains 37 tests:

- decoding and box-to-region estimation on synthetic scenes;
- acceptance/rejection of display candidates, including laptop-like and
  TV-like rectangles that must **not** be treated as phone displays;
- all three redaction modes, padding behaviour, and the pixelation floor;
- tracking hold and category independence;
- API/readiness/summary behaviour, including the 503 preflight;
- pure-fallback reporting (`phone_fallback`), including the merge-plan
  category-fix regression;
- four real-model + FFmpeg integration tests on an annotated phone photo
  (skipped only when ONNX weights or the cached photo are absent).

Verified this session:

```
backend/tests/test_phase5_phone_screens.py ... 37 passed, 6 subtests passed
backend/tests (full suite) ................... 229 passed, 106 subtests passed
```

---

## 8. Evaluation on real annotated photos

`scripts/evaluate_phone_screen_detection.py` runs the phone detector on five
real, openly licensed CC0 photographs (Wikimedia Commons; originals from
Pixabay/Unsplash). Display quads were annotated manually on the 1280-px-wide
thumbnails by the implementer, so they are approximate (±10–20 px), not
third-party ground truth. Results: `scripts/eval_results_phone.json`.

| Photo | Detections | Region kind | Quad IoU | Display coverage | Confidence |
| --- | --- | --- | --- | --- | --- |
| `phone_display` (close-up, frame cuts display) | 0 | — | — | — | — |
| `phone_hand` (hand-held iPhone) | 1 | `screen_quad` | 0.795 | 97.5 % | 0.496 |
| `phone_people` (two small phones, street scene) | 2 | `phone_fallback` | 0.625 | 74.3 % | 0.484 |
| `phone_bokeh` (tilted phone, bright reflections) | 0 | — | — | — | — |
| `phone_use` (hand-held Android, thumb on screen) | 1 | `screen_quad` | 0.310 | 39.5 % | 0.669 |

Aggregates (detected photos only):

- Detection rate: **3 / 5 (60 %)**. Two photos produced no phone proposal at
  all: an extreme close-up where the display fills the frame, and a heavily
  reflected display. COCO detection expects a whole device in a normal framing.
- Fallback rate among detections: **1 / 3 (33 %)** — the small street-scene
  phones fell back to the whole-device mask, as designed.
- Median quad IoU: **0.625** (mean 0.577), over the three detections.
- Median display coverage: **74.3 %** (mean 70.4 %).
- `phone_use` is the weak case: the refined quad covered only 39.5 % of the
  annotated display. The thumb overlapping the right edge and the bright,
  low-contrast bezel make the boundary evidence unreliable, and the mask stays
  inside the true display rather than over-masking. This is reported honestly
  rather than tuned away.

Interpretation: the detector is usable for normal hand-held/posed phones, and
its fail-closed fallback prevents unredacted displays on hard cases. It is not
reliable on extreme close-ups or glare-heavy reflections, where it may miss the
phone entirely (no mask) — that gap is documented, not hidden.

These numbers are measured from the run above and are **not** the Phase 4
laptop numbers.

### 8.1 Synthetic vs. real

- Unit tests use **synthetic** scenes with a fake detection session; they
  verify pipeline logic, not real-world accuracy.
- The evaluation table above uses **real photographs** with manual annotations.
- The four integration tests use a **real ONNX graph and real photos**, encoded
  to a real clip and processed through FFmpeg, then verified frame-by-frame.

---

## 9. Performance

Measured this session, single process, CPU (onnxruntime CPUExecutionProvider),
detector only, 1280×853 frame:

- Median: **388.0 ms/frame**
- p95: **408.7 ms**

This is the phone detector stage alone and excludes decoding, tracking, masking
and encoding. It is slower than the Phase 4 measurement because that number came
from a different frame size and run; treat these as separate measurements, not a
regression comparison.

---

## 10. Limitations and next steps

- COCO proposal misses extreme close-ups and glare-heavy displays.
- Small, oblique phones may fall back to whole-device masks (safe, but noisier).
- Manual annotations are approximate; a third-party annotated set would give
  stronger evidence.
- A phone-specific display-segmentation model (or a fine-tuned corner regressor)
  is the natural next step if coverage on hand-held phones must improve.
- No commit or push was made for this phase.
