# SafeFrame AI Engineering Audit

**Audit date:** 2026-09-29
**Scope:** Read-only code/runtime audit of the repository as found. No application source, configuration, API contract, model weight, or existing media file was changed. This report is the only audit artifact created.

## 1. Executive Summary

The reported 95% completion is not substantiated. The repository contains a usable React/Vite presentation layer and a FastAPI-shaped backend, but the privacy-critical implementation is not production-ready and cannot be considered end-to-end verified.

**Estimated completion:** **35% overall; under 10% of the safety case required for production privacy redaction is verified.** The estimate credits the compiled frontend, routing, upload request wiring, schemas, a real OpenCV/FFmpeg-oriented pipeline shape, and a decodable pre-existing output. It does not credit functions that return fixed telemetry, detections, evaluation scores, or completion claims; fallbacks that conceal missing models/input; or unexecuted AI/video behavior.

Critical blockers are: (1) detector and Re-ID fallbacks fabricate outputs, (2) a real detection is intentionally left unblurred until its second hit, (3) task completion does not require a valid encoder result, (4) arbitrary/missing video IDs create synthetic successful jobs, (5) results and dashboard metrics are fabricated, and (6) task/pipeline state is process-local and shared between concurrent jobs.

| Area | Evidence-based state |
|---|---|
| Frontend compilation | VERIFIED: production build and lint pass. |
| Browser UI interaction | IMPLEMENTED BUT UNVERIFIED: no browser E2E suite or manual browser run was present/executed. |
| API health | NOT VERIFIED: importing the app did not complete within 60 seconds because model initialization occurs during import. |
| Real model inference | NOT VERIFIED. YOLO weights are present, but startup/inference did not complete in the safe test window. |
| Video output decodability | PARTIALLY VERIFIED: a pre-existing output opens and reports 150 frames at 640x480; provenance and redaction quality were not established. |
| Privacy redaction correctness | FAILING as a safety claim: confirmed code paths can export/proclaim success without reliable redaction. |

The working tree was already dirty before audit actions (source edits, cached bytecode, storage and test media). Those changes were not attributed to or modified by this audit.

## 2. Architecture Audit

### Actual architecture

Frontend entry: `src/main.tsx:6`; application providers/router: `src/App.tsx:6-20`; browser routes: `src/routes/index.tsx:8-36`; Zustand store: `src/store/useAppStore.ts:53-99`.

Backend entry: `backend/main.py`. It registers `/api/v1` routes and a WebSocket at `/ws/redact/{task_id}`. API endpoint modules are `health.py`, `videos.py`, `redact.py`, and `dashboard.py`. Persistent storage is only directories beneath `backend/storage`; tasks are held in `TaskManager.tasks`, an in-memory dictionary (`backend/workers/task_manager.py:10-13`).

Intended pipeline wiring is present in `backend/services/ai/video_processor_service.py:122-210`: OpenCV decode -> detector -> spatial tracker -> Re-ID invocation -> ghost prediction -> blur -> FFmpeg/OpenCV encode -> WebSocket. This is structural wiring, not proof of effective redaction.

### Structural gaps

- Importing any `backend.services.ai.*` submodule first executes `backend/services/ai/__init__.py:1`, which imports and initializes `model_manager`; this makes even isolated blur tests depend on all model setup.
- The claimed BoT-SORT instance is initialized opportunistically but never called; `TrackingService.update_tracks` uses custom greedy IoU matching (`tracking_service.py:119-225`).
- Re-ID returns matches but the pipeline does not apply a match to tracker IDs or tracks (`reid_service.py:122-206`, `video_processor_service.py:154`): it is observational only.
- The global tracker is reset for each task (`video_processor_service.py:115-116`), while global Re-ID gallery and ghost dictionaries are not reset. Concurrent tasks can corrupt each other and identities can survive into later tasks.
- No database, queue, cancellation API, durable task record, retry/recovery policy, ownership check, retention policy, or cleanup scheduler exists.
- `BlurConfigPanel.tsx` and `RedactionTargetPicker.tsx` exist but are not rendered by `UploadPage.tsx`; upload hard-codes Gaussian/50 (`UploadPage.tsx:79-84`).

There is no circular import found in the inspected runtime graph, but dependency direction is poor: an AI package initializer has heavyweight startup side effects, and endpoint/service imports cause model initialization before health routing can be exercised.

## 3. Frontend Audit

### Verified and implemented

The five pages exist: Landing, Dashboard, Upload, Processing, Results. The production TypeScript build passed. Upload calls `/videos/upload`; its response is mapped from snake_case (`src/services/api.ts:36-64`). Redaction-start request mapping is consistent with the backend (`api.ts:67-76`). WebSocket cleanup is present in `ProcessingPage.tsx:49-51`.

### Defects and limitations

- **Demo routes and state:** `/processing` and `/results` redirect to `demo-task` (`routes/index.tsx:22-35`); `RedactService` creates a demo task on demand (`redact_service.py:53-65`). The dashboard returns fixed activity and remote stock thumbnails (`dashboard_service.py:8-47`), not stored task history.
- **False processing UI:** `ProcessingPage.tsx:54-56` invents 5% progress when merely connected, uses a synthetic task filename (`:67`), and has no polling fallback, task-status request, error UI, or terminal failed state. The Cancel button only closes the socket/navigates (`:75-78`); it does not cancel backend work.
- **False success UI:** `ResultsPage.tsx:58-67` always labels the result “Verification Success” and “Processing Complete,” even while `activeSummary` is absent or its fetch failed (`:18-24`). It enables protected download without checking task status or output integrity.
- **WebSocket contract drift:** `FrameTelemetryPacket.status` is required (`types/index.ts:153-167`), but normal pipeline packets omit `status` (`video_processor_service.py:195-209`), so the client manufactures `PROCESSING` (`websocket.ts:67`). Completion is never delivered via this packet. `TaskManager.simulate_task_progress` has a different mock payload but is no longer scheduled.
- **Leak/reconnect risk:** `WebSocketManager` has a singleton, creates reconnect timers without retaining/clearing them, and `disconnect()` does not null callbacks/task ID (`websocket.ts:5-120`). A delayed reconnect can occur after a component has unmounted/reconnected to a different task.
- **Validation/accessibility:** `FileDropzone.tsx:24-35` accepts the first dropped file without enforcing type, extension, size, or video decodability; its clickable `<div>` is not keyboard-operable. The live camera component is explicitly a placeholder and ignores its callback (`LiveCameraCard.tsx:3-50`). Fixed `left-[240px]` results footer (`ResultsPage.tsx:103`) is likely to overflow/misplace on narrow screens. Multiple source strings show mojibake (`Â©`, `â€¢`).
- **Telemetry/marketing claims not backed by runtime:** TensorRT/CUDA claims appear in dashboard/top navigation although TensorRT is neither declared nor verified.

## 4. Backend and API Audit

### Route inventory

| Route | Actual behavior |
|---|---|
| `GET /api/v1/health` | Static healthy/version JSON; does not prove model, storage, FFmpeg, or task health (`health.py:5-11`). |
| `POST /api/v1/videos/upload` | Saves entire body synchronously, then reads limited OpenCV metadata. |
| `GET /api/v1/videos/download/{task_id}` | Returns redacted file if present; otherwise returns a 200 JSON placeholder. `raw` is ignored; `audit` returns a fixed CSV. |
| `POST /api/v1/redact/start` | Creates memory-only task and schedules process-local `asyncio.create_task`. Does not validate video existence. |
| `GET /api/v1/redact/status/{task_id}` | Reads memory task or manufactures `demo-task`. |
| `GET /api/v1/redact/results/{task_id}` | Returns fixed counts/evaluation, including for unknown tasks. |
| `GET /api/v1/dashboard/overview` | Fixed demo dashboard. |
| `WS /ws/redact/{task_id}` | Unauthenticated accepted connection, waits for inbound text, broadcasts only if a process emits data. |

### Complete request trace

`UploadPage.handleFileSelect` -> `api.uploadVideo` -> `POST /videos/upload` -> `VideoService.process_upload` -> `StorageManager.save_raw_video`. `handleStartProcess` -> `POST /redact/start` -> `TaskManager.create_task` -> `asyncio.create_task(VideoProcessorService.process_video_pipeline)`. The processor reads frames, calls the service chain, writes an output and broadcasts. Processing page only consumes WebSocket traffic; Results page calls `/redact/results` and builds download URLs.

Discrepancies: backend task filename is hard-coded to `surveillance_city_square.mp4` (`redact_service.py:31`); initial task metrics are fixed at 30 FPS/3,600 frames/120 seconds (`task_manager.py:22-31`); pipeline does not update `throughput_frames`; `created_at` is emitted by no status schema and client substitutes the current browser time (`api.ts:101`). `TaskSummaryResponse` omits `evaluation_report` (`schemas/redact.py:75-86`), so FastAPI response-model filtering removes the report that `redact_service.py:108` returns; the frontend's evaluation JSON download is therefore unverified/unavailable through this declared contract.

### Input/resource handling

There is no upload MIME, signature/container, file-size, duration, dimensions, codec, empty-file, or extension validation. `MAX_UPLOAD_SIZE_MB` is declared but unused (`settings.py:22`). `shutil.copyfileobj` is synchronous inside an async request (`storage.py:19-27`) and offers no streaming limit. The client-provided filename is incorporated directly in the path; separators such as `a\\..\\..\\..\\x` can escape intended storage after path normalization (`storage.py:20-21`). Uploads/raw outputs are retained indefinitely, logged with filesystem paths, and have no access control.

Errors are inconsistent: domain errors use `{success,message,status_code}` but FastAPI validation errors retain default shape; unknown downloads return 200 placeholder; results for unknown task return fabricated results. CORS permits listed localhost origins with credentials; there is no authentication/authorization or deployment-specific origin configuration evidence.

## 5. AI Pipeline Audit

### Model manager and execution

`ModelManager` detects CUDA and sets `cuda:0` if available (`model_manager.py:46-72`); the environment reported `torch 2.11.0+cu128` and `torch.cuda.is_available() == True`. This verifies only runtime availability, not that the model loaded or inferred. Model loading labels failed loads as `FALLBACK_READY` rather than failed (`model_manager.py:82-163`). YOLO weights `yolo11m.pt` are present, but YuNet ONNX weight is absent from file inventory and no OSNet weights/setup are documented. App import/model initialization did not finish in 60 seconds, so actual model status and warmup are **NOT VERIFIED**.

### Detection, tracking, Re-ID, ghosting

- The COCO mapping treats a **person as a face**, vehicles as **license plates**, and a book as an ID card (`detection_service.py:15-25`). Standard COCO YOLO is not a face/plate/document detector; these labels cannot substantiate sensitive-object detection.
- If inference returns zero (including legitimate no-sensitive-content frames or model error), it fabricates face and plate detections at fixed normalized boxes (`detection_service.py:143-164`), hiding model failure and producing false telemetry/redaction.
- YuNet also fabricates a face if no secondary inference output exists (`:209-218`).
- Tracking filters to 0.50 confidence, performs greedy IoU, reports zero ID switches, and does not use the constructed BoT-SORT binding (`tracking_service.py:85-106,119-188`). `boxmot` is not installed and is not in requirements.
- OSNet fallback uses a deterministic random vector derived only from track ID (`reid_service.py:74-77`). Even a successful similarity does not merge/reassign tracks.
- Ghost predictors are initialized only at loss with zero velocity and receive no measurement update from tracked boxes (`ghost_box_service.py:15-79,132-154`). Ghost state is never reset per job.

## 6. Redaction Quality Investigation (No Changes Made)

### Confirmed causes of insufficient redaction

1. **First-frame exposure:** `BlurService.render_blur` plans only `is_confirmed` tracks (`blur_service.py:72-79`). New tracks require two hits (`tracking_service.py:23-28,36-38`). A newly detected sensitive object is not blurred on its first frame. This is a confirmed privacy failure, independent of model quality.
2. **Missing/incorrect detector semantics:** the COCO class substitutions and unconditional fixed fallbacks above can miss actual faces/plates/documents while masking failure. Since blur only consumes tracks/ghosts, missed or incorrectly located detections remain unredacted.
3. **No output fail-safe:** frame/encoder errors do not block completion. A broken FFmpeg pipe is only logged and processing continues (`video_processor_service.py:171-178`); FFmpeg exit code/output decodability is never checked (`:215-254`).
4. **Per-task cross-contamination:** shared tracker/ghost/Re-ID state and tracker resets make locations/identities unreliable under concurrent jobs.

### Additional supported hypotheses to test next

- Normalized box conversion itself is top-left/width/height-consistent between detection and blur (`detection_service.py:120-135`, `blur_service.py:128-153`), so no direct percent-coordinate mismatch was found. However, semantic source boxes are wrong for the target entities.
- The feathered mode blends original pixels back at the mask edge/corners (`blur_service.py:193-207`); that is unsuitable as a fail-safe mask. It is not accepted by the current backend request enum even though frontend types allow it.
- Frame timestamps/VFR are discarded: OpenCV sequential decode plus FFmpeg `-r` emits constant-rate output (`video_processor_service.py:69-104`). Fast motion/VFR synchronization remains unverified.

**Recommended investigation order:** (1) establish fail-closed task/output criteria, (2) remove validation-masking fallbacks and obtain annotated redaction tests, (3) validate detector labels/coordinates against ground truth, (4) test first appearance, occlusion and lost-track sequences, (5) independently inspect encoded frames and audio/timing. This is the distinct next engineering phase; no blur code was changed.

## 7. Video Integrity and Privacy Audit

The processor tries to preserve audio by mapping the original audio stream (`video_processor_service.py:98-100`), but this is not verified. It forces a constant FPS and does not retain timestamps, has no corrupted/empty/large-video policy, uses no per-frame error isolation, silences FFmpeg stderr, and never checks encoder return code, produced-file existence, dimensions, duration, or decodability before setting `COMPLETED`.

`GET /videos/download` returns no raw original even with `raw=true`; it either returns the protected output or a successful-looking placeholder (`videos.py:16-46`). This is both an API correctness and privacy/auditability defect. The fixed CSV audit log is unrelated to frames/objects. No authentication means any party able to address a guessed task ID can request output/audit endpoints.

**Fail-safe verdict:** failing. A missing video ID drives synthetic frames (`video_processor_service.py:74-80,283-288`) and can receive completed status. An invalid/failed encoder can receive completed status. Results then state verification success and expose a download action.

## 8. Testing Report

| Command | Status | Actual result / what it proves | Does not prove |
|---|---|---|---|
| `npm run build` | PASS | `tsc -b && vite build`; 150 modules transformed; production bundle emitted. | Browser behavior, backend integration, model inference, privacy. |
| `npm run lint` | PASS | `oxlint` exited 0. | Type-aware rules, runtime behavior, tests. |
| `python -m compileall -q backend` | PASS | Exit 0. | Imports, FastAPI startup, runtime pipeline. |
| `python -c "import fastapi,pydantic,cv2,torch; ..."` | PASS | Python 3.11.0; FastAPI 0.140.7; Pydantic 2.13.4; OpenCV 5.0.0; Torch 2.11.0+cu128; CUDA True. | Model weights loaded, GPU inference succeeds. |
| dependency probe | PASS/PARTIAL | `ultralytics`, `torchreid`, `filterpy`, `httpx` found; `boxmot` absent. | Correct model/version/weights. |
| metadata-only OpenCV inspection of existing `test_input.mp4` and pre-existing redacted output | PASS/PARTIAL | Both opened; inspected output reported 150 frames, 30 FPS, 640x480. No pixels/content were exposed in this report. | Output came from that input; it is sufficiently blurred; audio/duration integrity. |
| `python -c "from fastapi.testclient import TestClient; from backend.main import app; ..."` | BLOCKED | Did not reach health response within 60 seconds; model initialization began during import. Audit process was terminated. | Health, routes, startup, model load. |
| existing `test_script.py` end-to-end script | NOT RUN | It writes new uploads/downloads into repository storage and downloads a result; audit avoided changing retained media. | End-to-end video processing. |
| frontend/backend unit/integration/WebSocket suites | NOT RUN | No test framework/config or discoverable suite exists; only ad-hoc `test_script.py` was found. | All listed behaviors. |
| FFmpeg binary probe | BLOCKED | `Get-Command ffmpeg` produced no path. Pipeline encoder behavior could not be independently exercised. | FFmpeg availability in the eventual service environment. |

The initial combined ultralytics import and isolated blur test were also terminated after 30 seconds because importing `backend.services.ai` triggered full model initialization. This is reported as an architecture/testability issue, not an inference result.

## 9. Findings Register

| ID | Severity | Category | Location | Observed behavior/evidence | Impact | Recommended remediation & verification |
|---|---|---|---|---|---|---|
| SF-001 | CRITICAL | Privacy correctness | `blur_service.py:72-79`; `tracking_service.py:23-28` | Only confirmed tracks are blurred; confirmation needs two hits. | First appearance is exposed. | Define immediate conservative mask policy; annotated first-frame test. |
| SF-002 | CRITICAL | AI integrity | `detection_service.py:15-25,143-164,209-218` | COCO proxy labels and fixed fallback detections. | Actual sensitive entities can be missed; false success/masks. | Use validated target models and fail-visible inference errors; ground-truth recall test. |
| SF-003 | CRITICAL | Fail-safe/export | `video_processor_service.py:171-178,215-254`; `ResultsPage.tsx:58-90` | Pipe/encoder errors are ignored; complete/success asserted without output validation. | Unredacted/invalid output can be presented protected. | Require encoder exit/existence/decode/redaction gate before completion; forced encoder-failure test. |
| SF-004 | CRITICAL | Job integrity | `video_processor_service.py:74-80,283-288`; `redact_service.py:20-46` | Missing video IDs create synthetic jobs; no video ownership/existence check. | Fictional protected results. | Reject unknown IDs; API test expects 404/no output. |
| SF-005 | HIGH | Data integrity/concurrency | `tracking_service.py:78-86`; `reid_service.py:31-36`; `ghost_box_service.py:93-96`; processor `:115-116` | Global mutable state shared; only tracker reset. | Cross-job leakage/corrupted redaction. | Per-task instances/state and concurrency test. |
| SF-006 | HIGH | API integrity | `redact_service.py:91-108`; `schemas/redact.py:75-86` | Fixed summary/evaluation; declared response removes evaluation report. | UI statistics/audit claims untrustworthy. | Derive real aggregates and align contract; schema contract test. |
| SF-007 | HIGH | Security/storage | `storage.py:19-27`; `settings.py:22` | Unsanitized client filename; no size/format limit; synchronous unbounded copy. | Traversal, disk exhaustion, blocking DoS. | Canonical server filename + streaming quotas/validation; negative tests. |
| SF-008 | HIGH | Authorization/privacy | all routes; `videos.py:16-46`; `main.py:34-40` | No auth/task ownership; raw/audit/output accessible by ID. | Sensitive media disclosure. | AuthZ and opaque ownership-bound IDs; access-control tests. |
| SF-009 | HIGH | Media integrity | `video_processor_service.py:87-113,248-254` | FFmpeg stderr suppressed/no exit code; no verification; command unavailable in audit shell. | Silent malformed/truncated output. | Capture errors and verify output before terminal state. |
| SF-010 | HIGH | Tracking claim | `tracking_service.py:88-106,119-225` | `boxmot` absent; BoT-SORT never used. | Claimed tracking quality unsupported. | Integrate one tracker or revise claim; identity continuity benchmark. |
| SF-011 | HIGH | Re-ID claim | `reid_service.py:74-77,122-206`; processor `:154` | Deterministic random fallback and no track reassignment. | Re-ID not functional. | Remove fallback/implement association; occlusion test. |
| SF-012 | HIGH | Task reliability | `task_manager.py:10-58`; `redact_service.py:42-46` | Process-local tasks, fire-and-forget work, no cancel/retry/recovery. | Restart loses jobs; resource control absent. | Durable task system/terminal failure recovery; restart test. |
| SF-013 | MEDIUM | WebSocket contract | `video_processor_service.py:195-209`; `websocket.ts:54-75` | Real packet lacks `status`; client invents state. | Completion navigation unreliable. | Contract schema/test and polling terminal fallback. |
| SF-014 | MEDIUM | UI correctness | `ProcessingPage.tsx:54-78`; `ResultsPage.tsx:18-67` | Invented 5% progress; cancel is local; success unconditional. | Users misled about privacy outcome. | Render backend status/error/output validation only; E2E tests. |
| SF-015 | MEDIUM | Demo data | `dashboard_service.py:8-47`; `videos.py:21-45`; `evaluation_service.py:45-134` | Fixed dashboard, CSV, metrics, counts, performance. | Operational/audit telemetry is false. | Replace with task-derived data; fixture-marked tests. |
| SF-016 | MEDIUM | Model startup | `services/ai/__init__.py:1`; `model_manager.py:35-43` | AI imports initialize all models; health could not complete in 60 seconds. | Poor availability/testability; health not meaningful. | Lazy/startup-managed initialization and readiness endpoint; cold-start test. |
| SF-017 | MEDIUM | Video timing | `video_processor_service.py:69-104,275-288` | Constant-rate output; no timestamps/VFR handling. | A/V drift/frame timing risk. | FFprobe/OpenCV duration/frame timestamp test corpus. |
| SF-018 | MEDIUM | Documentation | `README.md` | Stock React template, no SafeFrame install/model/backend/test/GPU instructions. | Not reproducible. | Accurate runbook; clean-machine validation. |
| SF-019 | LOW | Config/interface | `types/index.ts:8`; `schemas/redact.py:12-15`; `UploadPage.tsx:79-84` | Frontend allows `feathered`; backend rejects it; UI hard-codes settings. | User choices unavailable/inconsistent. | Align declared options and test each mode. |
| SF-020 | LOW | Accessibility/responsive | `FileDropzone.tsx:41-87`; `ResultsPage.tsx:103` | Nonsemantic clickable drop zone; fixed desktop sidebar offset. | Keyboard/mobile usability defect. | Keyboard/mobile accessibility tests. |

## 10. Production Readiness Matrix

| Capability | Status | Basis |
|---|---|---|
| Frontend | PARTIALLY VERIFIED | Build/lint pass; key UI states and accessibility unverified/incorrect. |
| Backend APIs | PARTIALLY VERIFIED | Routes/schemas compile; live health/routes blocked; contracts contain fabricated paths. |
| Model loading | IMPLEMENTED BUT UNVERIFIED | Weights/dependencies partly present; startup not completed. |
| Detection | FAILING | Wrong COCO semantics and fabricated fallback detections. |
| Tracking | IMPLEMENTED BUT UNVERIFIED | Custom IoU implementation only; BoT-SORT inactive. |
| Re-identification | FAILING | Random fallback and no association integration. |
| Ghost Box prediction | IMPLEMENTED BUT UNVERIFIED | Code path exists; state/velocity lifecycle invalid for claimed function. |
| Redaction | FAILING | First-frame bypass and fail-open completion path. |
| Video encoding | PARTIALLY VERIFIED | Existing output decodes; actual current encoder/failure behavior unverified. |
| WebSocket telemetry | PARTIALLY VERIFIED | Endpoint/client exist; contract completion mismatch. |
| Error handling | FAILING | Synthetic/placeholder success and unhandled task recovery. |
| Security | FAILING | No auth/ownership, upload validation, quota, retention, or safe path handling. |
| Testing | FAILING | No automated suite; only build/lint/compile checks pass. |
| Deployment readiness | NOT IMPLEMENTED | No deployment/runbook/health-readiness/security posture. |

## 11. Prioritized Remediation Roadmap

### P0 — Correctness, privacy, data integrity

1. Establish a fail-closed redaction contract: no protected download/completed state unless input exists, every frame outcome is accounted for, encoder succeeds, and output independently validates.
2. Eliminate fabricated detector/Re-ID/telemetry/evaluation/download fallbacks from production behavior; report explicit failure/unverified states instead.
3. Address immediate masking for newly detected sensitive regions and validate it against annotated first-frame/occlusion sequences.
4. Bind tasks to input ownership, sanitize storage paths, enforce upload/video limits, protect raw/output/audit endpoints, and define deletion/retention.

### P1 — Core function and integration

1. Use real appropriate face/plate/document/screen models and evaluate detection boxes; do not infer those labels from generic COCO classes.
2. Integrate a single tested tracker, connect Re-ID decisions to association, and isolate tracker/Re-ID/ghost state per task.
3. Make task execution durable/cancellable with bounded concurrency and consistent terminal/error responses.
4. Align frontend/backend response and WebSocket schemas; derive actual task name, metrics, summaries and audit records.

### P2 — Reliability and performance

1. Move blocking CPU/GPU/video work off the ASGI event loop and measure throughput/memory under bounded concurrency.
2. Capture FFmpeg diagnostics; preserve/validate A/V timing, VFR policy, dimensions, codec, duration and decodability.
3. Add corrupted, empty, short, large, frame-error and restart recovery tests.

### P3 — Deployment and usability

1. Replace README with reproducible Windows CPU/GPU/model/FFmpeg/test instructions.
2. Add accessible drop-zone keyboard support, responsive results layout, complete loading/empty/error states, and remove unsupported live camera/TensorRT claims.
3. Add monitoring/redacted logging policies without logging sensitive paths or content.

**Next engineering phase:** a dedicated, instrumented redaction-quality investigation using approved non-sensitive annotated fixtures, beginning with detection/box fidelity and first-frame fail-safe masking. Do not optimize the blur filter until those prerequisites and export gates are in place.

## 12. Final Engineering Verdict

What genuinely works: the React project builds and lints; frontend API request mapping is largely shaped correctly; the backend source defines routes and a pipeline sequence; the environment has CUDA-capable PyTorch; and a pre-existing encoded output opens in OpenCV.

What remains uncertain: live FastAPI startup, all actual model loads/inference, real detection recall, tracking/Re-ID/Ghost quality, FFmpeg execution, audio handling, WebSocket delivery, and whether any existing output is sufficiently redacted.

SafeFrame AI is **not production-ready**. Its present code can falsely claim a protected/verified video under missing input, missing/failed model output, frame-level exposure, encoder failure, and fabricated task data. The exact next phase is the P0 privacy-correctness and redaction-quality engineering phase described above, followed by controlled end-to-end verification.
