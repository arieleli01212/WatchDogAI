# WatchDogAI — Development Document and Scientific Basis

| | |
|---|---|
| **Project** | WatchDogAI — Real-Time Violence Detection System |
| **Author(s)** | [Ariel Elishayev, Barak Panker, Ness Kotenco ]|
| **Advisor(s)** | [Shimon Turchak ] |
| **Institution / Track** | [HIT — B.Sc. Computer Science] |
| **Date** | 2026-06-20 |
| **Status** | Draft |
| **Repository** | https://github.com/arieleli01212/WatchDogAI |

> This document incorporates the main content of the project's [SRS](SRS.md) and [PRD](PRD.md) and adds the scientific/theoretical basis, architecture, implementation, and testing detail required for submission. Where a claim is drawn from a specific file, the file is named so it can be independently verified against the repository.

---

## 1. Introduction: Problem, Need, and Goals

### 1.1 The Problem and the Need Behind It

Continuous human monitoring of video surveillance does not scale. A single operator cannot attentively watch many camera feeds at once, and in practice, violent incidents are often discovered only afterward, while reviewing recorded footage — at which point the opportunity for timely intervention has already passed. There is a need for a system that watches a video feed automatically, flags violent behavior as it happens, and preserves short, reviewable evidence of each incident, so a human only has to review a small number of flagged clips instead of hours of raw footage.

### 1.2 Project Goals

1. Detect violent behavior in real time from a live camera or a recorded video file.
2. Avoid over-triggering on single noisy frames by requiring corroborating evidence over a short time window before treating an event as real.
3. Automatically capture a short video clip around each confirmed incident — including lead-up and aftermath — as reviewable evidence.
4. Provide a lightweight live-monitoring dashboard and a historical, filterable alert log.
5. Run on ordinary hardware (CPU-only), with optional GPU acceleration if present, without requiring specialized surveillance equipment.

### 1.3 Functional Requirements (summary)

The full, itemized functional requirements are specified in [SRS.md §3.1](SRS.md#31-functional-requirements) (FR-1 through FR-6). In summary, the system must: capture frames from a webcam or video file (FR-1); classify frames for violence using a pre-trained ViT model with temporal smoothing (FR-2); maintain a rolling pre/post-event buffer and record MP4 clips around confirmed incidents (FR-3); persist cooldown-gated alerts to SQLite with delete support (FR-4); serve a live dashboard, alert history, and REST API (FR-5); and load all runtime parameters from environment variables at startup (FR-6).

## 2. Non-Functional Requirements

The full non-functional requirements are specified in [SRS.md §3.2](SRS.md#32-non-functional-requirements) (NFR-1 through NFR-6). The key constraints are:

- **Performance** — detection throughput is bounded by per-frame ViT inference time (~200–500ms on CPU); the capture loop must never block on inference or disk I/O.
- **Security** — the current implementation has no authentication on the dashboard or API; this is an accepted gap for a single-operator/local-network deployment, flagged explicitly in §9 (Recommendations) rather than silently left undocumented.
- **Scalability** — the system is single-process and single-camera by design (`camera_id` defaults to `"cam0"` everywhere); multi-camera support would require running multiple capture/detection pipelines and is out of scope for this version.
- **Reliability** — the system must shut down gracefully on `SIGINT`/`SIGTERM`, joining worker threads and releasing the camera device.

## 3. Scientific Basis

This section covers the theoretical background needed to understand the core algorithmic choices in the system: the vision transformer classifier, softmax-based confidence scoring, temporal smoothing as a signal-processing technique, and the concurrency model used to keep detection responsive.

### 3.1 Vision Transformers for Image Classification

The detection model, `jaranohaal/vit-base-violence-detection`, is a **Vision Transformer (ViT)** — specifically a `vit_base_patch16_224` architecture. ViT applies the transformer architecture, originally developed for sequence modeling in NLP, to images:

1. An input image is divided into a grid of fixed-size, non-overlapping patches (16×16 pixels, hence `patch16`).
2. Each patch is flattened and linearly projected into an embedding vector, analogous to a word embedding in NLP.
3. A learned position embedding is added to each patch embedding, since the transformer's self-attention has no inherent notion of spatial order.
4. The resulting sequence of patch embeddings (plus a special classification token, `[CLS]`) is passed through a stack of transformer encoder blocks, each combining **multi-head self-attention** (letting every patch attend to every other patch, so the model can relate distant regions of the frame — e.g. two people on opposite sides of the frame) with a position-wise feed-forward network.
5. The final `[CLS]` token's representation is passed through a linear classification head producing 2 output logits (this checkpoint: `num_classes=2`).

This project reuses this pre-trained model as-is rather than training a classifier from scratch — an application of **transfer learning**, where a model trained on a large labeled dataset (violent/non-violent video frames) is reused directly for inference in a new deployment context.

### 3.2 From Logits to a Confidence Score

The two raw output logits are converted to a probability distribution over the two classes using the **softmax function**:

softmax(z)_i = e^(z_i) / Σⱼ e^(z_j)

This guarantees the two class probabilities are non-negative and sum to 1, so the model's output can be interpreted as a calibrated-looking confidence rather than an arbitrary score. `ViolenceDetector.predict()` (`src/detector/model.py`) applies `torch.softmax(logits, dim=1)` and reads index `VIOLENCE_CLASS_IDX = 1` as the violence probability. A label of `"violence"` is emitted when this probability is ≥ 0.5, otherwise `"normal"` (with confidence expressed as `1 - p_violence`).

### 3.3 Preprocessing Correctness

A classifier's accuracy depends not only on its weights but on using the *exact* preprocessing pipeline it was trained with. This project's checkpoint is stored in `timm` weight format, which made it tempting to also use `timm`'s own preprocessing utilities (`resolve_data_config` / `create_transform`). During development, this was tried and found to silently corrupt predictions: `timm`'s default config for this architecture implies a `crop_pct=0.9` — the image is resized to a slightly larger size and then center-cropped to 224×224 — but this specific checkpoint was trained with a **direct resize to 224×224, no crop**. The mismatch was subtle: the model still produced *plausible-looking* confidence values, just wrong ones (root-caused and documented as **ISSUE-001** in `docs/issues/ISSUES.md`). The fix was to keep loading weights via `timm` (since the checkpoint format requires it) but perform preprocessing via `transformers.ViTImageProcessor.from_pretrained()`, which reconstructs the exact preprocessing config the model was trained with. This is a concrete, verified illustration of a general principle from machine learning practice: **model weights and preprocessing pipeline are a matched pair and cannot be substituted independently.**

### 3.4 Temporal Smoothing as a Sliding-Window Decision Rule

A single-frame classifier is inherently noisy at any fixed confidence threshold: a benign frame can momentarily look violent (motion blur, occlusion, similar pose) and vice versa. Rather than alerting on any one frame above threshold, the detection loop (`main.py:detection_loop`) implements a simple **sliding counter / debouncing algorithm**, conceptually related to techniques used in digital signal processing and control systems to reject transient noise before acting on a signal:

- Maintain a counter `consecutive_violence`, incremented on each frame classified `"violence"` with confidence ≥ `CONFIDENCE_THRESHOLD`, and reset to 0 on any frame that fails that test.
- Only treat the event as confirmed once the counter reaches `CONSECUTIVE_HITS` (default 3).

This is a minimal, O(1)-per-frame, constant-memory algorithm (a single integer counter, no history buffer needed) — an efficient way to trade a small amount of detection latency (at most `CONSECUTIVE_HITS` frames, i.e. roughly `CONSECUTIVE_HITS × inference_time`) for a large reduction in single-frame false positives, without the cost of a larger windowed/majority-vote scheme that would require buffering multiple frames' results.

### 3.5 The Clip Recorder as a Finite State Machine

`ClipRecorder` (`src/alerts/clip_recorder.py`) is modeled explicitly as a **finite state machine** with three states — `IDLE → RECORDING → SAVING → IDLE` — driven purely by the boolean smoothed detection result, decoupled from the raw per-frame confidence. Modeling it this way (rather than as ad hoc flags) makes the valid transitions and their side effects explicit and easy to reason about and test:

- **IDLE**: incoming frames are pushed into a bounded ring buffer (`collections.deque(maxlen=...)`) sized `fps × PRE_EVENT_SECONDS` — a classic **sliding window data structure** that automatically evicts the oldest frame once full, in O(1) per push.
- **IDLE → RECORDING**: triggered by confirmed violence; the pre-event ring buffer's contents seed the new recording, so the saved clip includes lead-up footage that occurred *before* the transition was detected.
- **RECORDING**: continued confirmed violence resets a post-event deadline; once violence stops, a countdown of `POST_EVENT_SECONDS` starts before moving on.
- **RECORDING → SAVING**: frames are hazarded out to a background thread for disk I/O (`cv2.VideoWriter`), specifically so that writing to disk — which is comparatively slow and unpredictable in latency — never blocks the capture thread, which must keep reading frames at full source FPS.
- **SAVING → IDLE**: once the file is written, `AlertManager.on_clip_saved()` is invoked and the state resets.

### 3.6 Concurrency and Synchronization

The system runs three concurrent execution contexts (capture thread, detection thread, dashboard/main thread) that share in-process objects rather than communicating through queues or inter-process channels. This requires explicit **mutual exclusion**: every shared, mutable structure (`Camera`'s latest-frame reference and clip buffer; `ClipRecorder`'s state and buffers) is guarded by a `threading.Lock`, so that, e.g., a frame is never read by the detection loop mid-write by the capture loop. This is a direct, practical application of concurrent-programming concepts (mutual exclusion, critical sections, and the general risk of race conditions on shared mutable state) covered in standard algorithms/systems coursework, applied here to keep video ingestion, inference, and web serving decoupled in latency from one another while still sharing the same objects rather than paying the overhead and complexity of an inter-process message queue.

## 4. Architecture and Design

### 4.1 High-Level Data Flow

```
Camera Feed → [Capture Thread] → Frame Buffer + ClipRecorder pre-event buffer (full FPS)
                                          │
                                  [Detection Thread] → ViT Model (single-frame inference)
                                          │                         │
                                  temporal smoothing        app.state.detector_status
                                          │                         │
                                  [ClipRecorder] state machine      │
                                          │                         │
                                  [AlertManager] → SQLite + MP4      │
                                          │                         │
                                  [Dashboard / FastAPI] ← live feed, alerts, /api/status
```

The capture loop is the only writer of raw frames; it fans out each frame to two independent consumers (`camera.add_frame()`, a small ring buffer for "what to show/analyze right now", and `clip_recorder.add_frame()`, the pre-event ring buffer for clip writing) so that neither consumer can block the other or the capture loop itself.

### 4.2 Data Structures

| Structure | Location | Purpose |
|---|---|---|
| `collections.deque(maxlen=...)` | `Camera._buffer`, `ClipRecorder._pre_buffer` | Fixed-size ring buffers giving O(1) append with automatic eviction of the oldest frame — used for both the (currently unused-by-live-detection) clip window and the pre-event buffer. |
| `list[np.ndarray]` | `ClipRecorder._rec_frames` | Append-only frame list accumulated for the duration of an active recording, handed off wholesale to the writer thread at finalization. |
| `dataclass(frozen=True)` | `Settings` (`src/config.py`) | Immutable configuration snapshot resolved once at startup from environment variables / `.env`. |
| SQLite table `alerts` | `AlertStorage` | `id INTEGER PRIMARY KEY, timestamp TEXT, confidence REAL, clip_path TEXT, camera_id TEXT, status TEXT DEFAULT 'new'`. |
| `enum.Enum` | `ClipRecorder._State` | `IDLE / RECORDING / SAVING` — the finite-state-machine states discussed in §3.5. |

### 4.3 Technological Interfaces

| Concern | Technology | Notes |
|---|---|---|
| Video I/O | OpenCV (`cv2.VideoCapture`, `cv2.VideoWriter`) | Reads webcam/video-file frames; writes MP4 clips with the `mp4v` fourcc. |
| Model inference | PyTorch + `timm` (weights) + `transformers` (preprocessing) + `safetensors` (weight loading) + `huggingface-hub` (download) | See §3.3 for why two separate libraries are involved. |
| Web layer | FastAPI + `uvicorn` (ASGI server) + Jinja2 (server-rendered HTML templates) | Runs in the main thread; blocks until shutdown. |
| Storage | SQLite (`sqlite3`, `check_same_thread=False`) | Single-file, zero-setup persistence; adequate for single-instance, single-writer-thread usage. |
| Config | `python-dotenv` | Loads a `.env` file (if present) before reading environment variables. |

### 4.4 Algorithms

The two central algorithms are described in depth in the Scientific Basis: the **consecutive-hits temporal smoothing rule** (§3.4) that gates when a detection becomes an "event," and the **pre/post-event ring-buffer state machine** (§3.5) that decides which frames get written to a clip. Both are deliberately simple, constant-memory, single-pass algorithms chosen for predictable low latency over the more complex alternative of buffering and re-analyzing whole clips (the approach the project moved away from — see ISSUE-002 in `docs/issues/ISSUES.md`).

### 4.5 User Interface`

The dashboard has two pages, both server-rendered via Jinja2 templates (`src/dashboard/templates/`):

- **Live View (`/`, `live.html`)** — an `<img>` tag pointed at the `/video_feed` MJPEG stream, plus a status bar that polls `GET /api/status` every 500ms and re-colors green/red based on the current smoothed label, showing the live violence-probability percentage.
- **Alerts (`/alerts`, `alerts.html`)** — a paginated (20/page) table of alert rows, each showing timestamp, camera id, confidence, status badge, an inline `<video>` element pointed at the clip under `/clips/...` (click to play/pause), and a delete button that calls `DELETE /api/alerts/{id}` and removes the row from the DOM on success. A status filter (`new` / `reviewed` / `dismissed`) is present in the UI and is passed through to `AlertStorage.get_alerts(status=...)`, though nothing in the current codebase ever transitions an alert's status away from its default `"new"` (see §9).

## 5. Solution Overview

WatchDogAI is implemented in Python 3.10+, wiring together five subsystems (`src/capture`, `src/detector`, `src/alerts`, `src/dashboard`, `src/config.py`) from a single entry point, `main.py`, which starts the capture and detection threads as daemon threads and then blocks the main thread on the `uvicorn` server. Configuration is read once at startup (`get_settings()`) and passed by reference into each component's constructor; there is no dependency-injection framework — components are wired together directly by `main()`.

## 6. Implementation

### 6.1 Tech Stack

Python 3.10+ · PyTorch · `timm` · `transformers` · OpenCV · FastAPI · `uvicorn` · Jinja2 · SQLite.

### 6.2 Project Structure

```
WatchDogAI/
├── main.py                    # Entry point: thread orchestration, capture_loop, detection_loop
├── src/
│   ├── config.py              # Settings — frozen dataclass from environment variables
│   ├── capture/camera.py      # OpenCV capture, thread-safe latest-frame + clip buffer
│   ├── detector/model.py      # ViT-based classifier (weights: timm, preprocessing: transformers)
│   ├── alerts/
│   │   ├── clip_recorder.py   # IDLE/RECORDING/SAVING state machine, MP4 writing
│   │   ├── manager.py         # Cooldown enforcement, alert creation/deletion
│   │   └── storage.py         # SQLite backend
│   └── dashboard/
│       ├── app.py             # FastAPI application factory
│       ├── routes.py          # Pages, REST API, MJPEG stream
│       └── templates/         # live.html, alerts.html, base.html
├── data/                      # clips/ + watchdog.db (git-ignored, created at runtime)
├── models/                    # cached HuggingFace model (git-ignored)
└── tests/                     # pytest suite, mirrors src/ layout
```

### 6.3 Configuration

All tunables are environment-variable driven (full table in [SRS.md §4](SRS.md#4-appendix--configuration-reference)): camera source, confidence threshold, consecutive-hits count, cooldown period, pre/post-event durations, clip/database/log directories, and dashboard port.

## 7. Verification of Correctness (Development History)

Two significant defects were found, root-caused, and fixed during development, both documented in detail in `docs/issues/ISSUES.md`:

- **ISSUE-001 (preprocessing mismatch)** — wrong preprocessing caused near-random/inverted predictions (a blank frame reading ~94% violent; genuine violent footage reading as normal). Root cause and fix are described in §3.3 above. Verified post-fix against sample violent and non-violent clips, with confidences in the 87–95% range for correctly-labeled examples.
- **ISSUE-002 (latency and snapshot mismatch)** — a 16-frame sliding window plus an added 0.5s sleep plus taking the snapshot from the live frame instead of the analyzed frame combined to make the dashboard feel laggy and occasionally save the wrong frame as evidence. Fixed by switching to single-frame prediction, removing the sleep, and guaranteeing the frame passed to `predict_frame()` is the one used as the alert's basis. Verified with a timestamped trace showing near-instant transitions between `normal` and `violence` states as the underlying footage changed.

## 8. Testing and Verification

### 8.1 Automated Test Suite

The project has an automated `pytest` suite (`tests/`, mirroring `src/`'s package layout) covering:

| Module under test | What is verified |
|---|---|
| `tests/test_config.py` | Default values and environment-variable overrides for every `Settings` field. |
| `tests/capture/test_camera.py` | Frame buffering, sliding-window eviction, clip readiness gating, latest-frame access, camera lifecycle (open/release/context manager). |
| `tests/detector/test_model.py`, `test_preprocessing.py` | Model loads and runs in eval mode on CPU; predictions return a valid label and a confidence in range; preprocessing output shape/dtype/normalization. |
| `tests/alerts/test_manager.py` | Alerts are created above threshold and suppressed below it; cooldown suppresses duplicate alerts and expires correctly; last-alert-time tracking. |
| `tests/alerts/test_storage.py` | Alert CRUD against SQLite: save/get/count/status-filter/update-status/delete, including incrementing ids and pagination. |
| `tests/dashboard/test_routes.py` | Live/alerts pages return 200; `/api/status`, `/api/alerts` return well-formed JSON; `/video_feed` returns the correct MJPEG content type. |
| `tests/test_main.py` | Logging setup; `detection_loop` processes frames, updates `app.state`, and stops cleanly on the shutdown event. |

Run the full suite with `pytest`, or a single test with e.g. `pytest tests/detector/test_model.py::test_predict_frame_returns_label_and_confidence`.

### 8.2 Known Gaps in Test Coverage

In the interest of an accurate, non-inflated verification record:

- **`ClipRecorder` has no dedicated test file** (`tests/alerts/` currently only covers `manager.py` and `storage.py`). The IDLE/RECORDING/SAVING state machine (§3.5, §4.2) — including pre-event buffer seeding, post-event countdown reset, and background-thread clip writing — is exercised manually/end-to-end but not by an automated unit test.
- Some test and configuration names still reference the project's earlier **snapshot-based** design (e.g. `Settings` still has an unused `model_path`/`MODEL_PATH` field, see below) predating the switch to video-clip recording (see the repository's commit history, e.g. "replace snapshot alerts with video clip recording"). This is a documentation/naming residue rather than a functional defect, but is worth cleaning up.
- **`Settings.model_path` (`MODEL_PATH` env var) is dead configuration** — `ViolenceDetector` is always constructed with no arguments in `main.py`, so it never reads this setting; the model source is effectively hardcoded (local cache path, else the HuggingFace id). Anyone trying to point the app at a different model via `MODEL_PATH` today would silently have no effect.
- **Model accuracy has not been independently re-validated** against this project's own held-out dataset; the ~98.8% figure cited in the README originates from the upstream model card, not from project-run evaluation.

## 9. Recommendations for Future Development

1. **Close the dead-configuration gap** — either wire `Settings.model_path` into `ViolenceDetector`'s constructor, or remove the unused field, so configuration surface matches actual behavior.
2. **Add automated tests for `ClipRecorder`** — specifically the state transitions in §3.5 (idle buffering, recording start with pre-event seeding, post-event countdown reset on continued violence, finalize-on-empty no-op).
3. **Add authentication/access control** to the dashboard and API before any deployment beyond a trusted local network (NFR-3 in the SRS).
4. **Implement the alert status workflow** — add API route(s) to transition an alert from `new` to `reviewed`/`dismissed`, since the schema and the alerts-page filter already assume this exists.
5. **Harden clip-writing failure handling** — currently a failure inside `ClipRecorder._write_clip` (e.g., disk full, unsupported codec) is not caught and would leave the recorder stuck in the `SAVING` state; add error handling with a fallback back to `IDLE`.
6. **Independently validate detection accuracy** against a held-out sample of this project's own footage, rather than relying solely on the upstream model card's reported accuracy.
7. **Multi-camera support** — generalize the single hardcoded `"cam0"` capture/detection pair into N independent pipelines feeding one shared dashboard and alert store.
8. **Notification integrations** (email/SMS/webhook) on new confirmed alerts, as originally sketched in the early design document (`docs/plans/2026-02-06-violence-detection-design.md`) but never implemented.
9. **Runtime reconfiguration** — investigate whether at least the detection thresholds (`CONFIDENCE_THRESHOLD`, `CONSECUTIVE_HITS`) can be safely hot-reloaded without a full restart, to ease tuning.
