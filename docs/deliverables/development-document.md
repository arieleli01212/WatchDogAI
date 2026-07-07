# WatchDogAI — Development Document and Scientific Basis

| | |
|---|---|
| **Project** | WatchDogAI — AI-Powered Smart Security System |
| **Author(s)** | [Ariel Elishayev, Barak Panker, Ness Kotenco ]|
| **Advisor(s)** | [Shimon Turchak ] |
| **Institution / Track** | [HIT — B.Sc. Computer Science] |
| **Date** | 2026-07-07 |
| **Status** | Draft |
| **Repository** | https://github.com/arieleli01212/WatchDogAI |

> This document incorporates the main content of the project's [SRS](SRS.md) and [PRD](PRD.md) and adds the scientific/theoretical basis, architecture, implementation, and testing detail required for submission. Where a claim is drawn from a specific file, the file is named so it can be independently verified against the repository.

---

## 1. Introduction: Problem, Need, and Goals

### 1.1 The Problem and the Need Behind It

Continuous human monitoring of video surveillance does not scale. A single operator cannot attentively watch many camera feeds at once, and in practice, violent incidents are often discovered only afterward, while reviewing recorded footage — at which point the opportunity for timely intervention has already passed. There is a need for a system that watches multiple video feeds automatically, flags violent and suspicious behavior as it happens, and preserves short, reviewable evidence of each incident, so a human only has to review a small number of flagged clips instead of hours of raw footage — and for that evidence to reach a control center automatically rather than waiting for someone to check a local dashboard.

### 1.2 Project Goals

1. Detect violent behavior in real time from live cameras (webcams or network streams), recorded video files, or whole folders of archived footage.
2. Avoid over-triggering on single noisy frames by requiring corroborating evidence over a short time window before treating an event as real.
3. Automatically capture a short video clip around each confirmed incident — including lead-up and aftermath — as reviewable evidence, playable directly in a browser.
4. Track people and vehicles across frames, count them (current and cumulative unique), and detect suspicious movement patterns (loitering, running, statistically anomalous motion) beyond the frame-level violence classifier.
5. Run multiple cameras concurrently in one process, each with its own capture/analysis pipeline, feeding a shared dashboard and alert store.
6. Push alerts (with clips) outward — to a municipal control center over HTTP and to a low-bandwidth LoRa gateway over MQTT — so the system integrates into existing infrastructure instead of being a silo.
7. Provide a lightweight live-monitoring dashboard and a historical, filterable alert log, optionally protected by an API token.
8. Run on ordinary hardware (CPU-only), with optional GPU acceleration if present, without requiring specialized surveillance equipment.

### 1.3 Functional Requirements (summary)

The full, itemized functional requirements are specified in [SRS.md §3.1](SRS.md#31-functional-requirements). In summary, the system must: capture frames from one or more cameras — webcam, video file, network stream (RTSP/HTTP), or a folder of video files (FR-1); classify frames for violence using a pre-trained ViT model with temporal smoothing, and track/count people and vehicles with behavior analytics (FR-2); maintain a rolling pre/post-event buffer per camera and record browser-playable MP4 clips around confirmed incidents (FR-3); persist cooldown-gated alerts to MongoDB (with SQLite fallback) with delete support, and push them to configured external endpoints (FR-4); serve a live multi-camera dashboard, alert history, and REST API with optional token authentication (FR-5); and load all runtime parameters from environment variables at startup (FR-6).

## 2. Non-Functional Requirements

The full non-functional requirements are specified in [SRS.md §3.2](SRS.md#32-non-functional-requirements). The key constraints are:

- **Performance** — detection throughput is bounded by per-frame ViT inference time (~200–500ms on CPU) plus YOLO tracking when enabled; each camera's capture loop must never block on inference, disk I/O, or network delivery. Heavy models are shared across cameras; per-camera state is not.
- **Security** — the dashboard and API can be protected by a shared API token (`API_TOKEN`), enforced by middleware with constant-time comparison; outbound control-center pushes carry an `X-API-Key` header. There is no per-user authentication or role model — an accepted limitation for a single-operator deployment, flagged in §9.
- **Scalability** — the system is single-process but multi-camera: each configured camera runs its own capture + analysis thread pair against shared model/alert components. Scaling beyond one machine (many dozens of cameras) is out of scope for this version.
- **Resilience** — optional subsystems degrade gracefully rather than crashing the pipeline: an unreachable MongoDB falls back to SQLite, a failing object tracker or behavior analyzer disables itself for that camera, an unreachable MQTT broker or control center is retried in the background, and a database outage does not silence outbound alerting.
- **Reliability** — the system must shut down gracefully on `SIGINT`/`SIGTERM`, joining worker threads, releasing cameras, and flushing/closing the notifier and MQTT clients.

## 3. Scientific Basis

This section covers the theoretical background needed to understand the core algorithmic choices in the system: the vision transformer classifier, softmax-based confidence scoring, temporal smoothing as a signal-processing technique, single-stage object detection with multi-object tracking, unsupervised anomaly detection on motion features, and the concurrency model used to keep detection responsive.

### 3.1 Vision Transformers for Image Classification

The detection model, `jaranohaal/vit-base-violence-detection`, is a **Vision Transformer (ViT)** — specifically a `vit_base_patch16_224` architecture. ViT applies the transformer architecture, originally developed for sequence modeling in NLP, to images:

1. An input image is divided into a grid of fixed-size, non-overlapping patches (16×16 pixels, hence `patch16`).
2. Each patch is flattened and linearly projected into an embedding vector, analogous to a word embedding in NLP.
3. A learned position embedding is added to each patch embedding, since the transformer's self-attention has no inherent notion of spatial order.
4. The resulting sequence of patch embeddings (plus a special classification token, `[CLS]`) is passed through a stack of transformer encoder blocks, each combining **multi-head self-attention** (letting every patch attend to every other patch, so the model can relate distant regions of the frame — e.g. two people on opposite sides of the frame) with a position-wise feed-forward network.
5. The final `[CLS]` token's representation is passed through a linear classification head producing 2 output logits (this checkpoint: `num_classes=2`).

This project reuses this pre-trained model as-is rather than training a classifier from scratch — an application of **transfer learning**, where a model trained on a large labeled dataset (violent/non-violent video frames) is reused directly for inference in a new deployment context. A single detector instance is shared by all camera pipelines, since inference is stateless per frame.

### 3.2 From Logits to a Confidence Score

The two raw output logits are converted to a probability distribution over the two classes using the **softmax function**:

softmax(z)_i = e^(z_i) / Σⱼ e^(z_j)

This guarantees the two class probabilities are non-negative and sum to 1, so the model's output can be interpreted as a calibrated-looking confidence rather than an arbitrary score. `ViolenceDetector.predict()` (`src/detector/model.py`) applies `torch.softmax(logits, dim=1)` and reads index `VIOLENCE_CLASS_IDX = 1` as the violence probability. A label of `"violence"` is emitted when this probability is ≥ 0.5, otherwise `"normal"` (with confidence expressed as `1 - p_violence`).

### 3.3 Preprocessing Correctness

A classifier's accuracy depends not only on its weights but on using the *exact* preprocessing pipeline it was trained with. This project's checkpoint is stored in `timm` weight format, which made it tempting to also use `timm`'s own preprocessing utilities (`resolve_data_config` / `create_transform`). During development, this was tried and found to silently corrupt predictions: `timm`'s default config for this architecture implies a `crop_pct=0.9` — the image is resized to a slightly larger size and then center-cropped to 224×224 — but this specific checkpoint was trained with a **direct resize to 224×224, no crop**. The mismatch was subtle: the model still produced *plausible-looking* confidence values, just wrong ones (root-caused and documented as **ISSUE-001** in `docs/issues/ISSUES.md`). The fix was to keep loading weights via `timm` (since the checkpoint format requires it) but perform preprocessing via `transformers.ViTImageProcessor.from_pretrained()`, which reconstructs the exact preprocessing config the model was trained with. This is a concrete, verified illustration of a general principle from machine learning practice: **model weights and preprocessing pipeline are a matched pair and cannot be substituted independently.**

### 3.4 Temporal Smoothing as a Sliding-Window Decision Rule

A single-frame classifier is inherently noisy at any fixed confidence threshold: a benign frame can momentarily look violent (motion blur, occlusion, similar pose) and vice versa. Rather than alerting on any one frame above threshold, each camera's analysis loop (`CameraPipeline._analysis_loop` in `src/pipeline.py`) implements a simple **sliding counter / debouncing algorithm**, conceptually related to techniques used in digital signal processing and control systems to reject transient noise before acting on a signal:

- Maintain a counter `consecutive_violence`, incremented on each frame classified `"violence"` with confidence ≥ `CONFIDENCE_THRESHOLD`, and reset to 0 on any frame that fails that test.
- Only treat the event as confirmed once the counter reaches `CONSECUTIVE_HITS` (default 3).
- Count only *distinct* frames: the camera exposes a monotonically increasing frame sequence number (`get_latest_frame_with_seq()`), and the analysis loop skips any frame it has already scored, so a fast analysis loop cannot double-count a slow camera's frame.

This is a minimal, O(1)-per-frame, constant-memory algorithm (a single integer counter, no history buffer needed) — an efficient way to trade a small amount of detection latency (at most `CONSECUTIVE_HITS` frames, i.e. roughly `CONSECUTIVE_HITS × inference_time`) for a large reduction in single-frame false positives, without the cost of a larger windowed/majority-vote scheme that would require buffering multiple frames' results.

### 3.5 The Clip Recorder as a Finite State Machine

`ClipRecorder` (`src/alerts/clip_recorder.py`) is modeled explicitly as a **finite state machine** with three states — `IDLE → RECORDING → SAVING → IDLE` — driven purely by the boolean smoothed detection result, decoupled from the raw per-frame confidence. Each camera pipeline owns its own recorder. Modeling it this way (rather than as ad hoc flags) makes the valid transitions and their side effects explicit and easy to reason about and test:

- **IDLE**: incoming frames are pushed into a bounded ring buffer (`collections.deque(maxlen=...)`) sized `fps × PRE_EVENT_SECONDS` — a classic **sliding window data structure** that automatically evicts the oldest frame once full, in O(1) per push.
- **IDLE → RECORDING**: triggered by a confirmed event; the pre-event ring buffer's contents seed the new recording, so the saved clip includes lead-up footage that occurred *before* the transition was detected. The recorder tags the recording with an `alert_type` (`violence`, or a behavior-event type such as `loitering`); if violence is confirmed during a behavior-triggered recording, `violence` outranks it.
- **RECORDING**: continued confirmed activity resets a post-event deadline; once it stops, a countdown of `POST_EVENT_SECONDS` starts before moving on. Recording length is capped at `MAX_CLIP_SECONDS` — a sustained event is *chunked* into multiple clips (with millisecond-suffixed filenames to keep them distinct) instead of growing without bound in memory.
- **RECORDING → SAVING**: frames are handed off to a background thread for disk I/O, specifically so that writing to disk — comparatively slow and unpredictable in latency — never blocks the capture thread. Encoding failures are caught: a clip that cannot be written is logged and skipped, and the recorder always returns to `IDLE` (no stuck states). A `tick()` method lets the analysis loop finalize an overdue recording even when the camera stops delivering frames (e.g. the camera died right after the incident), so captured footage is never stranded in `RECORDING`.
- **SAVING**: the writer first produces an MP4 via `cv2.VideoWriter` (`mp4v` fourcc, MPEG-4 Part 2 — which browsers cannot play), then **re-encodes it to H.264** with `ffmpeg` (`libx264`, `yuv420p`, `+faststart`) so the dashboard's inline `<video>` previews work; if `ffmpeg` is unavailable or fails, the original `mp4v` file is kept as a fallback. During `SAVING`, the pre-event buffer keeps filling and a newly confirmed event starts a fresh recording immediately — back-to-back events are not lost.
- **SAVING → IDLE**: once the file is written, `AlertManager.on_clip_saved()` is invoked and the state resets (unless a new event has already re-entered `RECORDING`).

By default only confirmed **violence** triggers a saved clip; behavior events (§3.7) are surfaced live in the status feed but only produce clips/alerts when `RECORD_BEHAVIOR_CLIPS` is enabled — a deliberate signal-to-noise decision so the alert log stays reviewable.

### 3.6 Single-Stage Object Detection and Multi-Object Tracking

Beyond the frame-level violence classifier, each camera can run **people and vehicle analytics** (`ObjectTracker` in `src/detector/objects.py`), built from two well-studied components:

- **YOLO (You Only Look Once)** — a *single-stage* object detector (`yolov8n` by default) that predicts bounding boxes and class probabilities for a whole frame in one forward pass, rather than the two-stage propose-then-classify approach (e.g. Faster R-CNN). Single-stage detection trades some accuracy for the real-time throughput this system needs. Detections are filtered to the COCO classes of interest: `person`, and vehicle classes (`car`, `truck`, `bus`, `motorcycle`, `bicycle`).
- **ByteTrack multi-object tracking** — per-frame detections alone cannot say whether the person in frame *t* is the same person as in frame *t+1*. ByteTrack associates detections across frames (using motion prediction and IoU matching, notably keeping *low-confidence* detections in the association step, which is what makes it robust to partial occlusion) and assigns each object a **persistent track ID**. Track IDs power two features: **unique-visitor counting** (the set of distinct IDs seen, alongside currently-visible counts) and the trajectory input for behavior analytics.

Tracker state is inherently per-stream, so each camera pipeline owns its own `ObjectTracker` instance, while detection weights are shared.

### 3.7 Behavior Analytics: Rules plus Unsupervised Anomaly Detection

`BehaviorAnalyzer` (`src/analytics/behavior.py`) consumes tracked trajectories and emits *behavior events* through two complementary detectors — a deliberate hybrid of explainable rules and learned statistics:

**Motion features.** For each track, a short sliding window (~3s) of positions is reduced to five features: mean speed, max speed, **tortuosity** (path length ÷ net displacement — ~1 for straight walking, high for erratic motion), dwell time, and lifetime radius from the track's average position. All coordinates are normalized by the **frame diagonal**, so the same thresholds work across cameras with different resolutions.

**Rule-based detectors** (always active, explainable):
- *Loitering* — a person present longer than `LOITER_SECONDS` while staying within a small radius of their average position.
- *Running* — a person whose smoothed speed exceeds `RUN_SPEED_THRESHOLD` (in frame diagonals per second).

**Statistical anomaly detection** — an unsupervised **IsolationForest** (scikit-learn) is periodically fitted on the rolling history of motion features observed *by this camera*, separately for people and vehicles. Isolation forests detect anomalies by how *easily a point is isolated* by random axis-aligned splits: outliers sit in sparse regions and need few splits, so they get short average path lengths across the ensemble of random trees. This requires no labeled "abnormal" examples — the camera learns what normal motion looks like at its own location. Two refinements were needed in practice:
- A **range guard**: an isolation forest cannot split on features whose training values are nearly constant, so it under-scores out-of-range outliers when the camera has only seen uniform motion. Learned per-feature bounds (quantiles widened by IQR) flag feature vectors far outside everything previously observed, even if the forest scores them as inliers.
- **Per-track event cooldowns**, so one loitering person doesn't generate an event every frame.

### 3.8 Concurrency and Synchronization

The system runs *2N + 1* long-lived execution contexts for N cameras — one capture thread and one analysis thread per camera (`CameraPipeline`, `src/pipeline.py`), plus the dashboard/main thread — alongside short-lived clip-writer threads and optional background workers (control-center delivery, MQTT network loop, telemetry). They share in-process objects rather than communicating through inter-process channels; where cross-thread hand-off is needed, two standard patterns are used:

- **Mutual exclusion** — every shared, mutable structure (`Camera`'s latest-frame reference and buffer; `ClipRecorder`'s state and buffers) is guarded by a `threading.Lock`, so a frame is never read by the analysis loop mid-write by the capture loop. This is a direct application of the concurrent-programming concepts (critical sections, race conditions on shared mutable state) covered in standard systems coursework.
- **Producer/consumer queues for slow I/O** — outbound alert delivery (`ControlCenterNotifier`, §4.6) runs on a background worker fed by a **bounded queue**: the analysis pipeline enqueues and never blocks, the worker drains with retries, and when the queue is full the oldest work is *dropped with a logged error* rather than back-pressuring detection — an explicit choice that real-time detection outranks guaranteed delivery (alerts remain in local storage regardless).

Heavy, stateless resources (the ViT detector) are shared across all pipelines; stateful ones (camera buffers, clip recorders, trackers, behavior analyzers, smoothing counters) are per-pipeline, which is what makes the multi-camera design safe without global coordination.

## 4. Architecture and Design

### 4.1 High-Level Data Flow

```
                        ┌── per camera (× N) ──────────────────────────────┐
Camera / file / stream →│ [Capture Thread] → frame buffer + pre-event buffer│
                        │        │                                          │
                        │ [Analysis Thread] → ViT violence classifier       │
                        │        │            YOLO + ByteTrack tracker      │
                        │        │            BehaviorAnalyzer              │
                        │  temporal smoothing → status registry ────────────┼──→ [Dashboard / FastAPI]
                        │        │                                          │      live grid, /api/*,
                        │ [ClipRecorder FSM] → writer thread → H.264 MP4    │      per-camera MJPEG
                        └────────┼──────────────────────────────────────────┘
                                 │
                          [AlertManager] → MongoDB (or SQLite fallback)
                                 │
                        ┌────────┴────────────┐
                 [ControlCenterNotifier]  [MQTT gateway client]
                  HTTP push (JSON+clip)    alerts + telemetry (LoRa)
```

Within each pipeline, the capture loop is the only writer of raw frames; it fans out each frame to two independent consumers (`camera.add_frame()`, a small ring buffer for "what to show/analyze right now", and `clip_recorder.add_frame()`, the pre-event ring buffer for clip writing) so that neither consumer can block the other or the capture loop itself. All pipelines publish per-camera status into a shared registry (`app.state.camera_status`) read by the dashboard, and feed one shared `AlertManager`.

### 4.2 Camera Sources and Multi-Camera Configuration

Cameras are declared via the `CAMERAS` environment variable — a JSON array of `{id, name, source, width, height, fps}` entries (`src/config.py`); a single-camera `CAMERA_SOURCE` fallback (default: webcam 0) keeps the simple case simple. A `source` is classified (`classify_source`, `src/capture/camera.py`) as a **webcam index**, a **network stream URL** (RTSP/HTTP), or a **file path**. A source that is a *directory* is expanded into one camera per video file inside it (ids `archive-0`, `archive-1`, …), each running as its own independent pipeline with its own dashboard card and alert attribution — this is how a folder of recorded footage is processed alongside live streams in the same deployment. Camera ids are validated (`[A-Za-z0-9_-]+`, uniqueness enforced) because they end up in filenames, URLs, and MQTT topics. Each camera also reports **health** (`is_healthy`): whether a frame has arrived within `CAMERA_HEALTH_MAX_AGE` seconds, surfaced on the dashboard and in telemetry.

### 4.3 Data Structures

| Structure | Location | Purpose |
|---|---|---|
| `collections.deque(maxlen=...)` | `Camera._buffer`, `ClipRecorder._pre_buffer`, `_TrackState.positions` | Fixed-size ring buffers giving O(1) append with automatic eviction — latest-frame window, pre-event clip buffer, and per-track position history. |
| `list[np.ndarray]` | `ClipRecorder._rec_frames` | Append-only frame list accumulated during an active recording (bounded by `MAX_CLIP_SECONDS`), handed off wholesale to the writer thread at finalization. |
| `queue.Queue(maxsize=...)` | `ControlCenterNotifier._queue` | Bounded producer/consumer queue decoupling alert delivery from the analysis pipelines (§3.8). |
| `dataclass(frozen=True)` | `Settings`, `CameraConfig` (`src/config.py`), `TrackedObject` (`src/detector/objects.py`) | Immutable configuration snapshots and per-frame tracked-object records. |
| `dict[str, dict]` status registry | `app.state.camera_status` | Per-camera live status (label, confidence, streak, counts, tracked objects, behavior events) written by analysis threads, read by dashboard routes. |
| `set[int]` | `ObjectTracker._unique_people/_unique_vehicles` | Cumulative unique track-ID sets powering unique-visitor counts. |
| MongoDB collection / SQLite table `alerts` | `MongoAlertStorage` / `AlertStorage` | `id, timestamp, confidence, clip_path, camera_id, alert_type, status` — identical shape on both backends (Mongo assigns integer ids from an atomic counter document so the REST API behaves identically; SQLite migrates older DBs by adding `alert_type`). |
| `enum.Enum` | `ClipRecorder._State`, `Camera.SourceType` | FSM states (§3.5) and source classification (webcam/file/stream). |
| `IsolationForest` + bounds dict | `BehaviorAnalyzer._models/_bounds` | Per-category anomaly model and learned per-feature range guard (§3.7). |

### 4.4 Technological Interfaces

| Concern | Technology | Notes |
|---|---|---|
| Video I/O | OpenCV (`cv2.VideoCapture`, `cv2.VideoWriter`) | Webcams, video files, and RTSP/HTTP streams; raw clip writing. |
| Clip encoding | `ffmpeg` (subprocess) | Re-encodes `mp4v` writer output to browser-playable H.264 (`libx264`, `+faststart`); optional — falls back to `mp4v` with a logged warning. |
| Violence model | PyTorch + `timm` (weights) + `transformers` (preprocessing) + `safetensors` + `huggingface-hub` | See §3.3 for why two libraries are involved. |
| Object tracking | `ultralytics` (YOLOv8 + ByteTrack) | Per-camera tracker instances; lazily imported. |
| Behavior analytics | `pandas` + `scikit-learn` (IsolationForest) | Feature pipeline and unsupervised anomaly model (§3.7). |
| Web layer | FastAPI + `uvicorn` + Jinja2 | Runs in the main thread; token-auth middleware when `API_TOKEN` is set. |
| Storage | MongoDB (`pymongo`) preferred, SQLite (`sqlite3`) fallback | `DB_BACKEND=auto` (default) pings Mongo at startup and falls back to SQLite if unreachable, so the system keeps alerting without a database server. |
| Control-center push | `requests` (HTTP multipart) | JSON alert + MP4 clip in one POST; `X-API-Key` header. |
| LoRa gateway | `paho-mqtt` | Compact JSON only (LoRa is kbit/s-class — video never travels this link): `watchdog/alerts/<camera_id>` per alert, `watchdog/telemetry/<camera_id>` periodic health/counts. |
| Config | `python-dotenv` | Loads `.env` (if present) before reading environment variables. |

### 4.5 Algorithms

The central algorithms are described in depth in the Scientific Basis: the **consecutive-hits temporal smoothing rule** (§3.4) that gates when a detection becomes an "event"; the **pre/post-event ring-buffer state machine** with length-capped chunking (§3.5) that decides which frames get written to a clip; **YOLO + ByteTrack tracking** (§3.6) that turns per-frame detections into persistent trajectories; and the **hybrid rule/IsolationForest behavior analytics** (§3.7) over those trajectories. The real-time path (smoothing, ring buffers, feature extraction) is deliberately built from simple, constant-memory, single-pass steps chosen for predictable low latency — the project moved away from buffering and re-analyzing whole clips (see ISSUE-002 in `docs/issues/ISSUES.md`); the learned component (anomaly model refitting) runs only periodically and off the per-frame critical path.

### 4.6 Alert Flow and Outbound Delivery

`AlertManager` (`src/alerts/manager.py`) coordinates what happens after a clip is saved:

1. **Per-camera cooldown** — a `COOLDOWN_SECONDS` window is enforced *per camera* (one camera's incident doesn't suppress another's); a clip suppressed by cooldown is deleted so no orphaned files accumulate.
2. **Persistence** — the alert (timestamp, confidence, clip path, camera id, alert type) is saved via the storage backend selected at startup (§4.4). A database outage is logged but does **not** silence outbound alerting — the control center is still notified, just without a stored id.
3. **Notification fan-out** — every registered notifier receives the alert dict + clip path; a failing notifier is logged and skipped, never crashing alerting. Two notifiers exist:
   - `ControlCenterNotifier` (`src/alerts/notifier.py`) — POSTs multipart/form-data (JSON `alert` field + MP4 `clip` file) to `CONTROL_CENTER_URL`, on a background worker with a bounded queue and **exponential-backoff retries**; non-retryable 4xx responses are abandoned immediately (except 408/429), and shutdown interrupts in-flight backoff waits, reporting undelivered counts.
   - `MqttGatewayClient` (`src/mqtt/client.py`) — publishes a compact alert payload (id, timestamp, confidence, camera, type, clip *path* — the clip itself stays on the server for retrieval over IP) with QoS 1; `connect_async` + a paho network thread keeps reconnecting so a gateway outage never blocks the pipeline. A companion `TelemetryLoop` thread publishes per-camera health/counts every `TELEMETRY_INTERVAL` seconds.
4. **Deletion safety** — deleting an alert removes its clip, but only after resolving the path and verifying it lies inside the configured clip directory, so a tampered database row can never unlink arbitrary files.

### 4.7 User Interface

The dashboard is a desktop-oriented, calm "command-console"-styled UI, server-rendered via Jinja2 templates (`src/dashboard/templates/`):

- **Live View (`/`, `live.html`)** — a grid of per-camera cards, each with its own MJPEG stream (`/video_feed/{camera_id}`) overlaid with tracked-object bounding boxes and track IDs (drawn server-side per frame, JPEG-encoded off the event loop so one viewer can't stall others), plus per-camera health, smoothed violence status, streak progress, and live people/vehicle counts polled from `GET /api/status`. Overall system status is reported as `active` / `degraded` (some cameras unhealthy) / `inactive`.
- **Alerts (`/alerts`, `alerts.html`)** — a paginated (20/page) table of alert rows showing timestamp, camera id, alert type, confidence, status badge, an inline `<video>` element playing the H.264 clip under `/clips/...`, and a delete button calling `DELETE /api/alerts/{id}`. Filters for status (`new` / `reviewed` / `dismissed`) and camera id are passed through to storage; note that nothing in the current codebase transitions an alert's status away from `"new"` (see §9).
- **REST API** — `GET /api/status`, `GET /api/cameras`, `GET /api/counts`, `GET /api/alerts` (paginated, filterable), `DELETE /api/alerts/{id}`. When `API_TOKEN` is set, middleware (`src/dashboard/app.py`) requires it on every request — as an `X-API-Token` header (API clients), a one-time `token` query parameter (browser login; answered with a redirect that strips the token from the URL so it doesn't linger in history), or the `httponly` session cookie set after that login (which is what lets MJPEG `<img>` tags and `fetch()` calls authenticate). Comparison is constant-time (`hmac.compare_digest`).

## 5. Solution Overview

WatchDogAI is implemented in Python 3.10+, wiring together seven subsystems (`src/capture`, `src/detector`, `src/analytics`, `src/alerts`, `src/mqtt`, `src/dashboard`, `src/config.py`) from a single entry point, `main.py`. At startup it builds the shared components (one `ViolenceDetector`, one `AlertManager` with the selected storage backend, optional control-center and MQTT notifiers), then constructs one `CameraPipeline` per configured camera — each owning its camera, clip recorder, tracker, and behavior analyzer — starts every pipeline's capture and analysis threads as daemons, and finally blocks the main thread on the `uvicorn` server. Configuration is read once at startup (`get_settings()`) and passed by reference into each component's constructor; there is no dependency-injection framework — components are wired together directly by `main()`.

## 6. Implementation

### 6.1 Tech Stack

Python 3.10+ · PyTorch · `timm` · `transformers` · `ultralytics` (YOLOv8 + ByteTrack) · scikit-learn · pandas · OpenCV · `ffmpeg` · FastAPI · `uvicorn` · Jinja2 · MongoDB (`pymongo`) / SQLite · `paho-mqtt` · `requests`.

### 6.2 Project Structure

```
WatchDogAI/
├── main.py                    # Entry point: builds shared components, one pipeline per camera
├── src/
│   ├── config.py              # Settings + CameraConfig — frozen dataclasses from env vars
│   ├── pipeline.py            # CameraPipeline: per-camera capture + analysis threads
│   ├── capture/camera.py      # OpenCV capture (webcam/file/stream), health, frame sequencing
│   ├── detector/
│   │   ├── model.py           # ViT violence classifier (weights: timm, preprocessing: transformers)
│   │   └── objects.py         # YOLO + ByteTrack people/vehicle tracking and counting
│   ├── analytics/behavior.py  # Loitering/running rules + IsolationForest anomaly detection
│   ├── alerts/
│   │   ├── clip_recorder.py   # IDLE/RECORDING/SAVING FSM, chunking, H.264 re-encode
│   │   ├── manager.py         # Per-camera cooldown, persistence, notifier fan-out, deletion
│   │   ├── storage.py         # SQLite backend + create_alert_storage() backend selection
│   │   ├── mongo_storage.py   # MongoDB backend (interface-compatible with SQLite)
│   │   └── notifier.py        # Control-center HTTP push (queue + backoff retries)
│   ├── mqtt/client.py         # LoRa gateway MQTT client + telemetry loop
│   └── dashboard/
│       ├── app.py             # FastAPI application factory + token-auth middleware
│       ├── routes.py          # Pages, REST API, per-camera MJPEG streams with overlays
│       └── templates/         # live.html, alerts.html, base.html
├── data/                      # clips/ + watchdog.db (git-ignored, created at runtime)
├── models/                    # cached HuggingFace model (git-ignored)
└── tests/                     # pytest suite, mirrors src/ layout
```

### 6.3 Configuration

All tunables are environment-variable driven (full table in [SRS.md §4](SRS.md#4-appendix--configuration-reference)): the camera list (`CAMERAS` JSON, or `CAMERA_SOURCE`), detection thresholds (`CONFIDENCE_THRESHOLD`, `CONSECUTIVE_HITS`), cooldown and clip timing (`COOLDOWN_SECONDS`, `PRE/POST_EVENT_SECONDS`, `MAX_CLIP_SECONDS`), storage backend selection (`DB_BACKEND`, `MONGODB_URI/DB`, `DB_PATH`), object detection and behavior analytics (`OBJECT_DETECTION`, `YOLO_MODEL`, `YOLO_CONFIDENCE`, `BEHAVIOR_DETECTION`, `LOITER_SECONDS`, `RUN_SPEED_THRESHOLD`, `ANOMALY_MIN_SAMPLES`, `BEHAVIOR_EVENT_COOLDOWN`, `RECORD_BEHAVIOR_CLIPS`), outbound integrations (`CONTROL_CENTER_URL/API_KEY`, `MQTT_HOST/PORT/USERNAME/PASSWORD/BASE_TOPIC`, `TELEMETRY_INTERVAL`), dashboard (`DASHBOARD_PORT`, `API_TOKEN`), and paths/logging (`CLIP_DIR`, `LOG_DIR`, `LOG_LEVEL`, `CAMERA_HEALTH_MAX_AGE`).

## 7. Verification of Correctness (Development History)

Several significant defects were found, root-caused, and fixed during development; the first two are documented in detail in `docs/issues/ISSUES.md`:

- **ISSUE-001 (preprocessing mismatch)** — wrong preprocessing caused near-random/inverted predictions (a blank frame reading ~94% violent; genuine violent footage reading as normal). Root cause and fix are described in §3.3 above. Verified post-fix against sample violent and non-violent clips, with confidences in the 87–95% range for correctly-labeled examples.
- **ISSUE-002 (latency and snapshot mismatch)** — a 16-frame sliding window plus an added 0.5s sleep plus taking the snapshot from the live frame instead of the analyzed frame combined to make the dashboard feel laggy and occasionally save the wrong frame as evidence. Fixed by switching to single-frame prediction, removing the sleep, and guaranteeing the frame passed to `predict_frame()` is the one used as the alert's basis. Verified with a timestamped trace showing near-instant transitions between `normal` and `violence` states as the underlying footage changed.
- **Browser-unplayable clips** — clips written with OpenCV's `mp4v` fourcc are MPEG-4 Part 2, which browsers cannot decode, so the dashboard's inline `<video>` previews rendered black despite the files being valid. Fixed by re-encoding every saved clip to H.264 via `ffmpeg` (§3.5), with a logged fallback to the raw `mp4v` file when `ffmpeg` is unavailable.
- **Adversarial-review hardening** — a dedicated review pass over the multi-camera/analytics work surfaced and fixed a batch of defects (see commit "Fix defects confirmed by adversarial review"), including: unbounded pagination parameters overflowing SQLite's 64-bit `OFFSET` binding into HTTP 500s (now capped), path traversal in clip URLs/deletion (clip paths are now resolved and confined to the clip directory), camera ids reaching filenames unsanitized, orphaned clip files left behind by cooldown-suppressed alerts, and the isolation forest under-scoring out-of-range outliers on low-variance history (the range guard in §3.7).

## 8. Testing and Verification

### 8.1 Automated Test Suite

The project has an automated `pytest` suite (`tests/`, mirroring `src/`'s package layout) covering:

| Module under test | What is verified |
|---|---|
| `tests/test_config.py` | Default values and environment-variable overrides for `Settings`; `CAMERAS` JSON parsing, id validation, and folder-of-videos expansion. |
| `tests/capture/test_camera.py` | Frame buffering and eviction, latest-frame + sequence-number access, source classification (webcam/file/stream), health reporting, camera lifecycle. |
| `tests/detector/test_model.py` | Model loads and runs in eval mode on CPU; predictions return a valid label and in-range confidence; preprocessing shape/dtype/normalization. |
| `tests/detector/test_objects.py` | YOLO/ByteTrack wrapper: category filtering, tracked-object construction, visible and cumulative unique counts. |
| `tests/analytics/test_behavior.py` | Feature extraction, loitering/running rules, anomaly model fitting, range guard, event cooldowns, stale-track pruning. |
| `tests/test_pipeline.py` | Per-camera pipeline: smoothing over distinct frames, status publishing, violence vs. behavior clip-recording policy, tracker/behavior failure isolation. |
| `tests/alerts/test_clip_recorder.py` | The IDLE/RECORDING/SAVING state machine: pre-event seeding, post-event countdown reset, max-length chunking, `tick()` finalization, encode-failure recovery. |
| `tests/alerts/test_manager.py` | Per-camera cooldown enforcement, persistence-failure tolerance, notifier fan-out and failure isolation, clip-deletion path confinement. |
| `tests/alerts/test_storage.py` / `test_mongo_storage.py` | Alert CRUD, filters, pagination, and status updates on both backends; `alert_type` schema migration; backend selection/fallback. |
| `tests/alerts/test_notifier.py` | Queueing, multipart delivery, retry/backoff behavior, non-retryable 4xx handling, interruptible shutdown. |
| `tests/mqtt/test_client.py` | Compact alert payload shape, topic construction, telemetry loop publishing. |
| `tests/dashboard/test_routes.py` | Pages return 200; JSON API shapes; per-camera MJPEG content type; pagination caps; token-auth middleware behavior. |
| `tests/test_main.py` | Logging setup and entry-point wiring. |

Run the full suite with `pytest`, or a single test file with e.g. `pytest tests/detector/test_model.py`.

### 8.2 Known Gaps in Test Coverage

In the interest of an accurate, non-inflated verification record:

- **MongoDB tests run against a stub/mocked client** rather than a live server in CI; the SQLite fallback path is the one continuously exercised end-to-end.
- **YOLO/behavior tests use synthetic detections**, not real footage — tracking quality (ID switches under occlusion, crowded scenes) has only been assessed manually.
- **The H.264 re-encode path depends on a system `ffmpeg`** and is exercised only where one is installed; the fallback branch is what runs otherwise.
- **Model accuracy has not been independently re-validated** against this project's own held-out dataset; the ~98.8% figure cited in the README originates from the upstream model card, not from project-run evaluation. Likewise the behavior-rule thresholds (loiter radius, run speed) were tuned by observation, not by systematic evaluation.

## 9. Recommendations for Future Development

1. **Implement the alert status workflow** — `update_status` exists on both storage backends and the alerts-page filter already assumes `reviewed`/`dismissed` states, but no API route or UI action transitions an alert away from `new`.
2. **Independently validate detection accuracy** against a held-out sample of this project's own footage — both the violence classifier (rather than relying on the upstream model card) and the behavior rules/anomaly detector (false-positive rate per camera-hour).
3. **Per-user authentication and roles** — the shared `API_TOKEN` protects the dashboard, but there is no notion of individual operators, audit trail, or permission levels; needed before multi-operator deployment.
4. **Horizontal scaling** — the multi-camera design is single-process; scaling to many dozens of cameras means splitting capture/analysis workers across processes or machines, which would replace shared in-process state with a message bus and shared storage.
5. **Runtime reconfiguration** — investigate whether at least the detection thresholds (`CONFIDENCE_THRESHOLD`, `CONSECUTIVE_HITS`) and per-camera enable/disable can be safely hot-reloaded without a full restart, to ease tuning; adding/removing cameras at runtime would follow.
6. **Fuse the violence classifier with tracking** — the ViT scores whole frames while the tracker knows where people are; cropping to person regions (or weighting by them) could raise precision and attribute violence alerts to specific tracks.
7. **Delivery durability** — the control-center notifier drops the newest alert when its in-memory queue overflows and loses queued alerts on shutdown (they remain in local storage); a persistent outbox with resume-on-start would make outbound delivery at-least-once.
8. **GPU batching across cameras** — with many cameras sharing one detector, batching concurrent frames into a single forward pass would raise total throughput on GPU hardware.
