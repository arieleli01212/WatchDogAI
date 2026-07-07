# WatchDogAI — Software Requirements Specification (SRS)

| | |
|---|---|
| **Project** | WatchDogAI — Real-Time Violence Detection System |
| **Author(s)** | [Student name(s)] |
| **Advisor(s)** | [Advisor name(s)] |
| **Institution / Track** | [Institution — B.Sc. Computer Science] |
| **Date** | 2026-07-03 |
| **Status** | Draft |
| **Repository** | https://github.com/arieleli01212/WatchDogAI |

> This SRS describes the system as currently implemented in the repository (branch `master`). It is written to IEEE 830-style conventions, adapted for a single-developer academic project. See [PRD.md](PRD.md) for the product-level rationale and [development-document.md](development-document.md) for the combined submission document.

---

## 1. Introduction

### 1.1 Purpose

This document specifies the functional and non-functional requirements of WatchDogAI, a system that detects violent behavior in real time from a video source and records evidentiary clips of detected events. It is intended for whoever implements, tests, or extends the system, and to serve as the technical requirements baseline referenced by the project's development document.

### 1.2 Scope

WatchDogAI is a single-process, single-camera application that:

- Reads frames from a webcam or a video file.
- Classifies frames as `violence` / `normal` using a pre-trained Vision Transformer (ViT) model.
- Applies temporal smoothing to suppress false positives before treating an event as confirmed.
- Records an MP4 clip spanning the lead-up to, duration of, and aftermath of each confirmed event.
- Persists an alert record (timestamp, confidence, clip path, camera id, status) to a local SQLite database, subject to a cooldown period.
- Serves a browser-based dashboard (live feed + alert history) and a small REST API.

Out of scope: multi-camera fan-out, person/weapon identification, authentication, remote/cloud storage, and notification integrations (email/SMS/webhook) — none of these exist in the current implementation (see [PRD.md §7](PRD.md#7-out-of-scope-current-version)).

### 1.3 Definitions, Acronyms, Abbreviations

| Term | Meaning |
|---|---|
| ViT | Vision Transformer — the image classification architecture underlying the detection model |
| FR / NFR | Functional Requirement / Non-Functional Requirement |
| MJPEG | Motion JPEG — a video stream format that is a sequence of independently encoded JPEG frames, used for the live feed |
| Confidence threshold | Minimum per-frame violence probability (`CONFIDENCE_THRESHOLD`, default 0.85) required for a frame to count as a "hit" |
| Consecutive hits | Number of back-to-back qualifying frames (`CONSECUTIVE_HITS`, default 3) required before an event is "confirmed violence" |
| Cooldown | Minimum time (`COOLDOWN_SECONDS`, default 5s) between two persisted alerts, regardless of how long the underlying event lasts |
| Pre/post-event buffer | Frames retained before/after a confirmed event so the saved clip includes lead-up and aftermath footage |

### 1.4 References

- `README.md` — user-facing overview, configuration table, quick start
- `docs/issues/ISSUES.md` — root-cause/fix log for two significant bugs (preprocessing correctness, detection latency) discovered during development
- `docs/plans/2026-02-06-violence-detection-design.md` — original design proposal (partially superseded; retained for history)
- HuggingFace model card: `jaranohaal/vit-base-violence-detection`

### 1.5 Document Overview

Section 2 gives an overall description of the product and its constraints. Section 3 lists functional requirements grouped by subsystem, non-functional requirements, and external interfaces.

---

## 2. Overall Description

### 2.1 Product Perspective

WatchDogAI is a standalone application, not a component of a larger system. It runs as a single OS process with three concurrent execution contexts sharing in-process objects (no queues/IPC — thread-safety is via locks inside each component):

1. **Capture thread** — reads frames from the video source at full source FPS.
2. **Detection thread** — runs ViT inference on the latest frame and drives the clip-recording state machine.
3. **Dashboard (main thread)** — a blocking `uvicorn` server exposing the live feed, alert history, and REST API.

The only external dependency at runtime is the HuggingFace model download on first run (or a locally cached copy under `models/`).

### 2.2 Product Functions (summary)

- Real-time single-frame violence classification with temporal smoothing.
- Automatic pre/post-event MP4 clip recording.
- Cooldown-gated alert persistence to SQLite.
- Live MJPEG dashboard with detection status banner.
- Paginated, filterable alert history with clip playback and delete.
- REST API (`/api/status`, `/api/alerts`, `DELETE /api/alerts/{id}`).
- Full configuration via environment variables / `.env`.

### 2.3 User Classes and Characteristics

| User class | Description |
|---|---|
| Operator | Views the live dashboard and alert history in a browser; no configuration access required. |
| Administrator/Developer | Edits environment variables or `.env` and restarts the process to change detection/recording behavior; runs the app from a terminal. |

No authentication distinguishes these classes today — anyone who can reach the dashboard port has both views (see NFR-3).

### 2.4 Constraints

- Python 3.10+, single process, single machine.
- `Settings` (`src/config.py`) is a frozen dataclass resolved once at startup — **no runtime reconfiguration**; every settings change requires a restart.
- Camera identity is effectively fixed: `camera_id` defaults to `"cam0"` throughout (`ClipRecorder`, `main.py`) with no mechanism to run multiple camera pipelines in one process.
- SQLite (`AlertStorage`) is opened with `check_same_thread=False` to allow cross-thread access, but is not designed for concurrent multi-process writers.
- Detection is single-frame (`predict_frame`), not clip-based, by deliberate design decision (ISSUE-002) — the multi-frame `Camera.get_clip()` buffer exists but is not on the live detection path.

### 2.5 Assumptions and Dependencies

- The configured `CAMERA_SOURCE` is readable by OpenCV (a working webcam index or a decodable video file path).
- Either network access to HuggingFace is available on first run, or the model is pre-populated under `models/vit-violence-detection/`.
- The `timm`-format checkpoint remains compatible with `vit_base_patch16_224` (2-class head) and with `transformers.ViTImageProcessor` preprocessing (see FR-2.3 and Development Document §4).

---

## 3. Specific Requirements

### 3.1 Functional Requirements

#### FR-1 — Video Capture (`src/capture/camera.py`, `main.py:capture_loop`)

| ID | Requirement |
|---|---|
| FR-1.1 | The system shall capture frames from a webcam (integer index) or a video file (path), selected by the `CAMERA_SOURCE` setting. |
| FR-1.2 | For a video-file source, the system shall loop playback from the beginning when the end of the file is reached. |
| FR-1.3 | For a video-file source, the system shall pace frame reads to the source's native FPS. |
| FR-1.4 | The system shall maintain a thread-safe "latest frame" reference, concurrently readable by the detection loop and the dashboard's MJPEG stream while the capture thread writes to it. |
| FR-1.5 | The system shall maintain a sliding-window clip buffer (`Camera.get_clip()`, size `CLIP_LENGTH`) in addition to the latest-frame reference. This buffer is retained for a clip-based detection mode but is not consumed by the current live detection path (see FR-2.1). |

#### FR-2 — Violence Detection (`src/detector/model.py`, `main.py:detection_loop`)

| ID | Requirement |
|---|---|
| FR-2.1 | The system shall classify the current latest frame using a pre-trained ViT model (`jaranohaal/vit-base-violence-detection`), returning a label (`"violence"` / `"normal"`) and a confidence in `[0, 1]`. |
| FR-2.2 | The system shall load model weights from a local cached directory (`models/vit-violence-detection/`) if present, otherwise download them from HuggingFace on first run. |
| FR-2.3 | The system shall preprocess frames using `transformers.ViTImageProcessor` (direct resize to 224×224, no center-crop) rather than `timm`'s default transform pipeline, per the root cause documented in ISSUE-001. |
| FR-2.4 | The system shall apply temporal smoothing: an event is treated as "confirmed violence" only after `CONSECUTIVE_HITS` consecutive frames classified `"violence"` at or above `CONFIDENCE_THRESHOLD`. |
| FR-2.5 | The system shall reset the consecutive-hit counter to zero on any frame that is below threshold or classified `"normal"`. |
| FR-2.6 | The system shall auto-select a CUDA device if available, otherwise fall back to CPU. |

#### FR-3 — Clip Recording (`src/alerts/clip_recorder.py`)

| ID | Requirement |
|---|---|
| FR-3.1 | While idle, the system shall maintain a rolling pre-event buffer of `PRE_EVENT_SECONDS × fps` frames. |
| FR-3.2 | On confirmed violence, the system shall transition to recording, seeding the clip with the current pre-event buffer. |
| FR-3.3 | While recording, continued confirmed violence shall reset the post-event countdown. |
| FR-3.4 | Once confirmed violence stops, the system shall continue recording for `POST_EVENT_SECONDS` before finalizing the clip. |
| FR-3.5 | The system shall write the finalized clip as an MP4 file (`mp4v` fourcc) to `CLIP_DIR/<YYYY-MM-DD>/<HH-MM-SS>_<camera_id>.mp4` on a background thread, so the capture loop is never blocked by disk I/O. |
| FR-3.6 | If a recording finalizes with zero buffered frames, the system shall discard it without writing a file or creating an alert. |

#### FR-4 — Alert Management (`src/alerts/manager.py`, `src/alerts/storage.py`)

| ID | Requirement |
|---|---|
| FR-4.1 | On clip save, the system shall create a persisted alert row unless the elapsed time since the previous alert is less than `COOLDOWN_SECONDS`. |
| FR-4.2 | Each alert record shall include: `timestamp` (UTC, ISO-8601), `confidence`, `clip_path`, `camera_id`, `status` (default `"new"`). |
| FR-4.3 | The system shall support deleting an alert, which removes both the database row and the associated clip file from disk. |
| FR-4.4 | The system shall support retrieving alerts with pagination (`limit`/`offset`) and an optional `status` filter. |

#### FR-5 — Dashboard and API (`src/dashboard/app.py`, `src/dashboard/routes.py`)

| ID | Requirement |
|---|---|
| FR-5.1 | The system shall serve a live-view page (`/`) with an MJPEG video stream (`/video_feed`) and a detection status banner that polls `/api/status` every 500ms. |
| FR-5.2 | The system shall serve a paginated alerts page (`/alerts`, 20 per page) with an optional status filter, inline clip playback, and a delete action per row. |
| FR-5.3 | The system shall expose REST endpoints: `GET /api/status`, `GET /api/alerts` (with `limit`, `offset`, `status`), `DELETE /api/alerts/{id}`. |
| FR-5.4 | The system shall serve saved clip files as static content under `/clips`. |

#### FR-6 — Configuration (`src/config.py`)

| ID | Requirement |
|---|---|
| FR-6.1 | All runtime parameters shall be configurable via environment variables or a `.env` file, resolved once at process startup into an immutable settings object. |
| FR-6.2 | Changing configuration shall require an application restart; there is no hot-reload mechanism. |

> **Known inconsistency:** `Settings.model_path` (env var `MODEL_PATH`) is defined and defaulted but never read by `ViolenceDetector`, which always uses the hardcoded HuggingFace id / local-cache path (`src/detector/model.py`). This is a dead configuration field — see Development Document §9 (Recommendations).

### 3.2 Non-Functional Requirements

#### NFR-1 — Performance

- Detection throughput is bounded by per-frame inference time (~200–500ms on CPU, per `docs/issues/ISSUES.md`); a CUDA GPU reduces this when available (FR-2.6).
- The capture loop must not block on detection or on disk I/O; clip writing happens on a dedicated background thread (FR-3.5) specifically to keep frame capture responsive.
- The dashboard's MJPEG stream targets ~30 fps encode rate (33ms sleep between frames).

#### NFR-2 — Reliability

- The capture loop must tolerate transient read failures from a live webcam by retrying rather than crashing (`main.py:capture_loop`).
- The system must shut down gracefully on `SIGINT`/`SIGTERM`: signal both worker threads to stop, join them (5s timeout), and release the camera device.
- **Gap:** clip-writing failures (e.g., disk full, codec unavailable) are not currently caught or surfaced; a failure inside `ClipRecorder._write_clip` would leave the recorder stuck in `SAVING` state. Flagged in Development Document §9.

#### NFR-3 — Security

- **Gap:** no authentication or authorization exists on the dashboard or API — anyone with network access to the configured host/port can view the live feed, browse alert history, and delete alert records.
- All SQLite access uses parameterized queries (`src/alerts/storage.py`) — no SQL-injection surface was found.
- The system does not handle external secrets or credentials; the only external call is the (optional, first-run) HuggingFace model download.

#### NFR-4 — Scalability

- The design is single-process/single-camera; `camera_id` is hardcoded to `"cam0"` as the default throughout. Supporting multiple cameras would require re-architecting to N independent capture/detection loop pairs sharing one dashboard.
- SQLite is adequate at single-instance scale but is not intended for concurrent multi-process writers.

#### NFR-5 — Maintainability

- Settings are centralized in one frozen dataclass with explicit defaults, simplifying configuration auditing.
- Components communicate via direct method calls guarded by locks rather than a message queue, which keeps the codebase simple to read but ties all three execution contexts to a single process/machine (relates to NFR-4).

#### NFR-6 — Usability

- The dashboard requires no login and no client software beyond a browser.
- Status color coding (green/red) and 500ms polling give the operator near-real-time feedback without manual refresh.

### 3.3 External Interface Requirements

| Interface type | Description |
|---|---|
| Hardware | Any OpenCV-compatible webcam, or a video file already present on disk. No specialized hardware. |
| Software / library | PyTorch, `timm`, `transformers`, `safetensors`, `huggingface-hub` (model); OpenCV (video I/O); FastAPI, Jinja2, `uvicorn` (web); SQLite (storage). |
| User interface | Browser-based dashboard: live view (`/`) and alerts (`/alerts`). |
| API interface | JSON REST endpoints under `/api/*`, plus MJPEG at `/video_feed` and static file serving under `/clips`. |

---

## 4. Appendix — Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `CAMERA_SOURCE` | `0` | Webcam index (int) or video file path |
| `CONFIDENCE_THRESHOLD` | `0.85` | Violence confidence threshold for a per-frame "hit" |
| `CONSECUTIVE_HITS` | `3` | Consecutive hits required before an event is confirmed |
| `COOLDOWN_SECONDS` | `5` | Minimum seconds between two persisted alerts |
| `PRE_EVENT_SECONDS` | `3` | Seconds of video kept before a confirmed event |
| `POST_EVENT_SECONDS` | `2` | Seconds of video recorded after violence ends |
| `CLIP_DIR` | `data/clips` | Directory for saved MP4 clips |
| `DASHBOARD_PORT` | `8000` | Web dashboard port |
| `DB_PATH` | `data/watchdog.db` | SQLite database path |
| `LOG_DIR` | `logs` | Log file directory |
| `LOG_LEVEL` | `INFO` | Logging level |

Acceptance criteria for each functional requirement are covered by the automated test suite described in the Development Document, §8 (Testing and Verification).
