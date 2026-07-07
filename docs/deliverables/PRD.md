# WatchDogAI — Product Requirements Document (PRD)

| | |
|---|---|
| **Project** | WatchDogAI — Real-Time Violence Detection System |
| **Author(s)** | [Student name(s)] |
| **Advisor(s)** | [Advisor name(s)] |
| **Institution / Track** | [Institution — B.Sc. Computer Science] |
| **Date** | 2026-07-03 |
| **Status** | Draft |
| **Repository** | https://github.com/arieleli01212/WatchDogAI |

> This PRD describes the product rationale, users, and priorities behind WatchDogAI. See [SRS.md](SRS.md) for the detailed technical/functional requirements and [development-document.md](development-document.md) for the combined submission document.

---

## 1. Problem Statement

Manual video surveillance does not scale: a human operator cannot continuously watch many camera feeds, and violent incidents are frequently noticed only afterward, during a manual review of recorded footage — by which point intervention is too late. WatchDogAI addresses this by automatically flagging violent behavior in a live or recorded feed in real time and preserving short video evidence of each event, so a human reviewer is alerted promptly and only has to review flagged clips instead of full-length footage.

## 2. Goals and Objectives

- Detect violent behavior in real time from a live camera or a recorded video file.
- Minimize false positives through temporal smoothing rather than acting on any single noisy frame.
- Automatically capture short evidentiary clips (lead-up + event + aftermath) around each detected incident, so a reviewer sees context, not just a single frame.
- Present a simple live-monitoring dashboard and a searchable/paginated alert history to a human reviewer.
- Run on commodity hardware — CPU-only, with optional GPU acceleration — with no specialized surveillance hardware required.

## 3. Target Users and Use Cases

| User | Use case |
|---|---|
| Facility/security operator | Runs the system against a webcam covering a monitored space; periodically checks the alerts page instead of watching the live feed continuously. |
| Researcher / developer | Runs the system against a recorded video file (`CAMERA_SOURCE=path/to/video.mp4`) to observe detection behavior against known violent/non-violent segments, e.g. while tuning `CONFIDENCE_THRESHOLD` or `CONSECUTIVE_HITS`. |
| Downstream integrator | Polls the REST API (`/api/status`, `/api/alerts`) from another system rather than using the bundled dashboard. |

## 4. Key Features

### P0 — Implemented, core to the product

- Real-time single-frame violence classification using a pre-trained ViT model (`jaranohaal/vit-base-violence-detection`).
- Temporal smoothing (`CONSECUTIVE_HITS` consecutive high-confidence frames) before an event counts as confirmed, to suppress single-frame false positives.
- Automatic MP4 clip recording with a configurable pre-event and post-event buffer.
- Cooldown-gated alert persistence to SQLite, so a single sustained incident does not flood the alert log.
- Live MJPEG dashboard with a real-time detection status banner (label + confidence, 500ms polling).
- Alerts page: pagination, status filter, inline clip playback, delete.

### P1 — Implemented, supporting

- REST API for external integration: `GET /api/status`, `GET /api/alerts`, `DELETE /api/alerts/{id}`.
- Full configuration via environment variables / `.env` (thresholds, timings, paths, ports).
- GPU auto-detection with CPU fallback.

### P2 — Not implemented (future work)

- Multi-camera support — `camera_id` is currently hardcoded to `"cam0"` as the default across the codebase; there is no mechanism to run more than one capture/detection pipeline per process.
- Alert status workflow beyond storage — the database schema and the alerts-page filter both support `new` / `reviewed` / `dismissed`, but there is no API route to transition an alert's status; only creation and deletion exist today.
- Notification integrations (email / SMS / webhook) — described as a future direction in the original design doc, not implemented.
- Authentication / access control on the dashboard and API.
- Remote or cloud storage of clips (currently local filesystem only).

## 5. Success Metrics

- **Detection accuracy** — inherited from the base model's published accuracy (~98.8%, per the HuggingFace model card); this project has not independently re-validated accuracy against its own dataset, and that validation is called out as follow-up work (Development Document §9).
- **Alert latency** — time from the start of a violent event to a confirmed/persisted alert is roughly `CONSECUTIVE_HITS × per-frame inference time` (~200–500ms/frame on CPU, per `docs/issues/ISSUES.md`), i.e. under ~1.5s on CPU with default settings.
- **Alert suppression correctness** — a single sustained incident should not generate more than one alert row within a `COOLDOWN_SECONDS` window, verified by `tests/alerts/test_manager.py`.
- **Operational responsiveness** — dashboard status reflects the current detection state within one polling interval (500ms).

## 6. Out of Scope (current version)

- Multi-camera fan-out within a single process.
- Person/weapon identification, face recognition, or any identity-level analysis.
- Mobile application or push notifications.
- Cloud deployment, multi-tenant hosting, or remote clip storage.
- Runtime (hot) reconfiguration — all settings changes require an application restart.

## 7. Current Status

The system is implemented and working end-to-end: capture → detection (with temporal smoothing) → clip recording (pre/post-event buffering) → cooldown-gated alert persistence → dashboard (live feed, alert history, REST API). It has been exercised against both a live webcam and recorded video files, with two notable bugs found and fixed during development (documented in `docs/issues/ISSUES.md`):

1. A model-preprocessing mismatch that produced near-random/inverted predictions (fixed by preprocessing with `transformers.ViTImageProcessor` instead of `timm`'s default transform).
2. Multi-second detection latency and a mismatched alert snapshot, caused by an unnecessarily large sliding-window buffer, an added sleep in the detection loop, and reading the live frame instead of the analyzed frame (fixed by moving to single-frame prediction).

See the [Development Document](development-document.md) for the full technical write-up, scientific basis, architecture, and testing details.
