# WatchDogAI — AI-Powered Smart Security System

Real-time video analytics for the HIT smart campus: two (or more)
cameras stream to an AI processing server that detects violence, counts
and tracks people and vehicles, flags suspicious movement and abnormal
behavior, and sends automated alerts — with video clips — to the Holon
municipal control center. Camera telemetry and alert events are also
published to the LoRa smart-campus gateway over MQTT.

See [docs/architecture.md](docs/architecture.md) for the full design,
including why video travels over IP while LoRa carries telemetry, and
[docs/INSTALL.md](docs/INSTALL.md) for the complete installation and
prerequisites guide (MongoDB, MQTT, GPU, service setup, troubleshooting).

## Features

- **Multi-camera pipelines** — N cameras (webcam / video file / RTSP
  stream) configured via one `CAMERAS` JSON variable; per-camera
  capture + analysis threads, automatic stream reconnect with backoff,
  configurable resolution/FPS
- **Violence detection** — pre-trained ViT classifier
  ([jaranohaal/vit-base-violence-detection](https://huggingface.co/jaranohaal/vit-base-violence-detection))
  with temporal smoothing over distinct frames
- **People & vehicle counting and tracking** — YOLOv8 + ByteTrack:
  live visible counts, cumulative unique counts, persistent track IDs,
  box overlay on the live feeds
- **Suspicious movement / abnormal behavior** — rule-based loitering
  and running detection plus a scikit-learn IsolationForest that learns
  each camera's normal motion (pandas feature pipeline) and flags
  movement unlike anything seen before
- **Automated alerts with video clips** — MP4 clips (~3s before +
  during + ~2s after each event) pushed to the municipal control
  center over HTTPS (multipart JSON + clip, retry queue) and announced
  on the LoRa gateway via MQTT
- **MongoDB storage** (preferred) with automatic thread-safe SQLite
  fallback, so alerting never stops
- **Interactive web dashboard** — live camera grid with health badges,
  detection status, and counts; paginated alert table with clip
  playback, camera/status filters, and delete; JSON REST API
- **Camera health telemetry** — per-camera online/healthy state on the
  dashboard and published periodically over MQTT
- **Optional token auth** for the dashboard and API (`API_TOKEN`)
- GPU auto-detection with CPU fallback

## Quick Start

```bash
git clone https://github.com/arieleli01212/WatchDogAI.git
cd WatchDogAI

python -m venv venv
source venv/Scripts/activate   # Windows (Git Bash)
# source venv/bin/activate     # Linux/macOS
pip install -r requirements.txt

# Configure (defaults run a single webcam without MQTT/control center)
cp .env.example .env           # then edit .env

python main.py
# Dashboard: http://localhost:8000
```

Run the tests:

```bash
python -m pytest
```

## Configuration

Set via environment variables or a `.env` file
(see [.env.example](.env.example) for the complete annotated list):

| Variable | Default | Description |
|---|---|---|
| `CAMERAS` | — | JSON array of cameras: `[{"id", "name", "source", "width", "height", "fps"}, ...]`; source = webcam index, file path, or RTSP/HTTP URL |
| `CAMERA_SOURCE` | `0` | Single-camera fallback when `CAMERAS` is unset |
| `CONFIDENCE_THRESHOLD` | `0.85` | Violence confidence threshold |
| `CONSECUTIVE_HITS` | `3` | Distinct frames required to confirm violence |
| `COOLDOWN_SECONDS` | `5` | Per-camera minimum seconds between alerts |
| `OBJECT_DETECTION` / `YOLO_MODEL` / `YOLO_CONFIDENCE` | `true` / `yolov8n.pt` / `0.4` | People & vehicle tracking |
| `BEHAVIOR_DETECTION` | `true` | Loitering/running/anomaly analytics |
| `LOITER_SECONDS` / `RUN_SPEED_THRESHOLD` | `60` / `0.35` | Rule thresholds (speed in frame diagonals per second) |
| `PRE_EVENT_SECONDS` / `POST_EVENT_SECONDS` | `3` / `2` | Clip padding around events |
| `DB_BACKEND` | `auto` | `mongodb`, `sqlite`, or `auto` (MongoDB when reachable) |
| `MONGODB_URI` / `MONGODB_DB` | `mongodb://localhost:27017` / `watchdog` | MongoDB connection |
| `CONTROL_CENTER_URL` / `CONTROL_CENTER_API_KEY` | — | HTTPS alert push (disabled when unset) |
| `MQTT_HOST` / `MQTT_PORT` / `MQTT_BASE_TOPIC` | — / `1883` / `watchdog` | LoRa gateway broker (disabled when unset) |
| `TELEMETRY_INTERVAL` | `30` | Seconds between camera-health MQTT messages |
| `API_TOKEN` | — | Require a token for all dashboard/API access |
| `DASHBOARD_PORT` | `8000` | Web dashboard port |

## Dashboard & API

- **Live View** (`/`) — camera grid: MJPEG feed with tracked-object
  overlay, health badge, violence status bar, people/vehicle counts
- **Alerts** (`/alerts`) — paginated table with timestamps, camera,
  alert type, confidence, clip playback, filters, delete
- **API** (spec: [docs/api/openapi.json](docs/api/openapi.json),
  Postman: [docs/api/WatchDogAI.postman_collection.json](docs/api/WatchDogAI.postman_collection.json)):
  - `GET /api/status` — system + per-camera state and detection results
  - `GET /api/cameras` — configured cameras and health
  - `GET /api/counts` — live people/vehicle counts per camera
  - `GET /api/alerts` — alert list (`limit`, `offset`, `status`, `camera_id`)
  - `DELETE /api/alerts/{id}` — delete an alert and its clip
  - `GET /video_feed/{camera_id}` — MJPEG live stream

With `API_TOKEN` set, browsers authenticate once via
`http://host:8000/?token=...` (session cookie); API clients send the
`X-API-Token` header.

## Project Structure

```
WatchDogAI/
├── main.py                      # Entry point: builds pipelines, notifiers, dashboard
├── src/
│   ├── config.py                # Settings from environment variables
│   ├── pipeline.py              # Per-camera capture + analysis threads
│   ├── capture/camera.py        # Webcam/file/RTSP capture with reconnect
│   ├── detector/
│   │   ├── model.py             # ViT violence classifier (shared, thread-safe)
│   │   └── objects.py           # YOLO + ByteTrack people/vehicle tracking
│   ├── analytics/behavior.py    # Loitering/running rules + IsolationForest
│   ├── alerts/
│   │   ├── clip_recorder.py     # Pre/post-event MP4 recording state machine
│   │   ├── manager.py           # Cooldowns, persistence, notifier fan-out
│   │   ├── storage.py           # SQLite backend + backend factory
│   │   ├── mongo_storage.py     # MongoDB backend (preferred)
│   │   └── notifier.py          # Control-center HTTPS push (retry queue)
│   ├── mqtt/client.py           # LoRa gateway MQTT client + telemetry loop
│   └── dashboard/               # FastAPI app, routes, Jinja2 templates
├── tests/                       # 200+ pytest tests (all offline, models mocked)
├── docs/
│   ├── architecture.md          # System design, LoRa split, deployment
│   └── api/                     # OpenAPI spec + Postman collection
└── requirements.txt
```

## Tech Stack

Python 3.10+ · PyTorch · timm/transformers (ViT) · Ultralytics YOLOv8 +
ByteTrack · OpenCV · pandas · scikit-learn · MongoDB (pymongo) ·
paho-mqtt · requests · FastAPI · Jinja2
