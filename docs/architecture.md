# WatchDogAI — System Architecture

AI-powered smart security system for the HIT smart campus, reporting to
the Holon municipal control center.

## Overview

```
                       IP network (high bandwidth)
  ┌────────────┐  RTSP   ┌──────────────────────────────────────────┐
  │ Camera 1   ├────────►│            AI processing server           │
  │ (North)    │         │                                          │
  └────────────┘         │  per camera (CameraPipeline):            │
  ┌────────────┐  RTSP   │   capture thread ──► frame + seq         │
  │ Camera 2   ├────────►│   analysis thread:                       │
  │ (South)    │         │     • ViT violence classifier (shared)   │
  └────────────┘         │     • YOLO + ByteTrack tracker           │
                         │     • behavior analytics (rules + ML)    │
                         │   ClipRecorder ──► MP4 clips             │
                         │                                          │
                         │  shared: AlertManager ──► MongoDB        │
                         │          FastAPI dashboard (MJPEG, API)  │
                         └───────┬──────────────────┬───────────────┘
                                 │ HTTPS             │ MQTT (LoRa gateway)
                                 ▼                   ▼
                     ┌───────────────────┐   ┌──────────────────────┐
                     │ Municipal control │   │ LoRa smart-campus    │
                     │ center (alert +   │   │ gateway (telemetry,  │
                     │ MP4 clip upload)  │   │ compact alert events)│
                     └───────────────────┘   └──────────────────────┘
```

## Why video does NOT go over LoRa

LoRa/LoRaWAN is a long-range, low-power radio link with a throughput in
the **0.3–50 kbit/s** range and strict duty-cycle limits. A single
720p video stream needs roughly a thousand times more bandwidth, and
even one 5-second MP4 clip (~1 MB) would take minutes to hours to
transfer. Pushing video through LoRa is physically impossible, so the
system splits traffic by what each link is good at:

| Traffic                          | Link              | Why |
|----------------------------------|-------------------|-----|
| Continuous camera streams (RTSP) | IP network        | Bandwidth |
| MP4 alert clips to control center| IP network (HTTPS)| Bandwidth |
| Alert *events* (compact JSON)    | LoRa gateway (MQTT)| Reach every campus receiver, works when IP uplink is congested |
| Camera health telemetry          | LoRa gateway (MQTT)| Tiny periodic payloads — LoRa's sweet spot |

The LoRa gateway exposes an MQTT broker; the server publishes to it
with `paho-mqtt` (auto-reconnecting background loop):

- `watchdog/alerts/<camera_id>` — one message per alert:
  `{id, timestamp, confidence, camera_id, alert_type, clip_path}`.
  The clip itself stays on the server; `clip_path` tells the control
  center what to fetch over IP if the HTTPS push hasn't delivered it.
- `watchdog/telemetry/<camera_id>` — every `TELEMETRY_INTERVAL`
  seconds: `{online, healthy, label, counts, ts}`.

## Components

### Capture (`src/capture/camera.py`)
One `Camera` per configured source (webcam index, video file, or
RTSP/HTTP URL). Applies configured resolution/FPS, keeps only the
freshest frame for live streams (`CAP_PROP_BUFFERSIZE=1`), loops video
files for demos, and reopens dropped streams with exponential backoff.
Each captured frame gets a sequence number so consumers never process
the same frame twice, and a timestamp that drives health monitoring.

### Per-camera pipeline (`src/pipeline.py`)
Two threads per camera:

- **capture** — reads at full source speed, feeds the latest-frame slot
  and the clip recorder's rolling pre-event buffer.
- **analysis** — for every *new* frame: ViT violence classification
  (temporal smoothing over N distinct frames), YOLO + ByteTrack
  people/vehicle tracking, and behavior analytics. Publishes status
  (label, scores, counts, boxes, events) into a registry the dashboard
  reads.

The ViT violence model is shared across cameras behind an inference
lock; YOLO trackers are per camera because track-ID state belongs to
one stream.

### AI models

| Task | Model | Notes |
|------|-------|-------|
| Violence detection | ViT (`jaranohaal/vit-base-violence-detection`) | per-frame classification, temporal smoothing |
| People/vehicle detection | YOLOv8n (ultralytics) | COCO classes filtered to person + car/truck/bus/motorcycle/bicycle |
| Multi-object tracking | ByteTrack | persistent track IDs → unique counting + trajectories |
| Abnormal behavior | rules + IsolationForest (scikit-learn) | loitering, running, statistical movement anomalies |

Behavior analytics compute per-track motion features (speed, path
tortuosity, dwell time, radius — normalized by frame diagonal) with
pandas, and combine explainable rules with an unsupervised
IsolationForest fitted on each camera's own history, so "abnormal"
adapts to what each camera normally sees.

### Alerts (`src/alerts/`)
A confirmed event drives the `ClipRecorder` state machine
(IDLE → RECORDING → SAVING): pre-event buffer + live frames + post-event
tail are encoded to MP4 (writer validated — no alert ships without a
playable clip). `AlertManager` applies per-camera cooldowns, persists
the alert, and fans out to notifiers:

- `ControlCenterNotifier` — HTTPS POST (multipart: alert JSON + MP4)
  with API key, bounded queue, and exponential-backoff retries.
- `MqttGatewayClient` — compact event on the LoRa gateway.

### Storage (`src/alerts/storage.py`, `mongo_storage.py`)
MongoDB is the preferred backend (`DB_BACKEND=auto` uses it whenever
reachable); a thread-safe SQLite backend is the automatic fallback so
alerting never stops if the database server is down. Both expose the
same interface with integer alert ids.

### Dashboard (`src/dashboard/`)
FastAPI + Jinja2 + MJPEG. Live grid with one panel per camera (health
badge, violence status, people/vehicle counts, tracked-object box
overlay), paginated alert table with clip playback and filters, and a
JSON API (see `docs/api/openapi.json` / the Postman collection).
Optional token auth (`API_TOKEN`) covers every route: header for API
clients, one-time `?token=` for browsers (converted to an HttpOnly
session cookie).

## Data flow for one alert

1. Camera streams RTSP → capture thread hands frame to analysis.
2. ViT flags violence on N consecutive distinct frames (or behavior
   analytics emit a loitering/running/anomaly event).
3. ClipRecorder writes pre+during+post MP4; AlertManager stores the
   alert in MongoDB (camera id, type, confidence, clip path).
4. ControlCenterNotifier POSTs alert + clip to the municipal control
   center; MqttGatewayClient publishes the compact event via LoRa.
5. Dashboard shows the alert immediately; operators can review or
   delete it.

## Deployment notes

- **Server**: one machine runs `main.py` (GPU optional — CUDA is
  auto-detected). Both cameras stream to it over the campus IP network.
- **Scaling**: each camera adds two threads and one YOLO instance;
  the ViT model is shared. Two cameras on CPU ≈ 2-4 analysis fps per
  camera, which is sufficient because detection operates on smoothed
  streaks, while capture, clips, and MJPEG run at full frame rate.
- **Security**: set `API_TOKEN`, run behind HTTPS (reverse proxy), and
  scope `CONTROL_CENTER_API_KEY` per deployment. Video of public spaces
  should be retained only as alert clips (delete from the dashboard).
