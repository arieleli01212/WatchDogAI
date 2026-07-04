# WatchDogAI — Installation & Setup Guide

Complete guide for installing prerequisites, setting up the system, and
verifying every component — from a bare machine to a running two-camera
deployment.

---

## 1. Prerequisites

### 1.1 Hardware

| Component | Minimum | Recommended |
|---|---|---|
| CPU | 4 cores | 8+ cores (each camera runs its own analysis) |
| RAM | 8 GB | 16 GB |
| Disk | 10 GB free | SSD; clips accumulate under `data/clips/` |
| GPU | none (CPU works) | NVIDIA GPU with CUDA for faster inference |
| Camera(s) | any webcam or video file for testing | 2× IP cameras with RTSP (H.264), reachable over the campus IP network |

> **Note on inference speed:** on CPU, expect ~2–4 analyzed frames/sec
> per camera (ViT + YOLO). That is sufficient — detection works on
> smoothed streaks while capture, clip recording, and the live dashboard
> run at full frame rate. A CUDA GPU raises analysis to real-time.

### 1.2 Software

| Requirement | Version | Required? | Notes |
|---|---|---|---|
| Python | 3.10 – 3.13 | **yes** | 3.13 is what the project is tested on |
| pip + venv | bundled with Python | **yes** | |
| Git | any recent | **yes** | to clone the repository |
| MongoDB Community Server | 6.0+ | recommended | preferred alert storage; the system auto-falls back to SQLite without it |
| MQTT broker (LoRa gateway) | any MQTT 3.1.1/5 broker | optional | campus gateway in production; Mosquitto for local testing |
| FFmpeg | — | no | OpenCV wheels bundle their own codecs |

Supported operating systems: **Windows 10/11**, **Ubuntu 20.04+**,
**macOS 12+**. Commands below show Windows (PowerShell / Git Bash) and
Linux variants.

### 1.3 Network

- The **cameras → server** path is plain IP (RTSP over TCP 554 by
  default). Make sure the server can reach each camera's RTSP URL.
- The **LoRa gateway** exposes an MQTT broker (default port 1883).
- The **dashboard** listens on port 8000 (configurable) — open it in
  your firewall for operator machines only.
- First run needs **internet access** to download the AI models
  (~360 MB total, one time).

---

## 2. Installation

### 2.1 Get the code

```bash
git clone https://github.com/arieleli01212/WatchDogAI.git
cd WatchDogAI
```

### 2.2 Create a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**Windows (Git Bash):**
```bash
python -m venv venv
source venv/Scripts/activate
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2.3 Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This downloads ~2–3 GB (PyTorch is the bulk of it). By default you get
the **CPU** build of PyTorch, which works everywhere.

**Optional — NVIDIA GPU acceleration:** install the CUDA build of
PyTorch *before* the requirements file (pick the CUDA version matching
your driver at <https://pytorch.org/get-started/locally/>):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Verify GPU detection:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 2.4 Verify the installation

Run the test suite — it is fully offline (all models are mocked), so
this is a safe smoke test of the whole stack:

```bash
python -m pytest
```

Expected: **225 passed**. If anything fails here, fix the environment
before continuing (see Troubleshooting, section 7).

### 2.5 First-run model downloads (one time, automatic)

On the first real start, the system downloads:

- **ViT violence classifier** (`jaranohaal/vit-base-violence-detection`,
  ~350 MB) from HuggingFace into `models/` / the HF cache.
- **YOLOv8n** (`yolov8n.pt`, ~6 MB) from Ultralytics into the project
  directory.

No account or API key is needed. If the server has no internet, run
once on a connected machine and copy the `models/` directory and
`yolov8n.pt` over.

---

## 3. Configuration

All settings live in a `.env` file (or plain environment variables).
Start from the annotated template:

```bash
cp .env.example .env
```

### 3.1 Quick test — single webcam, no external services

An empty `.env` works out of the box: webcam 0, SQLite storage,
dashboard on port 8000, no MQTT / control-center push. Just run:

```bash
python main.py
# open http://localhost:8000
```

You can also point it at a video file for repeatable demos:

```
CAMERA_SOURCE=path/to/test_video.mp4
```

### 3.2 Production — two RTSP cameras

```env
CAMERAS=[{"id": "cam-north", "name": "North Gate", "source": "rtsp://user:pass@10.0.0.11:554/stream1", "width": 1280, "height": 720, "fps": 15}, {"id": "cam-south", "name": "South Gate", "source": "rtsp://user:pass@10.0.0.12:554/stream1", "width": 1280, "height": 720, "fps": 15}]
```

- `id` — letters/digits/`-`/`_` only (it appears in filenames, URLs,
  and MQTT topics).
- `source` — RTSP/HTTP URL, webcam index, or file path. Dropped streams
  reconnect automatically with backoff.
- `width`/`height`/`fps` — optional quality hints applied to the source.

### 3.3 MongoDB (preferred storage)

Install MongoDB Community Server:

- **Windows:** download the MSI from
  <https://www.mongodb.com/try/download/community>, install as a
  service (default settings are fine).
- **Ubuntu:** `sudo apt install mongodb-org` (after adding MongoDB's
  repo), then `sudo systemctl enable --now mongod`.
- **Docker (any OS):** `docker run -d -p 27017:27017 --name mongo mongo:7`

Then in `.env`:

```env
DB_BACKEND=auto                              # auto = Mongo when reachable, SQLite otherwise
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=watchdog
```

`DB_BACKEND=mongodb` makes MongoDB mandatory (startup fails loudly if
unreachable); `auto` keeps alerting alive on SQLite during outages.

### 3.4 Municipal control center (HTTPS alert push)

```env
CONTROL_CENTER_URL=https://control-center.holon.example/api/alerts
CONTROL_CENTER_API_KEY=<key issued by the control center>
```

Every alert is POSTed as `multipart/form-data`: field `alert` (JSON)
plus field `clip` (the MP4). Deliveries retry with exponential backoff;
failures never block detection.

### 3.5 LoRa gateway (MQTT)

```env
MQTT_HOST=gateway.campus.hit.ac.il
MQTT_PORT=1883
MQTT_USERNAME=...        # if the broker requires it
MQTT_PASSWORD=...
MQTT_BASE_TOPIC=watchdog
TELEMETRY_INTERVAL=30
```

Published topics:

- `watchdog/alerts/<camera_id>` — compact JSON per alert
- `watchdog/telemetry/<camera_id>` — health + counts every 30 s

**Local testing without the campus gateway** — run Mosquitto:

```bash
docker run -d -p 1883:1883 eclipse-mosquitto:2 mosquitto -c /mosquitto-no-auth.conf
# watch the traffic:
docker exec -it <container> mosquitto_sub -t 'watchdog/#' -v
```

### 3.6 Dashboard security (recommended)

```env
API_TOKEN=<long random string>
```

- Browsers: open `http://server:8000/?token=<value>` once — you get a
  session cookie and the token is stripped from the URL.
- API clients / Postman: send the `X-API-Token` header.
- For anything internet-facing, put the dashboard behind an HTTPS
  reverse proxy (nginx/Caddy).

### 3.7 Detection tuning (optional)

| Variable | Default | Meaning |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | 0.85 | violence confidence needed per frame |
| `CONSECUTIVE_HITS` | 3 | distinct frames to confirm violence |
| `COOLDOWN_SECONDS` | 5 | per-camera gap between alerts |
| `YOLO_MODEL` | yolov8n.pt | swap for yolov8s/m for accuracy vs speed |
| `LOITER_SECONDS` | 60 | dwell time before a loitering alert |
| `RUN_SPEED_THRESHOLD` | 0.35 | running speed, in frame-diagonals/sec |
| `ANOMALY_MIN_SAMPLES` | 200 | history needed before the anomaly model activates |
| `PRE_EVENT_SECONDS` / `POST_EVENT_SECONDS` | 3 / 2 | clip padding |
| `MAX_CLIP_SECONDS` | 30 | long events are chunked into clips of this length |

---

## 4. Running

```bash
# with the venv activated
python main.py
```

Startup sequence you should see in the logs: model load → one pipeline
per camera → dashboard on `http://0.0.0.0:8000`.

Open **http://localhost:8000**:

- **Live View** — one panel per camera: MJPEG feed with tracked-object
  boxes, health dot, violence status bar, people/vehicle counts.
- **Alerts** — table with clip playback, filters, delete.

Stop with `Ctrl+C` (clean shutdown: pipelines join, notifier and MQTT
close).

### Run as a service (production)

**Linux (systemd)** — `/etc/systemd/system/watchdogai.service`:

```ini
[Unit]
Description=WatchDogAI smart security system
After=network-online.target mongod.service

[Service]
WorkingDirectory=/opt/WatchDogAI
ExecStart=/opt/WatchDogAI/venv/bin/python main.py
Restart=on-failure
EnvironmentFile=/opt/WatchDogAI/.env

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now watchdogai
```

**Windows** — use Task Scheduler ("At startup", run
`C:\path\WatchDogAI\venv\Scripts\python.exe main.py` with the project
folder as the working directory), or NSSM to wrap it as a service.

---

## 5. Verification checklist

Run through this after installation:

1. `python -m pytest` → 225 passed.
2. `python main.py` starts without errors; logs show every configured
   camera opening.
3. Dashboard live view shows all camera feeds with green health dots.
4. `GET /api/status` returns `"status": "active"` and per-camera state.
5. Walk in front of a camera → people count increments; a box with a
   track ID follows you.
6. Trigger a test alert (play a violence test clip to a camera, or
   temporarily set `CONFIDENCE_THRESHOLD=0.3`) → alert appears in
   `/alerts` with a playable MP4.
7. If MongoDB is configured: `mongosh watchdog --eval "db.alerts.countDocuments()"`
   grows with each alert.
8. If MQTT is configured: `mosquitto_sub -t 'watchdog/#' -v` shows
   telemetry every 30 s and a message per alert.
9. If the control center is configured: its endpoint receives the
   multipart POST (test first with <https://webhook.site> as
   `CONTROL_CENTER_URL`).

---

## 6. API quick reference

Import `docs/api/WatchDogAI.postman_collection.json` into Postman
(set `base_url` and `api_token` variables), or read
`docs/api/openapi.json`. Interactive docs are served at `/docs` when
the app is running.

| Endpoint | Purpose |
|---|---|
| `GET /api/status` | system + per-camera health and detection state |
| `GET /api/cameras` | configured cameras |
| `GET /api/counts` | live people/vehicle counts |
| `GET /api/alerts` | alert list (`limit`, `offset`, `status`, `camera_id`) |
| `DELETE /api/alerts/{id}` | delete alert + clip |
| `GET /video_feed/{camera_id}` | MJPEG live stream |

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `pip install` is very slow or huge | Normal — PyTorch is ~2 GB. Use a wired connection; consider the CPU-only wheel. |
| **Windows:** `ImportError: DLL load failed ... An Application Control policy has blocked this file` (scipy/sklearn) | Windows Smart App Control blocking a freshly downloaded DLL on first touch. Simply re-run the command — the block clears after the reputation check. If it persists, allow the file under Windows Security → App & browser control. |
| Camera shows "no signal" on the dashboard | Test the RTSP URL directly: `ffplay rtsp://...` or VLC. Check credentials in the URL, camera firewall, and that the stream is H.264. The system retries with backoff automatically — watch `logs/watchdog.log`. |
| Webcam won't open (`Failed to open video source: 0`) | Another app is holding the camera; or try index 1/2. On Windows, check camera privacy settings. |
| First run hangs on "Loading model" | It is downloading ~350 MB from HuggingFace. Check internet access / proxy. Offline servers: copy `models/` and `yolov8n.pt` from a connected machine. |
| `MongoDB unreachable — falling back to SQLite` in logs | Fine for testing. For production, start `mongod` and check `MONGODB_URI`. Set `DB_BACKEND=mongodb` to make it a hard requirement. |
| Port 8000 already in use | Set `DASHBOARD_PORT=8080` in `.env`. |
| 401 on every dashboard request | `API_TOKEN` is set — open `/?token=<value>` once in the browser, or send the `X-API-Token` header. |
| Analysis feels slow / high CPU | Expected on CPU. Lower camera fps (`"fps": 10`), keep `yolov8n`, or add a CUDA GPU. |
| Too many / too few violence alerts | Raise / lower `CONFIDENCE_THRESHOLD` and `CONSECUTIVE_HITS`. |
| Loitering alerts too eager | Raise `LOITER_SECONDS`; anomaly alerts only start after `ANOMALY_MIN_SAMPLES` observations per camera. |

Logs live in `logs/watchdog.log` (`LOG_LEVEL=DEBUG` for detail).

---

## 8. Project layout after a successful install

```
WatchDogAI/
├── venv/                 # your virtual environment (not committed)
├── .env                  # your configuration (not committed)
├── models/               # ViT weights (auto-downloaded)
├── yolov8n.pt            # YOLO weights (auto-downloaded)
├── data/
│   ├── clips/            # recorded alert clips (auto-created)
│   └── watchdog.db       # SQLite fallback DB (auto-created)
└── logs/watchdog.log
```
