# WatchDogAI — Violence Detection System Design

## Overview

Real-time physical fight detection system using pre-trained AI models. Captures video from cameras (webcam for testing), detects violence, logs alerts with snapshots, and displays results on a web dashboard.

## Architecture

```
Camera Feed → [Detection Engine] → [Alert Manager] → [Web Dashboard]
                    │                      │
              Pre-trained model       Logs + SQLite
              (PyTorch)
```

### Core Modules

- **`capture/`** — Reads frames from webcam or video file. Handles frame buffering (16-frame sliding window clips).
- **`detector/`** — Loads pre-trained violence detection model. Classifies clips as violence/normal with confidence score. Runs at ~5-10 FPS inference.
- **`alerts/`** — Creates alerts when confidence exceeds threshold. Saves snapshots, logs to SQLite, deduplicates with cooldown period.
- **`dashboard/`** — FastAPI web app with live camera feed, real-time detection status, and alert history.

## Detection Engine

1. Camera feed read continuously via OpenCV (`cv2.VideoCapture`)
2. Frames collected into sliding window buffer (16 frames = ~0.5s at 30 FPS)
3. Every N frames, buffer passed to model for inference
4. Model outputs: `violence` or `normal` + confidence (0.0–1.0)
5. Alert triggered if confidence exceeds threshold (default 0.85)

### Model Selection (priority order)

1. HuggingFace video classification models fine-tuned on violence/fight datasets
2. Published PyTorch checkpoints (e.g., RWF-2000 trained models)
3. Fallback: lightweight action recognition model (MoViNet/X3D-S) for quick fine-tuning

### Key Decisions

- **Sliding window with stride** — process every 8th-16th frame batch for ~5-10 FPS inference
- **Configurable confidence threshold** — balance false positives vs missed detections
- **GPU optional** — auto-detects via `torch.cuda.is_available()`, falls back to CPU

## Alert Manager & Storage

### On Violence Detection

1. Save snapshot frame (JPEG) to `data/snapshots/`
2. Create SQLite record: timestamp, confidence, snapshot path, camera ID, status
3. Log to console + `logs/watchdog.log`

### Design Choices

- **SQLite** — zero setup, single file, easy for teammates, upgradeable later
- **Cooldown period** (default 5s) — prevents alert flooding for same incident
- **Plugin-ready** — alert manager designed for future email/API notification handlers

## Web Dashboard

### Tech Stack

- FastAPI + Jinja2 templates + vanilla CSS
- MJPEG streaming for live feed
- Optional HTMX for live-updating alerts

### Pages

- **Live view (`/`)** — camera feed stream, detection status banner (green/red), confidence score
- **Alerts (`/alerts`)** — table with timestamp, confidence, snapshot thumbnail, status. Filterable and paginated.

### API Endpoints

- `GET /api/alerts` — JSON alert list (for future integrations)
- `GET /api/status` — current detection state

## Project Structure

```
WatchDogAI/
├── src/
│   ├── __init__.py
│   ├── capture/
│   │   ├── __init__.py
│   │   └── camera.py
│   ├── detector/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   └── preprocessing.py
│   ├── alerts/
│   │   ├── __init__.py
│   │   ├── manager.py
│   │   └── storage.py
│   ├── dashboard/
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── routes.py
│   │   └── templates/
│   │       ├── base.html
│   │       ├── live.html
│   │       └── alerts.html
│   └── config.py
├── tests/
├── data/
├── logs/
├── models/
├── pyproject.toml
├── requirements.txt
└── main.py
```

## Configuration (`config.py`)

All settings configurable via environment variables or `.env`:

- Camera source (0 for webcam, or file path)
- Confidence threshold (default 0.85)
- Cooldown period (default 5 seconds)
- Clip length (default 16 frames)
- Dashboard port (default 8000)

## Runtime

`main.py` starts:
1. Detection loop in a background thread
2. FastAPI dashboard server

## Future Upgrade Path

- ONNX Runtime conversion for edge device performance
- Email/API notification handlers
- Multi-camera support
- Fine-tuning on custom dataset for improved accuracy
