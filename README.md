# WatchDogAI

Real-time violence detection system using AI. Captures video from cameras or video files, detects violence using a pre-trained Vision Transformer (ViT) model, saves alert snapshots, and displays results on a live web dashboard.

## Features

- Real-time violence detection using [jaranohaal/vit-base-violence-detection](https://huggingface.co/jaranohaal/vit-base-violence-detection) (98.8% accuracy)
- Live MJPEG video feed with detection status overlay (green/red banner)
- Alert management with SQLite storage, JPEG snapshots, and cooldown deduplication
- Web dashboard with live view and paginated alerts table
- REST API for integration (`/api/status`, `/api/alerts`)
- Configurable via environment variables or `.env` file
- GPU auto-detection with CPU fallback

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run with webcam
python main.py

# Or run with a video file
CAMERA_SOURCE=path/to/video.mp4 python main.py

# Open dashboard
# http://localhost:8000
```

## Architecture

```
Camera Feed → [Capture Loop] → Frame Buffer (16-frame sliding window)
                                      ↓
                              [Detection Loop] → ViT Model (samples 4 frames)
                                      ↓
                              [Alert Manager] → SQLite DB + JPEG Snapshots
                                      ↓
                              [Web Dashboard] → Live Feed + Alerts Table
```

### Modules

- **`src/capture/`** — Camera reading via OpenCV with thread-safe sliding window buffer
- **`src/detector/`** — ViT-based violence classifier loaded via timm from HuggingFace
- **`src/alerts/`** — Alert management with SQLite storage and snapshot saving
- **`src/dashboard/`** — FastAPI web app with MJPEG streaming and Jinja2 templates
- **`src/config.py`** — Configuration via environment variables with sensible defaults

## Configuration

Set via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| CAMERA_SOURCE | 0 | Webcam index (int) or video file path |
| CONFIDENCE_THRESHOLD | 0.92 | Violence confidence threshold for alerts |
| CONSECUTIVE_HITS | 3 | Consecutive high-confidence detections required before alerting |
| COOLDOWN_SECONDS | 5 | Minimum seconds between alerts |
| DASHBOARD_PORT | 8000 | Web dashboard port |
| DB_PATH | data/watchdog.db | SQLite database path |
| SNAPSHOT_DIR | data/snapshots | Alert snapshot directory |
| LOG_DIR | logs | Log file directory |
| LOG_LEVEL | INFO | Logging level |

## Dashboard

- **Live View** (`/`) — Camera feed with real-time detection status banner and confidence score
- **Alerts** (`/alerts`) — Paginated table with timestamps, confidence, status, and snapshot thumbnails
- **API** — `GET /api/status` (system state) and `GET /api/alerts` (alert list as JSON)

## Tech Stack

Python 3.10+, PyTorch, timm, OpenCV, FastAPI, SQLite
