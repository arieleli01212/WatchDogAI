# WatchDogAI

Real-time violence detection system using AI. Captures video from cameras or video files, detects violence using a pre-trained Vision Transformer (ViT) model, records short MP4 video clips around detected events, and displays results on a live web dashboard.

## Features

- **Real-time violence detection** using [jaranohaal/vit-base-violence-detection](https://huggingface.co/jaranohaal/vit-base-violence-detection) (98.8% accuracy)
- **Video clip recording** — captures ~3s before + during + ~2s after violence events as MP4 clips
- **Live MJPEG video feed** with detection status overlay (green/red banner)
- **Alert management** with SQLite storage, video playback, and delete functionality
- **Web dashboard** with live view and paginated alerts table
- **REST API** for integration (`GET /api/status`, `GET /api/alerts`, `DELETE /api/alerts/{id}`)
- **Temporal smoothing** — requires N consecutive high-confidence detections before alerting
- Configurable via environment variables or `.env` file
- GPU auto-detection with CPU fallback

## Quick Start

```bash
# Clone the repository
git clone https://github.com/arieleli01212/WatchDogAI.git
cd WatchDogAI

# Create virtual environment and install dependencies
python -m venv venv
source venv/Scripts/activate   # Windows (Git Bash)
# source venv/bin/activate     # Linux/macOS
pip install -r requirements.txt

# Run with webcam
python main.py

# Or run with a video file
CAMERA_SOURCE=path/to/video.mp4 python main.py

# Open dashboard at http://localhost:8000
```

## Architecture

```
Camera Feed → [Capture Loop] → Frame Buffer + ClipRecorder (full FPS)
                                       ↓
                               [Detection Loop] → ViT Model (per-frame inference)
                                       ↓
                               [ClipRecorder] → State Machine (IDLE → RECORDING → SAVING)
                                       ↓
                               [Alert Manager] → SQLite DB + MP4 Clips
                                       ↓
                               [Web Dashboard] → Live Feed + Alerts + Video Playback
```

The **capture loop** reads frames at full camera FPS and feeds them to both the camera buffer (for the detector) and the ClipRecorder (for smooth video clips). The **detection loop** runs ViT inference on the latest frame and signals the ClipRecorder when violence is confirmed. The **ClipRecorder** state machine manages pre-event buffering, recording, and MP4 writing in a background thread.

### Project Structure

```
WatchDogAI/
├── main.py                          # Entry point, thread orchestration
├── src/
│   ├── config.py                    # Settings from environment variables
│   ├── capture/
│   │   └── camera.py                # OpenCV capture with thread-safe frame buffer
│   ├── detector/
│   │   └── model.py                 # ViT-based violence classifier
│   ├── alerts/
│   │   ├── clip_recorder.py         # State machine for recording MP4 clips
│   │   ├── manager.py               # Alert creation, deletion, cooldown
│   │   └── storage.py               # SQLite backend
│   └── dashboard/
│       ├── app.py                   # FastAPI application factory
│       ├── routes.py                # Routes and API endpoints
│       └── templates/               # Jinja2 HTML templates
├── data/
│   ├── clips/                       # Saved MP4 video clips (auto-created)
│   └── watchdog.db                  # SQLite database (auto-created)
├── models/                          # ViT model (auto-downloaded from HuggingFace)
└── requirements.txt
```

## Configuration

Set via environment variables or a `.env` file in the project root:

| Variable | Default | Description |
|---|---|---|
| `CAMERA_SOURCE` | `0` | Webcam index (int) or video file path |
| `CONFIDENCE_THRESHOLD` | `0.85` | Violence confidence threshold for alerts |
| `CONSECUTIVE_HITS` | `3` | Consecutive detections required before alerting |
| `COOLDOWN_SECONDS` | `5` | Minimum seconds between alerts |
| `PRE_EVENT_SECONDS` | `3` | Seconds of video to keep before violence |
| `POST_EVENT_SECONDS` | `2` | Seconds of video to record after violence ends |
| `CLIP_DIR` | `data/clips` | Directory for saved MP4 clips |
| `DASHBOARD_PORT` | `8000` | Web dashboard port |
| `DB_PATH` | `data/watchdog.db` | SQLite database path |
| `LOG_DIR` | `logs` | Log file directory |
| `LOG_LEVEL` | `INFO` | Logging level |

## Dashboard

- **Live View** (`/`) — Camera feed with real-time detection status banner and confidence score
- **Alerts** (`/alerts`) — Paginated table with timestamps, confidence, status, video playback, and delete buttons
- **API**:
  - `GET /api/status` — System state and detection info
  - `GET /api/alerts` — Alert list as JSON (supports `limit`, `offset`, `status` params)
  - `DELETE /api/alerts/{id}` — Delete an alert and its video clip

## Tech Stack

Python 3.10+ | PyTorch | timm | OpenCV | FastAPI | SQLite | Jinja2
