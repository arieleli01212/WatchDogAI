# WatchDogAI

Real-time violence detection system using AI.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run (uses webcam by default)
python main.py

# Open dashboard
# http://localhost:8000
```

## Configuration

Set via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| CAMERA_SOURCE | 0 | Webcam index or video file path |
| CONFIDENCE_THRESHOLD | 0.85 | Detection threshold |
| COOLDOWN_SECONDS | 5 | Alert cooldown period |
| DASHBOARD_PORT | 8000 | Web dashboard port |
| MODEL_PATH | models/violence_detector.pt | Model weights file |
