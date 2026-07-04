"""FastAPI application factory for the WatchDogAI dashboard."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""
    app = FastAPI(title="WatchDogAI", version="0.1.0")

    templates_dir = Path(__file__).parent / "templates"
    app.state.templates = Jinja2Templates(directory=str(templates_dir))

    # Shared state – populated by the caller / main entry point.
    app.state.camera = None
    app.state.detector_status = {
        "label": "normal",
        "confidence": 0.0,
        "last_update": None,
    }
    app.state.alert_manager = None

    # Mount clip files for serving video
    clips_dir = Path("data/clips")
    clips_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/clips", StaticFiles(directory=str(clips_dir)), name="clips")

    from src.dashboard.routes import router  # noqa: E402

    app.include_router(router)

    return app
