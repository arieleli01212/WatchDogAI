"""FastAPI application factory for the WatchDogAI dashboard."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and return the configured FastAPI application."""
    app = FastAPI(title="WatchDogAI", version="1.0.0")

    settings = settings or get_settings()
    app.state.settings = settings

    templates_dir = Path(__file__).parent / "templates"
    app.state.templates = Jinja2Templates(directory=str(templates_dir))

    # Shared state – populated by the caller / main entry point.
    app.state.cameras = {}        # camera_id -> Camera
    app.state.camera_status = {}  # camera_id -> latest analysis status dict
    app.state.alert_manager = None

    # Mount clip files for serving video
    clips_dir = Path(settings.clip_dir)
    clips_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/clips", StaticFiles(directory=str(clips_dir)), name="clips")

    from src.dashboard.routes import router  # noqa: E402

    app.include_router(router)

    return app
