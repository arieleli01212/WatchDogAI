"""FastAPI application factory for the WatchDogAI dashboard."""

from __future__ import annotations

import hmac
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.config import Settings, get_settings

TOKEN_COOKIE = "watchdog_token"


def _token_matches(provided: str, token: str) -> bool:
    """Constant-time token comparison that never raises on odd input."""
    return hmac.compare_digest(
        provided.encode("utf-8", "replace"), token.encode("utf-8", "replace")
    )


def _install_token_auth(app: FastAPI, token: str) -> None:
    """Require the API token on every request.

    Accepted as an ``X-API-Token`` header (API clients), a ``token``
    query parameter (first browser visit), or the session cookie the
    middleware sets after a successful query-parameter login — the
    cookie is what lets the dashboard's MJPEG <img> tags and fetch()
    calls authenticate without embedding the token in every URL. A
    query-parameter login is answered with a redirect that strips the
    token from the URL so it doesn't linger in history or logs.
    """

    @app.middleware("http")
    async def token_auth(request: Request, call_next):
        provided = (
            request.headers.get("X-API-Token")
            or request.query_params.get("token")
            or request.cookies.get(TOKEN_COOKIE)
            or ""
        )
        if not _token_matches(provided, token):
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        if request.method == "GET" and request.query_params.get("token") == token:
            response = RedirectResponse(
                request.url.remove_query_params("token"), status_code=303
            )
        else:
            response = await call_next(request)
        if request.query_params.get("token") == token:
            response.set_cookie(
                TOKEN_COOKIE, token, httponly=True, samesite="strict"
            )
        return response


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and return the configured FastAPI application."""
    app = FastAPI(title="WatchDogAI", version="1.0.0")

    settings = settings or get_settings()
    app.state.settings = settings

    if settings.api_token:
        _install_token_auth(app, settings.api_token)

    templates_dir = Path(__file__).parent / "templates"
    app.state.templates = Jinja2Templates(directory=str(templates_dir))

    # Shared state – populated by the caller / main entry point.
    app.state.cameras = {}        # camera_id -> Camera
    app.state.camera_status = {}  # camera_id -> latest analysis status dict
    app.state.alert_manager = None
    app.state.pipeline_manager = None  # runtime source-mode switching

    # Mount clip files for serving video
    clips_dir = Path(settings.clip_dir)
    clips_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/clips", StaticFiles(directory=str(clips_dir)), name="clips")

    from src.dashboard.routes import router  # noqa: E402

    app.include_router(router)

    return app
