"""Tests for the dashboard routes."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from src.alerts.manager import AlertManager
from src.capture.camera import SourceType
from src.config import Settings
from src.dashboard.app import create_app


class FakeCamera:
    """Minimal stand-in for capture.Camera used by the routes."""

    def __init__(self, name="Fake Cam", healthy=True, opened=True, frame=None):
        self.name = name
        self.source_type = SourceType.WEBCAM
        self._healthy = healthy
        self._opened = opened
        self._frame = frame

    def is_opened(self):
        return self._opened

    def is_healthy(self, max_age=5.0):
        return self._healthy

    def get_latest_frame(self):
        return self._frame


@pytest.fixture()
def app(tmp_path):
    """Create a fresh app instance for each test."""
    settings = Settings(
        clip_dir=str(tmp_path / "clips"),
        db_path=str(tmp_path / "db.sqlite"),
    )
    return create_app(settings)


@pytest.fixture()
def client(app):
    """Provide an async HTTP client wired to the app."""
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://testserver")


# ------------------------------------------------------------------
# Page routes
# ------------------------------------------------------------------


@pytest.mark.anyio
async def test_home_page_returns_200(app, client):
    app.state.cameras["cam0"] = FakeCamera()
    async with client as ac:
        resp = await ac.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "WatchDogAI" in resp.text
    assert "/video_feed/cam0" in resp.text


@pytest.mark.anyio
async def test_home_page_without_cameras(client):
    async with client as ac:
        resp = await ac.get("/")
    assert resp.status_code == 200
    assert "No cameras configured" in resp.text


@pytest.mark.anyio
async def test_alerts_page_returns_200(client):
    async with client as ac:
        resp = await ac.get("/alerts")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Alerts" in resp.text


# ------------------------------------------------------------------
# JSON API
# ------------------------------------------------------------------


@pytest.mark.anyio
async def test_api_status_inactive_without_cameras(client):
    async with client as ac:
        resp = await ac.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "inactive"
    assert data["cameras"] == {}
    assert data["alert_count"] == 0


@pytest.mark.anyio
async def test_api_status_active_with_healthy_cameras(app, client):
    app.state.cameras["cam0"] = FakeCamera(name="North")
    app.state.cameras["cam1"] = FakeCamera(name="South")
    app.state.camera_status["cam0"] = {"label": "violence", "violence_score": 0.9}

    async with client as ac:
        resp = await ac.get("/api/status")
    data = resp.json()
    assert data["status"] == "active"
    assert set(data["cameras"].keys()) == {"cam0", "cam1"}
    assert data["cameras"]["cam0"]["name"] == "North"
    assert data["cameras"]["cam0"]["healthy"] is True
    assert data["cameras"]["cam0"]["detection"]["label"] == "violence"
    # Camera without a status yet gets the idle default
    assert data["cameras"]["cam1"]["detection"]["label"] == "normal"


@pytest.mark.anyio
async def test_api_status_degraded_with_one_unhealthy_camera(app, client):
    app.state.cameras["cam0"] = FakeCamera(healthy=True)
    app.state.cameras["cam1"] = FakeCamera(healthy=False)

    async with client as ac:
        resp = await ac.get("/api/status")
    assert resp.json()["status"] == "degraded"


@pytest.mark.anyio
async def test_api_cameras_lists_cameras(app, client):
    app.state.cameras["cam0"] = FakeCamera(name="North")
    async with client as ac:
        resp = await ac.get("/api/cameras")
    data = resp.json()
    assert len(data) == 1
    assert data[0]["id"] == "cam0"
    assert data[0]["name"] == "North"
    assert data[0]["source_type"] == "webcam"


@pytest.mark.anyio
async def test_api_counts(app, client):
    app.state.cameras["cam0"] = FakeCamera()
    app.state.cameras["cam1"] = FakeCamera()
    app.state.camera_status["cam0"] = {
        "counts": {"people": 3, "vehicles": 1, "unique_people": 8, "unique_vehicles": 2},
    }

    async with client as ac:
        resp = await ac.get("/api/counts")
    data = resp.json()
    assert data["cam0"]["people"] == 3
    assert data["cam0"]["unique_vehicles"] == 2
    assert data["cam1"] == {}  # no analysis results yet


@pytest.mark.anyio
async def test_api_alerts_returns_list(client):
    async with client as ac:
        resp = await ac.get("/api/alerts")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 0  # no alert manager attached


@pytest.mark.anyio
async def test_api_alerts_with_alert_manager(app, client, tmp_path):
    settings = Settings(
        db_backend="sqlite",
        db_path=str(tmp_path / "alerts.db"),
        cooldown_seconds=0,
    )
    manager = AlertManager(settings)
    manager.on_clip_saved(confidence=0.9, clip_path="a.mp4", camera_id="cam0")
    manager.on_clip_saved(confidence=0.8, clip_path="b.mp4", camera_id="cam1")
    app.state.alert_manager = manager

    async with client as ac:
        resp = await ac.get("/api/alerts")
        assert len(resp.json()) == 2

        resp = await ac.get("/api/alerts?camera_id=cam1")
        data = resp.json()
        assert len(data) == 1
        assert data[0]["camera_id"] == "cam1"
        assert data[0]["alert_type"] == "violence"

    manager.storage.close()


@pytest.mark.anyio
async def test_api_delete_alert(app, client, tmp_path):
    settings = Settings(
        db_backend="sqlite",
        db_path=str(tmp_path / "alerts.db"),
        cooldown_seconds=0,
    )
    manager = AlertManager(settings)
    manager.on_clip_saved(confidence=0.9, clip_path="a.mp4", camera_id="cam0")
    app.state.alert_manager = manager
    alert_id = manager.get_alerts()[0]["id"]

    async with client as ac:
        resp = await ac.delete(f"/api/alerts/{alert_id}")
        assert resp.status_code == 200
        resp = await ac.delete(f"/api/alerts/{alert_id}")
        assert resp.status_code == 404

    manager.storage.close()


# ------------------------------------------------------------------
# Video feeds
# ------------------------------------------------------------------


@pytest.mark.anyio
async def test_video_feed_unknown_camera_404(app, client):
    async with client as ac:
        resp = await ac.get("/video_feed/nope")
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_video_feed_default_404_without_cameras(client):
    async with client as ac:
        resp = await ac.get("/video_feed")
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_video_feed_returns_mjpeg_content_type(app):
    """Verify the per-camera feed returns the MJPEG content type.

    Because the MJPEG stream is infinite we use a manual ASGI call and
    only inspect the response-start message (headers).
    """
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    app.state.cameras["cam0"] = FakeCamera(frame=frame)

    transport = ASGITransport(app=app)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "path": "/video_feed/cam0",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"testserver")],
    }

    status_code = None
    content_type = None
    got_headers = asyncio.Event()

    async def receive():
        # Keep the connection open until we have what we need
        await asyncio.sleep(10)
        return {"type": "http.disconnect"}

    async def send(message):
        nonlocal status_code, content_type
        if message["type"] == "http.response.start":
            status_code = message["status"]
            for name, value in message.get("headers", []):
                if name == b"content-type":
                    content_type = value.decode()
            got_headers.set()

    task = asyncio.create_task(app(scope, receive, send))
    try:
        await asyncio.wait_for(got_headers.wait(), timeout=5.0)
        assert status_code == 200
        assert content_type is not None
        assert "multipart/x-mixed-replace" in content_type
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
