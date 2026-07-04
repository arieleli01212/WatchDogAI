"""Tests for the dashboard routes."""

from __future__ import annotations

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from src.dashboard.app import create_app


@pytest.fixture()
def app():
    """Create a fresh app instance for each test."""
    return create_app()


@pytest.fixture()
async def client(app):
    """Provide an async HTTP client wired to the app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


# ------------------------------------------------------------------
# Page routes
# ------------------------------------------------------------------


@pytest.mark.anyio
async def test_home_page_returns_200(client):
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "WatchDogAI" in resp.text


@pytest.mark.anyio
async def test_alerts_page_returns_200(client):
    resp = await client.get("/alerts")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Alerts" in resp.text


# ------------------------------------------------------------------
# JSON API
# ------------------------------------------------------------------


@pytest.mark.anyio
async def test_api_status_returns_json(client):
    resp = await client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "detection" in data
    assert "alert_count" in data
    assert data["status"] == "inactive"  # no camera attached in tests
    assert data["detection"]["label"] == "normal"


@pytest.mark.anyio
async def test_api_alerts_returns_list(client):
    resp = await client.get("/api/alerts")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 0  # no alert manager attached


# ------------------------------------------------------------------
# Video feed
# ------------------------------------------------------------------


@pytest.mark.anyio
async def test_video_feed_returns_mjpeg_content_type(app):
    """Verify the video feed endpoint returns the correct MJPEG content type.

    Because the MJPEG stream is infinite we use a manual ASGI call and
    only inspect the response-start message (headers).
    """
    transport = ASGITransport(app=app)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "path": "/video_feed",
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
