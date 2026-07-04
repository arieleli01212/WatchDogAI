"""Dashboard routes: pages, API endpoints, and per-camera MJPEG video feeds."""

from __future__ import annotations

import asyncio
from pathlib import Path as FilePath
from typing import Optional

import cv2
from fastapi import APIRouter, Query, Request, Path
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

router = APIRouter()

# Upper bound for pagination params: unbounded ints overflow SQLite's
# 64-bit OFFSET binding into an HTTP 500
MAX_OFFSET = 1_000_000

IDLE_STATUS = {
    "label": "normal",
    "confidence": 0.0,
    "violence_score": 0.0,
    "streak": 0,
    "required": 0,
    "counts": {},
    "objects": [],
    "last_update": None,
}

# BGR overlay colors per tracked-object category
OVERLAY_COLORS = {"person": (80, 200, 120), "vehicle": (60, 140, 255)}


def _camera_snapshot(request: Request) -> dict:
    """Build a per-camera status map: identity, health, and latest detection."""
    cameras = request.app.state.cameras
    status_registry = request.app.state.camera_status
    max_age = request.app.state.settings.camera_health_max_age

    snapshot = {}
    for camera_id, camera in cameras.items():
        snapshot[camera_id] = {
            "id": camera_id,
            "name": camera.name,
            "source_type": camera.source_type.value,
            "online": camera.is_opened(),
            "healthy": camera.is_healthy(max_age),
            "detection": status_registry.get(camera_id, IDLE_STATUS),
        }
    return snapshot


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


@router.get("/", response_class=HTMLResponse)
async def live_view(request: Request) -> HTMLResponse:
    """Render the live camera grid page."""
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "live.html",
        {
            "cameras": _camera_snapshot(request),
        },
    )


def _clip_url(clip_path: str, clip_dir: str) -> Optional[str]:
    """Map a stored clip path to its /clips URL (None when outside clip_dir)."""
    try:
        rel = FilePath(clip_path).resolve().relative_to(FilePath(clip_dir).resolve())
    except (ValueError, OSError):
        return None
    return "/clips/" + rel.as_posix()


@router.get("/alerts", response_class=HTMLResponse)
async def alerts_page(
    request: Request,
    page: int = Query(1, ge=1, le=MAX_OFFSET),
    status: Optional[str] = Query(None),
    camera_id: Optional[str] = Query(None),
) -> HTMLResponse:
    """Render the alerts table page with pagination and filters."""
    per_page = 20
    offset = (page - 1) * per_page

    alert_manager = request.app.state.alert_manager
    if alert_manager is not None:
        alerts = await asyncio.to_thread(
            alert_manager.get_alerts,
            limit=per_page, offset=offset, status=status, camera_id=camera_id,
        )
        total = await asyncio.to_thread(
            alert_manager.get_alert_count, status=status, camera_id=camera_id,
        )
    else:
        alerts = []
        total = 0

    clip_dir = request.app.state.settings.clip_dir
    for alert in alerts:
        alert["clip_url"] = _clip_url(alert.get("clip_path", ""), clip_dir)

    total_pages = max(1, (total + per_page - 1) // per_page)

    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "alerts.html",
        {
            "alerts": alerts,
            "page": page,
            "total_pages": total_pages,
            "status_filter": status,
            "camera_filter": camera_id,
            "camera_ids": list(request.app.state.cameras.keys()),
        },
    )


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------


@router.get("/api/status")
async def api_status(request: Request) -> JSONResponse:
    """Return overall system status and per-camera state as JSON."""
    cameras = _camera_snapshot(request)
    alert_manager = request.app.state.alert_manager
    alert_count = (
        await asyncio.to_thread(lambda: alert_manager.alert_count)
        if alert_manager else 0
    )

    healthy = sum(1 for cam in cameras.values() if cam["healthy"])
    if not cameras or healthy == 0:
        status = "inactive"
    elif healthy < len(cameras):
        status = "degraded"
    else:
        status = "active"

    return JSONResponse(
        {
            "status": status,
            "cameras": cameras,
            "alert_count": alert_count,
        }
    )


@router.get("/api/cameras")
async def api_cameras(request: Request) -> JSONResponse:
    """Return the configured cameras and their health."""
    return JSONResponse(list(_camera_snapshot(request).values()))


@router.get("/api/counts")
async def api_counts(request: Request) -> JSONResponse:
    """Return live people/vehicle counts per camera."""
    status_registry = request.app.state.camera_status
    return JSONResponse(
        {
            camera_id: status_registry.get(camera_id, {}).get("counts", {})
            for camera_id in request.app.state.cameras
        }
    )


@router.get("/api/alerts")
async def api_alerts(
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0, le=MAX_OFFSET),
    status: Optional[str] = Query(None),
    camera_id: Optional[str] = Query(None),
) -> JSONResponse:
    """Return alerts as a JSON list with pagination and filters."""
    alert_manager = request.app.state.alert_manager
    if alert_manager is not None:
        alerts = await asyncio.to_thread(
            alert_manager.get_alerts,
            limit=limit, offset=offset, status=status, camera_id=camera_id,
        )
    else:
        alerts = []
    return JSONResponse(alerts)


@router.delete("/api/alerts/{alert_id}")
async def api_delete_alert(
    request: Request,
    alert_id: int = Path(..., ge=1),
) -> JSONResponse:
    """Delete an alert and its clip file."""
    alert_manager = request.app.state.alert_manager
    if alert_manager is None:
        return JSONResponse({"error": "alert manager not available"}, status_code=503)
    deleted = await asyncio.to_thread(alert_manager.delete_alert, alert_id)
    if not deleted:
        return JSONResponse({"error": "alert not found"}, status_code=404)
    return JSONResponse({"ok": True})


# ---------------------------------------------------------------------------
# MJPEG video feeds
# ---------------------------------------------------------------------------


def _encode_with_overlay(frame, objects: list[dict]) -> bytes:
    """Draw tracked-object boxes on a copy of the frame and JPEG-encode it."""
    if objects:
        frame = frame.copy()
        for obj in objects:
            x1, y1, x2, y2 = (int(v) for v in obj["box"])
            color = OVERLAY_COLORS.get(obj["category"], (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{obj['label']} #{obj['track_id']}",
                (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    _, buffer = cv2.imencode(".jpg", frame)
    return buffer.tobytes()


async def _generate_frames(request: Request, camera_id: str):
    """Yield JPEG frames for one camera's MJPEG stream."""
    while True:
        if await request.is_disconnected():
            break
        camera = request.app.state.cameras.get(camera_id)
        frame = camera.get_latest_frame() if camera else None
        if frame is not None:
            status = request.app.state.camera_status.get(camera_id, {})
            objects = status.get("objects", [])
            # Encode off the event loop so DB queries and other feeds aren't blocked
            payload = await asyncio.to_thread(_encode_with_overlay, frame, objects)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
            )
        await asyncio.sleep(0.033)  # ~30 fps


@router.get("/video_feed/{camera_id}")
async def video_feed(request: Request, camera_id: str) -> StreamingResponse:
    """Stream one camera's live frames as MJPEG."""
    if camera_id not in request.app.state.cameras:
        return JSONResponse({"error": "unknown camera"}, status_code=404)
    return StreamingResponse(
        _generate_frames(request, camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/video_feed")
async def video_feed_default(request: Request) -> StreamingResponse:
    """Stream the first configured camera (legacy single-camera route)."""
    cameras = request.app.state.cameras
    camera_id = next(iter(cameras), None)
    if camera_id is None:
        return JSONResponse({"error": "no cameras configured"}, status_code=404)
    return StreamingResponse(
        _generate_frames(request, camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
