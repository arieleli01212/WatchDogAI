"""Dashboard routes: pages, API endpoints, and MJPEG video feed."""

from __future__ import annotations

import asyncio
from typing import Optional

import cv2
from fastapi import APIRouter, Query, Request, Path
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

router = APIRouter()


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


@router.get("/", response_class=HTMLResponse)
async def live_view(request: Request) -> HTMLResponse:
    """Render the live camera feed page."""
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "live.html",
        {
            "detection": request.app.state.detector_status,
        },
    )


@router.get("/alerts", response_class=HTMLResponse)
async def alerts_page(
    request: Request,
    page: int = Query(1, ge=1),
    status: Optional[str] = Query(None),
) -> HTMLResponse:
    """Render the alerts table page with pagination."""
    per_page = 20
    offset = (page - 1) * per_page

    alert_manager = request.app.state.alert_manager
    if alert_manager is not None:
        alerts = alert_manager.get_alerts(limit=per_page, offset=offset, status=status)
        total = alert_manager.alert_count
    else:
        alerts = []
        total = 0

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
        },
    )


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------


@router.get("/api/status")
async def api_status(request: Request) -> JSONResponse:
    """Return system status as JSON."""
    detection = request.app.state.detector_status
    alert_manager = request.app.state.alert_manager
    camera = request.app.state.camera

    is_active = camera is not None
    alert_count = alert_manager.alert_count if alert_manager else 0

    return JSONResponse(
        {
            "status": "active" if is_active else "inactive",
            "detection": detection,
            "alert_count": alert_count,
        }
    )


@router.get("/api/alerts")
async def api_alerts(
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    status: Optional[str] = Query(None),
) -> JSONResponse:
    """Return alerts as a JSON list with pagination."""
    alert_manager = request.app.state.alert_manager
    if alert_manager is not None:
        alerts = alert_manager.get_alerts(limit=limit, offset=offset, status=status)
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
    deleted = alert_manager.delete_alert(alert_id)
    if not deleted:
        return JSONResponse({"error": "alert not found"}, status_code=404)
    return JSONResponse({"ok": True})


# ---------------------------------------------------------------------------
# MJPEG video feed
# ---------------------------------------------------------------------------


async def _generate_frames(request: Request):
    """Yield JPEG frames for the MJPEG stream."""
    while True:
        if await request.is_disconnected():
            break
        camera = request.app.state.camera
        frame = camera.get_latest_frame() if camera else None
        if frame is not None:
            _, buffer = cv2.imencode(".jpg", frame)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
        await asyncio.sleep(0.033)  # ~30 fps


@router.get("/video_feed")
async def video_feed(request: Request) -> StreamingResponse:
    """Stream live camera frames as MJPEG."""
    return StreamingResponse(
        _generate_frames(request),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
