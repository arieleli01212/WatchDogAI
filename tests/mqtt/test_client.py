"""Tests for the MQTT/LoRa gateway client and telemetry loop."""

from __future__ import annotations

import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from src.mqtt.client import MqttGatewayClient, TelemetryLoop


def _make_gateway(**kwargs):
    with patch("paho.mqtt.client.Client") as mock_client_cls:
        raw = mock_client_cls.return_value
        raw.publish.return_value.rc = 0
        gateway = MqttGatewayClient(host="gw.local", **kwargs)
    return gateway, raw


class TestGatewayClient:
    def test_connects_in_background(self):
        gateway, raw = _make_gateway(port=1884)
        raw.connect_async.assert_called_once_with("gw.local", 1884, 60)
        raw.loop_start.assert_called_once()

    def test_credentials_applied_when_configured(self):
        gateway, raw = _make_gateway(username="ops", password="pw")
        raw.username_pw_set.assert_called_once_with("ops", "pw")

    def test_no_credentials_by_default(self):
        gateway, raw = _make_gateway()
        raw.username_pw_set.assert_not_called()

    def test_publish_uses_base_topic_and_qos1(self):
        gateway, raw = _make_gateway(base_topic="campus/security")
        gateway.publish("alerts/cam0", {"a": 1})

        args, kwargs = raw.publish.call_args
        assert args[0] == "campus/security/alerts/cam0"
        assert json.loads(args[1]) == {"a": 1}
        assert kwargs["qos"] == 1

    def test_notify_publishes_compact_alert(self):
        gateway, raw = _make_gateway()
        alert = {
            "id": 3, "timestamp": "2026-07-04T10:00:00", "confidence": 0.9,
            "camera_id": "cam-north", "alert_type": "violence",
            "clip_path": "data/clips/x.mp4", "status": "new",
        }
        gateway.notify(alert, alert["clip_path"])

        topic, payload = raw.publish.call_args.args
        assert topic == "watchdog/alerts/cam-north"
        decoded = json.loads(payload)
        # Compact LoRa payload: no status field, but the clip reference stays
        assert "status" not in decoded
        assert decoded["clip_path"] == "data/clips/x.mp4"
        assert decoded["alert_type"] == "violence"

    def test_close_stops_network_loop(self):
        gateway, raw = _make_gateway()
        gateway.close()
        raw.loop_stop.assert_called_once()
        raw.disconnect.assert_called_once()


class TestTelemetryLoop:
    def test_publishes_camera_health(self):
        gateway = MagicMock()
        camera = MagicMock()
        camera.is_opened.return_value = True
        camera.is_healthy.return_value = True
        status = {"cam0": {"label": "normal", "counts": {"people": 2}}}
        stop_event = threading.Event()

        loop = TelemetryLoop(
            gateway=gateway,
            cameras={"cam0": camera},
            status_registry=status,
            interval=0.02,
            stop_event=stop_event,
        )
        loop.start()
        time.sleep(0.2)
        stop_event.set()
        loop.join(timeout=3)

        assert gateway.publish_telemetry.called
        camera_id, payload = gateway.publish_telemetry.call_args.args
        assert camera_id == "cam0"
        assert payload["online"] is True
        assert payload["healthy"] is True
        assert payload["counts"] == {"people": 2}
        assert "ts" in payload

    def test_stops_on_event(self):
        stop_event = threading.Event()
        loop = TelemetryLoop(
            gateway=MagicMock(), cameras={}, status_registry={},
            interval=0.02, stop_event=stop_event,
        )
        loop.start()
        stop_event.set()
        loop.join(timeout=3)
        assert not loop.is_alive()

    def test_publish_failure_does_not_kill_loop(self):
        gateway = MagicMock()
        gateway.publish_telemetry.side_effect = RuntimeError("offline")
        camera = MagicMock()
        stop_event = threading.Event()

        loop = TelemetryLoop(
            gateway=gateway, cameras={"cam0": camera}, status_registry={},
            interval=0.02, stop_event=stop_event,
        )
        loop.start()
        time.sleep(0.15)
        assert loop.is_alive()
        stop_event.set()
        loop.join(timeout=3)
        assert gateway.publish_telemetry.call_count >= 2
