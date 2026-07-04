"""MQTT client for the LoRa smart-campus gateway.

LoRa is a low-bandwidth telemetry link (kbit/s class), so video never
travels through it: the cameras stream to the AI server over IP, and
this client publishes only compact JSON messages — alert events and
periodic camera health telemetry — to the gateway's MQTT broker.

Topics (under ``base_topic``, default ``watchdog``):

- ``watchdog/alerts/<camera_id>``    — one message per alert
- ``watchdog/telemetry/<camera_id>`` — periodic camera health + counts
"""

from __future__ import annotations

import json
import logging
import threading
import time

logger = logging.getLogger(__name__)

# Only these alert fields are forwarded: the payload must stay small
# enough for the LoRa downlink. The clip itself stays on the server;
# the message carries its path for retrieval over IP.
ALERT_FIELDS = ("id", "timestamp", "confidence", "camera_id", "alert_type", "clip_path")


class MqttGatewayClient:
    """Publishes alerts and telemetry to the LoRa gateway MQTT broker."""

    def __init__(
        self,
        host: str,
        port: int = 1883,
        username: str = "",
        password: str = "",
        base_topic: str = "watchdog",
        client_id: str = "watchdogai",
        keepalive: int = 60,
    ) -> None:
        import paho.mqtt.client as mqtt

        self._base_topic = base_topic.rstrip("/")
        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2, client_id=client_id
        )
        if username:
            self._client.username_pw_set(username, password)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect

        # connect_async + loop_start: a background network thread keeps
        # retrying and reconnecting, so a gateway outage never blocks us
        self._client.connect_async(host, port, keepalive)
        self._client.loop_start()
        logger.info("MQTT gateway client connecting to %s:%d", host, port)

    # ------------------------------------------------------------------
    # Notifier interface (called by AlertManager)
    # ------------------------------------------------------------------

    def notify(self, alert: dict, clip_path: str) -> None:
        """Publish a compact alert event for the control center."""
        payload = {k: alert[k] for k in ALERT_FIELDS if k in alert}
        self.publish(f"alerts/{alert.get('camera_id', 'unknown')}", payload)

    def publish_telemetry(self, camera_id: str, telemetry: dict) -> None:
        """Publish periodic camera health/count telemetry."""
        self.publish(f"telemetry/{camera_id}", telemetry)

    def publish(self, subtopic: str, payload: dict) -> None:
        topic = f"{self._base_topic}/{subtopic}"
        result = self._client.publish(topic, json.dumps(payload), qos=1)
        if result.rc != 0:
            # rc != 0 with qos=1 usually means offline; paho queues and
            # retries after reconnect, so this is informational
            logger.debug("MQTT publish to %s returned rc=%s", topic, result.rc)

    def close(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_connect(self, client, userdata, flags, reason_code, properties) -> None:
        logger.info("MQTT gateway connected (reason=%s)", reason_code)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties) -> None:
        logger.warning("MQTT gateway disconnected (reason=%s)", reason_code)


class TelemetryLoop(threading.Thread):
    """Periodically publishes per-camera health telemetry to the gateway."""

    def __init__(
        self,
        gateway: MqttGatewayClient,
        cameras: dict,
        status_registry: dict,
        interval: float,
        stop_event: threading.Event,
        health_max_age: float = 5.0,
    ) -> None:
        super().__init__(name="mqtt-telemetry", daemon=True)
        self._gateway = gateway
        self._cameras = cameras
        self._status = status_registry
        self._interval = interval
        self._stop = stop_event
        self._health_max_age = health_max_age

    def run(self) -> None:
        logger.info("Telemetry loop started (interval=%.0fs)", self._interval)
        while not self._stop.wait(self._interval):
            for camera_id, camera in self._cameras.items():
                status = self._status.get(camera_id, {})
                try:
                    self._gateway.publish_telemetry(
                        camera_id,
                        {
                            "online": camera.is_opened(),
                            "healthy": camera.is_healthy(self._health_max_age),
                            "label": status.get("label", "normal"),
                            "counts": status.get("counts", {}),
                            "ts": time.time(),
                        },
                    )
                except Exception:
                    logger.exception("Telemetry publish failed for %s", camera_id)
        logger.info("Telemetry loop stopped")
