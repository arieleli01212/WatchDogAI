"""Tests for the runtime PipelineManager (source-mode switching)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.config import CameraConfig, Settings
from src.runtime import PipelineManager


LIVE_CAMERAS = (
    CameraConfig(id="live-a", source=0),
    CameraConfig(id="live-b", source="rtsp://x/stream"),
)


@pytest.fixture()
def recordings_folder(tmp_path):
    folder = tmp_path / "recordings"
    folder.mkdir()
    (folder / "one.mp4").write_bytes(b"fake")
    (folder / "two.mp4").write_bytes(b"fake")
    return folder


def _make_manager(recordings_dir: str = ""):
    settings = Settings(cameras=LIVE_CAMERAS, recordings_dir=recordings_dir)
    cameras: dict = {}
    status: dict = {}
    created: list = []

    def fake_pipeline(**kwargs):
        pipeline = MagicMock()
        pipeline.config = kwargs["config"]
        created.append(pipeline)
        return pipeline

    patcher = patch("src.runtime.CameraPipeline", side_effect=fake_pipeline)
    manager = PipelineManager(
        settings=settings,
        detector=MagicMock(),
        alert_manager=MagicMock(),
        cameras_registry=cameras,
        status_registry=status,
    )
    return manager, cameras, status, created, patcher


class TestAvailableModes:
    def test_live_only_without_recordings_dir(self):
        manager, *_ = _make_manager()
        assert manager.available_modes() == ["live"]

    def test_recordings_offered_when_configured(self, recordings_folder):
        manager, *_ = _make_manager(str(recordings_folder))
        assert manager.available_modes() == ["live", "recordings"]


class TestStartAndSwitch:
    def test_start_live_builds_configured_cameras(self):
        manager, cameras, status, created, patcher = _make_manager()
        with patcher:
            manager.start("live")

        assert manager.mode == "live"
        assert [p.config.id for p in created] == ["live-a", "live-b"]
        assert set(cameras) == {"live-a", "live-b"}
        for pipeline in created:
            pipeline.start.assert_called_once()

    def test_switch_to_recordings_swaps_generation(self, recordings_folder):
        manager, cameras, status, created, patcher = _make_manager(str(recordings_folder))
        with patcher:
            manager.start("live")
            status["live-a"] = {"label": "normal"}  # simulate analysis output
            live_pipelines = list(created)

            changed = manager.switch("recordings")

        assert changed is True
        assert manager.mode == "recordings"
        # Old generation stopped and released
        for pipeline in live_pipelines:
            pipeline.join.assert_called_once()
            pipeline.release.assert_called_once()
        # Registries mutated in place: old entries gone, one camera per file
        assert set(cameras) == {"rec-0", "rec-1"}
        assert "live-a" not in status

    def test_switch_to_active_mode_is_a_noop(self):
        manager, cameras, status, created, patcher = _make_manager()
        with patcher:
            manager.start("live")
            count_after_start = len(created)
            changed = manager.switch("live")

        assert changed is False
        assert len(created) == count_after_start  # nothing rebuilt

    def test_unknown_mode_rejected(self):
        manager, *_ , patcher = _make_manager()
        with patcher, pytest.raises(ValueError):
            manager.switch("bogus")

    def test_recordings_mode_without_dir_rejected(self):
        manager, *_, patcher = _make_manager()
        with patcher, pytest.raises(ValueError):
            manager.start("recordings")

    def test_failed_switch_keeps_current_generation(self, tmp_path):
        """An empty recordings folder must not tear down the live cameras."""
        empty = tmp_path / "empty"
        empty.mkdir()
        manager, cameras, status, created, patcher = _make_manager(str(empty))
        with patcher:
            manager.start("live")
            with pytest.raises(ValueError):
                manager.switch("recordings")

        # Live pipelines survived the failed swap
        assert manager.mode == "live"
        assert set(cameras) == {"live-a", "live-b"}
        for pipeline in created:
            pipeline.release.assert_not_called()

    def test_recordings_folder_rescanned_each_switch(self, recordings_folder):
        manager, cameras, status, created, patcher = _make_manager(str(recordings_folder))
        with patcher:
            manager.start("recordings")
            assert set(cameras) == {"rec-0", "rec-1"}

            (recordings_folder / "three.mp4").write_bytes(b"fake")
            manager.switch("live")
            manager.switch("recordings")

        assert set(cameras) == {"rec-0", "rec-1", "rec-2"}


class TestStop:
    def test_stop_tears_everything_down(self):
        manager, cameras, status, created, patcher = _make_manager()
        with patcher:
            manager.start("live")
            manager.stop()

        assert cameras == {}
        assert status == {}
        for pipeline in created:
            pipeline.join.assert_called_once()
            pipeline.release.assert_called_once()

    def test_stop_when_nothing_running_is_safe(self):
        manager, *_ = _make_manager()
        manager.stop()  # must not raise
