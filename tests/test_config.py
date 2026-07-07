"""Tests for src.config module."""

from __future__ import annotations

import json

import pytest

from src.config import CameraConfig, Settings, get_settings


class TestSettingsDefaults:
    """Settings should have sensible defaults when no env vars are set."""

    def test_default_single_camera(self, monkeypatch):
        monkeypatch.delenv("CAMERAS", raising=False)
        monkeypatch.delenv("CAMERA_SOURCE", raising=False)
        settings = Settings()
        assert len(settings.cameras) == 1
        assert settings.cameras[0].id == "cam0"
        assert settings.cameras[0].source == 0

    def test_default_confidence_threshold(self):
        settings = Settings()
        assert settings.confidence_threshold == 0.85

    def test_default_consecutive_hits(self):
        settings = Settings()
        assert settings.consecutive_hits == 3

    def test_default_cooldown_seconds(self):
        settings = Settings()
        assert settings.cooldown_seconds == 5

    def test_default_pre_event_seconds(self):
        settings = Settings()
        assert settings.pre_event_seconds == 3

    def test_default_post_event_seconds(self):
        settings = Settings()
        assert settings.post_event_seconds == 2

    def test_default_dashboard_port(self):
        settings = Settings()
        assert settings.dashboard_port == 8000

    def test_default_db_path(self):
        settings = Settings()
        assert settings.db_path == "data/watchdog.db"

    def test_default_clip_dir(self):
        settings = Settings()
        assert settings.clip_dir == "data/clips"

    def test_default_log_dir(self):
        settings = Settings()
        assert settings.log_dir == "logs"

    def test_default_log_level(self):
        settings = Settings()
        assert settings.log_level == "INFO"


class TestSettingsFromEnv:
    """Settings should respect environment variable overrides."""

    def test_confidence_threshold_from_env(self, monkeypatch):
        monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.5")
        settings = Settings()
        assert settings.confidence_threshold == 0.5

    def test_consecutive_hits_from_env(self, monkeypatch):
        monkeypatch.setenv("CONSECUTIVE_HITS", "5")
        settings = Settings()
        assert settings.consecutive_hits == 5

    def test_cooldown_seconds_from_env(self, monkeypatch):
        monkeypatch.setenv("COOLDOWN_SECONDS", "10")
        settings = Settings()
        assert settings.cooldown_seconds == 10

    def test_dashboard_port_from_env(self, monkeypatch):
        monkeypatch.setenv("DASHBOARD_PORT", "9000")
        settings = Settings()
        assert settings.dashboard_port == 9000

    def test_camera_source_int_from_env(self, monkeypatch):
        monkeypatch.delenv("CAMERAS", raising=False)
        monkeypatch.setenv("CAMERA_SOURCE", "2")
        settings = Settings()
        assert settings.cameras[0].source == 2

    def test_camera_source_path_from_env(self, monkeypatch):
        monkeypatch.delenv("CAMERAS", raising=False)
        monkeypatch.setenv("CAMERA_SOURCE", "/dev/video1")
        settings = Settings()
        assert settings.cameras[0].source == "/dev/video1"

    def test_log_level_from_env(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        settings = Settings()
        assert settings.log_level == "DEBUG"

    def test_db_path_from_env(self, monkeypatch):
        monkeypatch.setenv("DB_PATH", "/tmp/test.db")
        settings = Settings()
        assert settings.db_path == "/tmp/test.db"

    def test_clip_dir_from_env(self, monkeypatch):
        monkeypatch.setenv("CLIP_DIR", "/tmp/clips")
        settings = Settings()
        assert settings.clip_dir == "/tmp/clips"

    def test_log_dir_from_env(self, monkeypatch):
        monkeypatch.setenv("LOG_DIR", "/tmp/logs")
        settings = Settings()
        assert settings.log_dir == "/tmp/logs"

    def test_pre_event_seconds_from_env(self, monkeypatch):
        monkeypatch.setenv("PRE_EVENT_SECONDS", "5")
        settings = Settings()
        assert settings.pre_event_seconds == 5.0


class TestCamerasFromEnv:
    """The CAMERAS env var should configure multiple cameras from JSON."""

    def test_two_cameras_from_json(self, monkeypatch):
        monkeypatch.setenv(
            "CAMERAS",
            '[{"id": "cam-north", "name": "North Gate", "source": "rtsp://10.0.0.11/stream"},'
            ' {"id": "cam-south", "name": "South Gate", "source": 1}]',
        )
        settings = Settings()
        assert len(settings.cameras) == 2
        north, south = settings.cameras
        assert north.id == "cam-north"
        assert north.name == "North Gate"
        assert north.source == "rtsp://10.0.0.11/stream"
        assert south.id == "cam-south"
        assert south.source == 1

    def test_camera_quality_settings(self, monkeypatch):
        monkeypatch.setenv(
            "CAMERAS",
            '[{"id": "c1", "source": 0, "width": 1280, "height": 720, "fps": 15}]',
        )
        settings = Settings()
        cam = settings.cameras[0]
        assert cam.width == 1280
        assert cam.height == 720
        assert cam.fps == 15.0

    def test_camera_ids_default_to_index(self, monkeypatch):
        monkeypatch.setenv("CAMERAS", '[{"source": 0}, {"source": 1}]')
        settings = Settings()
        assert [c.id for c in settings.cameras] == ["cam0", "cam1"]

    def test_numeric_string_source_becomes_int(self, monkeypatch):
        monkeypatch.setenv("CAMERAS", '[{"id": "c1", "source": "0"}]')
        settings = Settings()
        assert settings.cameras[0].source == 0

    def test_invalid_json_raises(self, monkeypatch):
        monkeypatch.setenv("CAMERAS", "not json")
        with pytest.raises(ValueError):
            Settings()

    def test_duplicate_ids_raise(self, monkeypatch):
        monkeypatch.setenv("CAMERAS", '[{"id": "c1", "source": 0}, {"id": "c1", "source": 1}]')
        with pytest.raises(ValueError):
            Settings()

    def test_empty_list_raises(self, monkeypatch):
        monkeypatch.setenv("CAMERAS", "[]")
        with pytest.raises(ValueError):
            Settings()

    def test_explicit_cameras_override(self):
        cameras = (CameraConfig(id="a", source=0), CameraConfig(id="b", source=1))
        settings = Settings(cameras=cameras)
        assert settings.cameras == cameras

    def test_unsafe_camera_id_rejected(self, monkeypatch):
        # ids flow into filenames, URLs, and MQTT topics
        monkeypatch.setenv("CAMERAS", '[{"id": "cam/../evil", "source": 0}]')
        with pytest.raises(ValueError):
            Settings()

    def test_missing_source_rejected(self, monkeypatch):
        monkeypatch.setenv("CAMERAS", '[{"id": "c1"}]')
        with pytest.raises(ValueError):
            Settings()

    def test_null_source_rejected(self, monkeypatch):
        monkeypatch.setenv("CAMERAS", '[{"id": "c1", "source": null}]')
        with pytest.raises(ValueError):
            Settings()

    def test_integral_float_source_coerced(self, monkeypatch):
        monkeypatch.setenv("CAMERAS", '[{"id": "c1", "source": 1.0}]')
        settings = Settings()
        assert settings.cameras[0].source == 1
        assert isinstance(settings.cameras[0].source, int)

    def test_fractional_source_rejected(self, monkeypatch):
        monkeypatch.setenv("CAMERAS", '[{"id": "c1", "source": 1.5}]')
        with pytest.raises(ValueError):
            Settings()


class TestFolderSourceExpansion:
    """A camera source pointing at a folder expands into one camera per file."""

    @pytest.fixture()
    def video_folder(self, tmp_path):
        folder = tmp_path / "footage"
        folder.mkdir()
        (folder / "b_second.mp4").write_bytes(b"fake")
        (folder / "a_first.avi").write_bytes(b"fake")
        (folder / "notes.txt").write_bytes(b"not a video")  # must be ignored
        return folder

    def test_cameras_entry_folder_expands_per_file(self, monkeypatch, video_folder):
        monkeypatch.setenv(
            "CAMERAS", json.dumps([{"id": "archive", "source": str(video_folder)}])
        )
        settings = Settings()
        assert [c.id for c in settings.cameras] == ["archive-0", "archive-1"]

    def test_files_sorted_by_name(self, monkeypatch, video_folder):
        monkeypatch.setenv(
            "CAMERAS", json.dumps([{"id": "archive", "source": str(video_folder)}])
        )
        settings = Settings()
        # a_first.avi sorts before b_second.mp4
        assert settings.cameras[0].source.endswith("a_first.avi")
        assert settings.cameras[1].source.endswith("b_second.mp4")

    def test_non_video_files_ignored(self, monkeypatch, video_folder):
        monkeypatch.setenv(
            "CAMERAS", json.dumps([{"id": "archive", "source": str(video_folder)}])
        )
        settings = Settings()
        assert len(settings.cameras) == 2  # notes.txt excluded

    def test_name_includes_filename(self, monkeypatch, video_folder):
        monkeypatch.setenv(
            "CAMERAS",
            json.dumps([{"id": "archive", "name": "Archived", "source": str(video_folder)}]),
        )
        settings = Settings()
        assert settings.cameras[0].name == "Archived (a_first.avi)"

    def test_folder_expansion_coexists_with_live_camera(self, monkeypatch, video_folder):
        monkeypatch.setenv(
            "CAMERAS",
            json.dumps([
                {"id": "live", "source": "rtsp://10.0.0.1/stream"},
                {"id": "archive", "source": str(video_folder)},
            ]),
        )
        settings = Settings()
        ids = [c.id for c in settings.cameras]
        assert ids == ["live", "archive-0", "archive-1"]
        assert settings.cameras[0].source == "rtsp://10.0.0.1/stream"

    def test_empty_folder_raises(self, monkeypatch, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.setenv(
            "CAMERAS", json.dumps([{"id": "archive", "source": str(empty)}])
        )
        with pytest.raises(ValueError):
            Settings()

    def test_camera_source_fallback_supports_folder(self, monkeypatch, video_folder):
        monkeypatch.delenv("CAMERAS", raising=False)
        monkeypatch.setenv("CAMERA_SOURCE", str(video_folder))
        settings = Settings()
        assert [c.id for c in settings.cameras] == ["cam0-0", "cam0-1"]

    def test_single_file_source_unaffected(self, monkeypatch, tmp_path):
        """A plain file path (not a directory) must not be treated as a folder."""
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"fake")
        monkeypatch.delenv("CAMERAS", raising=False)
        monkeypatch.setenv("CAMERA_SOURCE", str(video))
        settings = Settings()
        assert len(settings.cameras) == 1
        assert settings.cameras[0].id == "cam0"
        assert settings.cameras[0].source == str(video)

    def test_webcam_index_unaffected(self, monkeypatch):
        monkeypatch.delenv("CAMERAS", raising=False)
        monkeypatch.setenv("CAMERA_SOURCE", "0")
        settings = Settings()
        assert len(settings.cameras) == 1
        assert settings.cameras[0].source == 0


class TestSourceModeSettings:
    """SOURCE_MODE / RECORDINGS_DIR feed the runtime source toggle."""

    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("SOURCE_MODE", raising=False)
        monkeypatch.delenv("RECORDINGS_DIR", raising=False)
        settings = Settings()
        assert settings.source_mode == "live"
        assert settings.recordings_dir == ""

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("SOURCE_MODE", "recordings")
        monkeypatch.setenv("RECORDINGS_DIR", "C:/footage")
        settings = Settings()
        assert settings.source_mode == "recordings"
        assert settings.recordings_dir == "C:/footage"


class TestRecordingsCameraConfigs:
    """recordings_camera_configs builds the recordings-mode camera list."""

    def test_one_camera_per_file(self, tmp_path):
        from src.config import recordings_camera_configs

        folder = tmp_path / "rec"
        folder.mkdir()
        (folder / "b.mp4").write_bytes(b"fake")
        (folder / "a.avi").write_bytes(b"fake")

        configs = recordings_camera_configs(str(folder))
        assert [c.id for c in configs] == ["rec-0", "rec-1"]
        assert configs[0].source.endswith("a.avi")
        assert configs[0].name == "Recording (a.avi)"

    def test_missing_folder_raises(self, tmp_path):
        from src.config import recordings_camera_configs

        with pytest.raises(ValueError):
            recordings_camera_configs(str(tmp_path / "nope"))

    def test_empty_string_raises(self):
        from src.config import recordings_camera_configs

        with pytest.raises(ValueError):
            recordings_camera_configs("")

    def test_file_path_raises(self, tmp_path):
        from src.config import recordings_camera_configs

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"fake")
        with pytest.raises(ValueError):
            recordings_camera_configs(str(video))


class TestGetSettings:
    """get_settings() should return a valid Settings instance."""

    def test_returns_settings_instance(self):
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_settings_is_frozen(self):
        settings = get_settings()
        with pytest.raises(AttributeError):
            settings.log_level = "DEBUG"  # type: ignore[misc]
