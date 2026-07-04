"""Tests for the Camera capture module."""

from __future__ import annotations

import cv2
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.capture.camera import Camera, SourceType, classify_source


def _make_frame(value: int = 0, height: int = 480, width: int = 640) -> np.ndarray:
    """Create a synthetic frame filled with a single value."""
    frame = np.full((height, width, 3), value, dtype=np.uint8)
    return frame


@pytest.fixture()
def mock_capture():
    """Return a mocked cv2.VideoCapture that yields synthetic frames."""
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.read.return_value = (True, _make_frame(42))
    cap.get.side_effect = lambda prop: {
        5: 30.0,   # CAP_PROP_FPS
        3: 640.0,  # CAP_PROP_FRAME_WIDTH
        4: 480.0,  # CAP_PROP_FRAME_HEIGHT
    }.get(prop, 0.0)
    return cap


class TestCameraOpensSource:
    """Test that the Camera correctly opens a video source."""

    @patch("src.capture.camera.cv2.VideoCapture")
    def test_camera_opens_source(self, mock_vc_cls, mock_capture):
        mock_vc_cls.return_value = mock_capture
        cam = Camera(source=0, clip_length=16)

        mock_vc_cls.assert_called_once_with(0)
        assert cam.is_opened() is True
        cam.release()


class TestFrameBuffer:
    """Test that the frame buffer collects frames correctly."""

    @patch("src.capture.camera.cv2.VideoCapture")
    def test_frame_buffer_collects_frames(self, mock_vc_cls, mock_capture):
        mock_vc_cls.return_value = mock_capture
        cam = Camera(source=0, clip_length=4)

        for i in range(3):
            cam.add_frame(_make_frame(i))

        # Buffer should have 3 frames
        assert len(cam._buffer) == 3
        cam.release()


class TestGetClip:
    """Test get_clip behaviour for partial and full buffers."""

    @patch("src.capture.camera.cv2.VideoCapture")
    def test_get_clip_returns_none_when_buffer_not_full(self, mock_vc_cls, mock_capture):
        mock_vc_cls.return_value = mock_capture
        cam = Camera(source=0, clip_length=4)

        cam.add_frame(_make_frame(0))
        cam.add_frame(_make_frame(1))

        assert cam.get_clip() is None
        cam.release()

    @patch("src.capture.camera.cv2.VideoCapture")
    def test_get_clip_returns_clip_when_buffer_full(self, mock_vc_cls, mock_capture):
        mock_vc_cls.return_value = mock_capture
        cam = Camera(source=0, clip_length=4)

        for i in range(4):
            cam.add_frame(_make_frame(i))

        clip = cam.get_clip()
        assert clip is not None
        assert isinstance(clip, np.ndarray)
        assert clip.shape[0] == 4
        # Verify frame ordering: first frame filled with 0, last with 3
        assert clip[0, 0, 0, 0] == 0
        assert clip[3, 0, 0, 0] == 3
        cam.release()


class TestSlidingWindow:
    """Test that the sliding window drops the oldest frame when full."""

    @patch("src.capture.camera.cv2.VideoCapture")
    def test_sliding_window_drops_oldest_frame(self, mock_vc_cls, mock_capture):
        mock_vc_cls.return_value = mock_capture
        cam = Camera(source=0, clip_length=4)

        # Fill buffer with frames 0..3
        for i in range(4):
            cam.add_frame(_make_frame(i))

        # Add one more frame (value=99); oldest (value=0) should be dropped
        cam.add_frame(_make_frame(99))

        clip = cam.get_clip()
        assert clip is not None
        assert clip.shape[0] == 4
        # First frame should now be value 1 (0 was dropped)
        assert clip[0, 0, 0, 0] == 1
        # Last frame should be 99
        assert clip[3, 0, 0, 0] == 99
        cam.release()


class TestCameraRelease:
    """Test that release properly cleans up."""

    @patch("src.capture.camera.cv2.VideoCapture")
    def test_camera_release(self, mock_vc_cls, mock_capture):
        mock_vc_cls.return_value = mock_capture
        cam = Camera(source=0, clip_length=16)

        cam.release()
        mock_capture.release.assert_called_once()
        assert cam.is_opened() is False


class TestContextManager:
    """Test context manager support."""

    @patch("src.capture.camera.cv2.VideoCapture")
    def test_context_manager(self, mock_vc_cls, mock_capture):
        mock_vc_cls.return_value = mock_capture

        with Camera(source=0, clip_length=16) as cam:
            assert cam.is_opened() is True

        mock_capture.release.assert_called_once()


class TestReadFrame:
    """Test read_frame method."""

    @patch("src.capture.camera.cv2.VideoCapture")
    def test_read_frame_returns_frame(self, mock_vc_cls, mock_capture):
        mock_vc_cls.return_value = mock_capture
        cam = Camera(source=0)

        ret, frame = cam.read_frame()
        assert ret is True
        assert isinstance(frame, np.ndarray)
        cam.release()

    @patch("src.capture.camera.cv2.VideoCapture")
    def test_read_frame_returns_false_when_no_frame(self, mock_vc_cls, mock_capture):
        mock_capture.read.return_value = (False, None)
        mock_vc_cls.return_value = mock_capture
        cam = Camera(source=0)

        ret, frame = cam.read_frame()
        assert ret is False
        assert frame is None
        cam.release()


class TestGetLatestFrame:
    """Test get_latest_frame returns the most recent frame."""

    @patch("src.capture.camera.cv2.VideoCapture")
    def test_get_latest_frame(self, mock_vc_cls, mock_capture):
        mock_vc_cls.return_value = mock_capture
        cam = Camera(source=0, clip_length=4)

        cam.add_frame(_make_frame(10))
        cam.add_frame(_make_frame(20))

        latest = cam.get_latest_frame()
        assert latest is not None
        assert latest[0, 0, 0] == 20
        cam.release()

    @patch("src.capture.camera.cv2.VideoCapture")
    def test_get_latest_frame_returns_none_when_empty(self, mock_vc_cls, mock_capture):
        mock_vc_cls.return_value = mock_capture
        cam = Camera(source=0, clip_length=4)

        assert cam.get_latest_frame() is None
        cam.release()


class TestCameraProperties:
    """Test fps, frame_width, frame_height properties."""

    @patch("src.capture.camera.cv2.VideoCapture")
    def test_camera_properties(self, mock_vc_cls, mock_capture):
        mock_vc_cls.return_value = mock_capture
        cam = Camera(source=0)

        assert cam.fps == 30.0
        assert cam.frame_width == 640
        assert cam.frame_height == 480
        cam.release()


class TestSourceClassification:
    """Sources should be classified as webcam, file, or stream."""

    def test_int_is_webcam(self):
        assert classify_source(0) is SourceType.WEBCAM

    def test_rtsp_is_stream(self):
        assert classify_source("rtsp://10.0.0.11/stream") is SourceType.STREAM

    def test_http_is_stream(self):
        assert classify_source("http://10.0.0.11/mjpeg") is SourceType.STREAM

    def test_path_is_file(self):
        assert classify_source("data/video.mp4") is SourceType.FILE

    @patch("src.capture.camera.cv2.VideoCapture")
    def test_camera_exposes_source_type(self, mock_vc_cls, mock_capture):
        mock_vc_cls.return_value = mock_capture
        cam = Camera(source="rtsp://example/stream")
        assert cam.source_type is SourceType.STREAM
        cam.release()


class TestCameraIdentity:
    """Cameras carry an id and display name."""

    @patch("src.capture.camera.cv2.VideoCapture")
    def test_camera_id_and_name(self, mock_vc_cls, mock_capture):
        mock_vc_cls.return_value = mock_capture
        cam = Camera(source=0, camera_id="cam-north", name="North Gate")
        assert cam.id == "cam-north"
        assert cam.name == "North Gate"
        cam.release()

    @patch("src.capture.camera.cv2.VideoCapture")
    def test_name_defaults_to_id(self, mock_vc_cls, mock_capture):
        mock_vc_cls.return_value = mock_capture
        cam = Camera(source=0, camera_id="cam-north")
        assert cam.name == "cam-north"
        cam.release()


class TestQualityConfiguration:
    """Requested capture quality should be applied to the device."""

    @patch("src.capture.camera.cv2.VideoCapture")
    def test_quality_settings_applied(self, mock_vc_cls, mock_capture):
        mock_vc_cls.return_value = mock_capture
        cam = Camera(source=0, width=1280, height=720, target_fps=15)

        mock_capture.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        mock_capture.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        mock_capture.set.assert_any_call(cv2.CAP_PROP_FPS, 15)
        cam.release()

    @patch("src.capture.camera.cv2.VideoCapture")
    def test_stream_gets_minimal_buffer(self, mock_vc_cls, mock_capture):
        mock_vc_cls.return_value = mock_capture
        cam = Camera(source="rtsp://example/stream")
        mock_capture.set.assert_any_call(cv2.CAP_PROP_BUFFERSIZE, 1)
        cam.release()

    @patch("src.capture.camera.cv2.VideoCapture")
    def test_no_quality_calls_by_default(self, mock_vc_cls, mock_capture):
        mock_vc_cls.return_value = mock_capture
        cam = Camera(source=0)
        mock_capture.set.assert_not_called()
        cam.release()


class TestRead:
    """read() should recover from source failures per source type."""

    @patch("src.capture.camera.time.sleep")
    @patch("src.capture.camera.cv2.VideoCapture")
    def test_read_returns_frame(self, mock_vc_cls, mock_sleep, mock_capture):
        mock_vc_cls.return_value = mock_capture
        cam = Camera(source=0)
        frame = cam.read()
        assert frame is not None
        cam.release()

    @patch("src.capture.camera.time.sleep")
    @patch("src.capture.camera.cv2.VideoCapture")
    def test_file_loops_on_end(self, mock_vc_cls, mock_sleep, mock_capture):
        mock_capture.read.return_value = (False, None)
        mock_vc_cls.return_value = mock_capture
        cam = Camera(source="video.mp4")

        assert cam.read() is None
        mock_capture.set.assert_any_call(cv2.CAP_PROP_POS_FRAMES, 0)
        cam.release()

    @patch("src.capture.camera.time.sleep")
    @patch("src.capture.camera.cv2.VideoCapture")
    def test_file_read_paces_to_native_fps(self, mock_vc_cls, mock_sleep, mock_capture):
        mock_vc_cls.return_value = mock_capture
        cam = Camera(source="video.mp4")

        assert cam.read() is not None
        mock_sleep.assert_called_once_with(pytest.approx(1.0 / 30.0))
        cam.release()

    @patch("src.capture.camera.time.sleep")
    @patch("src.capture.camera.cv2.VideoCapture")
    def test_stream_reconnects_after_repeated_failures(
        self, mock_vc_cls, mock_sleep, mock_capture
    ):
        mock_capture.read.return_value = (False, None)
        mock_vc_cls.return_value = mock_capture
        cam = Camera(source="rtsp://example/stream")

        for _ in range(Camera.FAILURES_BEFORE_RECONNECT):
            assert cam.read() is None

        # Initial open + one reconnect
        assert mock_vc_cls.call_count == 2
        mock_capture.release.assert_called()
        cam.release()

    @patch("src.capture.camera.time.sleep")
    @patch("src.capture.camera.cv2.VideoCapture")
    def test_success_resets_failure_counter(self, mock_vc_cls, mock_sleep, mock_capture):
        frame = _make_frame(1)
        results = [(False, None)] * 3 + [(True, frame)] + [(False, None)] * 3
        mock_capture.read.side_effect = results
        mock_vc_cls.return_value = mock_capture
        cam = Camera(source="rtsp://example/stream")

        for _ in range(len(results)):
            cam.read()

        # Never reached the reconnect threshold, so only the initial open
        assert mock_vc_cls.call_count == 1
        cam.release()

    @patch("src.capture.camera.cv2.VideoCapture")
    def test_read_after_release_returns_none(self, mock_vc_cls, mock_capture):
        mock_vc_cls.return_value = mock_capture
        cam = Camera(source=0)
        cam.release()
        assert cam.read() is None


class TestFrameSequence:
    """add_frame should bump the sequence counter for new-frame detection."""

    @patch("src.capture.camera.cv2.VideoCapture")
    def test_sequence_increments_per_frame(self, mock_vc_cls, mock_capture):
        mock_vc_cls.return_value = mock_capture
        cam = Camera(source=0, clip_length=4)

        _, seq0 = cam.get_latest_frame_with_seq()
        assert seq0 == 0

        cam.add_frame(_make_frame(1))
        frame, seq1 = cam.get_latest_frame_with_seq()
        assert seq1 == 1
        assert frame[0, 0, 0] == 1

        cam.add_frame(_make_frame(2))
        _, seq2 = cam.get_latest_frame_with_seq()
        assert seq2 == 2
        cam.release()


class TestHealth:
    """is_healthy should reflect recent frame activity."""

    @patch("src.capture.camera.cv2.VideoCapture")
    def test_unhealthy_without_frames(self, mock_vc_cls, mock_capture):
        mock_vc_cls.return_value = mock_capture
        cam = Camera(source=0)
        assert cam.last_frame_age is None
        assert cam.is_healthy() is False
        cam.release()

    @patch("src.capture.camera.cv2.VideoCapture")
    def test_healthy_after_recent_frame(self, mock_vc_cls, mock_capture):
        mock_vc_cls.return_value = mock_capture
        cam = Camera(source=0)
        cam.add_frame(_make_frame(1))
        assert cam.is_healthy() is True
        assert cam.last_frame_age is not None
        cam.release()

    @patch("src.capture.camera.cv2.VideoCapture")
    def test_unhealthy_when_frames_stale(self, mock_vc_cls, mock_capture):
        mock_vc_cls.return_value = mock_capture
        cam = Camera(source=0)
        cam.add_frame(_make_frame(1))
        assert cam.is_healthy(max_age_seconds=0.0) is False
        cam.release()
