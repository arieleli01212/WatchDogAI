"""Tests for the Camera capture module."""

from __future__ import annotations

from collections import deque
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from src.capture.camera import Camera


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
