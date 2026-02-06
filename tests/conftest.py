import pytest
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    path = Path(tempfile.mkdtemp())
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def sample_config():
    """Return a minimal test configuration dict."""
    return {
        "run_name": "test",
        "data": {
            "num_frames": 16,
            "resize_short_side": 128,
            "crop_size": 112,
        },
        "model": {
            "name": "r3d_18",
            "pretrained": False,
            "dropout": 0.0,
        },
        "detection": {
            "threshold": 0.7,
            "window_size": 10,
            "window_threshold": 6,
            "pre_buffer_seconds": 5,
            "post_buffer_seconds": 5,
        },
        "storage": {
            "clips_dir": "clips",
            "retention_days": 7,
        },
        "logging": {
            "level": "INFO",
            "format": "json",
        },
    }
