import sys
from pathlib import Path

import pytest

# Make `src` importable regardless of the pytest invocation directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture()
def vp_data_dir(tmp_path, monkeypatch):
    """Point the pilot data dir at a temp location for the test."""
    data_dir = tmp_path / "vp_data"
    monkeypatch.setenv("VP_DATA_DIR", str(data_dir))
    return data_dir


@pytest.fixture()
def conn(vp_data_dir):
    """A fresh, initialized pilot DB under the temp data dir."""
    from src.visual_pilot import db

    connection = db.init_db()
    yield connection
    connection.close()
