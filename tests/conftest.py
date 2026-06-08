import sys
from pathlib import Path

import pytest

# Add the project root to Python path so tests can import modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from synthetic_drs_fixtures import (  # noqa: E402
    SyntheticBallDetector,
    ensure_synthetic_drs_videos,
    save_synthetic_calibration,
)


@pytest.fixture(scope="session", autouse=True)
def synthetic_drs_videos() -> None:
    ensure_synthetic_drs_videos()


@pytest.fixture()
def synthetic_e2e_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("core.pitch_calibration.CALIBRATION_DIR", tmp_path)
    monkeypatch.setattr("core.pitch_calibration.READINESS_PATH", tmp_path / "readiness.json")
    readiness_path = save_synthetic_calibration(tmp_path)

    from core.readiness import ReadinessGate

    def readiness_factory(*args, **kwargs):
        kwargs["calibration_path"] = readiness_path
        return ReadinessGate(*args, **kwargs)

    monkeypatch.setattr("core.testing_pipeline.BallDetector", SyntheticBallDetector)
    monkeypatch.setattr("core.testing_pipeline.ReadinessGate", readiness_factory)
    return readiness_path
