"""Tests for production DRS decision service."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.drs_decision import DRSDecisionService
from core.pitch_calibration import ManualPitchCalibrator
from core.readiness import ReadinessGate


SAMPLE_MARKERS = {
    "off_stump": {"x": 100.0, "y": 300.0},
    "middle_stump": {"x": 180.0, "y": 298.0},
    "leg_stump": {"x": 260.0, "y": 300.0},
    "bowling_crease": {"x": 180.0, "y": 360.0},
    "popping_crease": {"x": 180.0, "y": 420.0},
}


def _sample_tracks() -> list[dict]:
    tracks = []
    for idx in range(20):
        tracks.append(
            {
                "frame_id": idx,
                "timestamp_ms": idx * 33.0,
                "x": 100 + idx * 8,
                "y": 300 - idx * 4 + (6 if idx > 10 else 0),
                "confidence": 0.8,
            }
        )
    return tracks


@pytest.fixture()
def calibrated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.pitch_calibration.CALIBRATION_DIR", tmp_path)
    monkeypatch.setattr("core.pitch_calibration.READINESS_PATH", tmp_path / "readiness.json")
    ManualPitchCalibrator().save_profile(1, SAMPLE_MARKERS, (1280, 720))
    readiness = {
        "reprojection_error_px": 0.8,
        "homography_error_cm": 1.2,
        "pitch_coordinate_error_cm": 1.2,
        "per_camera": {"1": {"homography_error_cm": 1.2}},
    }
    (tmp_path / "readiness.json").write_text(json.dumps(readiness), encoding="utf-8")


def test_decision_uses_lbw_engine_when_calibrated(tmp_path: Path, calibrated_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    readiness_path = tmp_path / "readiness.json"
    monkeypatch.setattr("core.pitch_calibration.CALIBRATION_DIR", tmp_path)
    service = DRSDecisionService()
    tracks = _sample_tracks()
    camera_results = [
        {
            "camera_id": 0,
            "confidence": 0.82,
            "ball_speed_kmh": 120.0,
            "tracking_points": tracks,
            "impact_point_px": [220, 250],
            "tracking_quality": {"score": 0.8, "reliability": "high", "max_missing_gap": 2},
        }
    ]
    calibration = ReadinessGate(calibration_path=readiness_path).calibration()
    sync = ReadinessGate().sync({"sync_error_ms": 2.0}, 30.0)
    model = {"map50": 0.9, "ball_recall": 0.92}
    decision = service.build_decision(tracks, camera_results, False, calibration, sync, ReadinessGate(), model)
    assert "lbw_engine" in decision
    assert decision["raw_lbw_recommendation"] in {"OUT", "NOT OUT", "UMPIRE'S CALL", "REVIEW INCONCLUSIVE"}
    assert "reasoning" in decision


def test_decision_inconclusive_without_calibration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.pitch_calibration.CALIBRATION_DIR", tmp_path)
    service = DRSDecisionService()
    camera_results = [{"camera_id": 0, "confidence": 0.5, "ball_speed_kmh": 100.0, "tracking_points": _sample_tracks(), "tracking_quality": {}}]
    calibration = ReadinessGate(calibration_path=tmp_path / "readiness.json").calibration()
    sync = ReadinessGate().sync({"sync_error_ms": 2.0}, 30.0)
    decision = service.build_decision(_sample_tracks(), camera_results, False, calibration, sync, ReadinessGate(), {})
    assert decision["raw_lbw_recommendation"] == "REVIEW INCONCLUSIVE"


def test_explicit_bounce_pixel_is_mapped_through_calibration(calibrated_env: None) -> None:
    service = DRSDecisionService()
    mapped = service.calibrated_pixel_point([180, 360], 1)
    assert mapped is not None
    assert abs(mapped["lateral_mm"]) < 1.0
    assert abs(mapped["along_mm"]) < 1.0


def test_stump_projection_targets_calibrated_wicket_plane() -> None:
    service = DRSDecisionService()
    pitch_path = []
    for idx in range(8):
        pitch_path.append(
            {
                "lateral_mm": 0.0,
                "along_mm": -900.0 + (idx * 100.0),
                "timestamp_ms": idx * 20.0,
                "confidence": 0.9,
                "frame_id": idx,
            }
        )

    probability, extension = service.stump_hit_probability(pitch_path, {"lateral_mm": 0.0, "along_mm": -450.0})
    assert probability >= 0.72
    assert extension
