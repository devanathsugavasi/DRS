"""Tournament-grade readiness metrics and decision gates."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class GateThresholds:
    model_map50: float = 0.88
    ball_recall: float = 0.90
    tracking_score: float = 0.58
    max_missing_gap: int = 5
    calibration_reprojection_px: float = 1.5
    homography_error_cm: float = 5.0
    sync_error_ms: float = 8.0
    replay_fps: float = 24.0
    decision_confidence: float = 0.70


@dataclass(slots=True)
class CalibrationReadiness:
    reprojection_error_px: float | None
    homography_error_cm: float | None
    pitch_coordinate_error_cm: float | None
    per_camera: dict[str, Any]
    usable: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SyncReadiness:
    sync_error_ms: float | None
    dropped_frames: int
    replay_fps: float
    usable: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DecisionGateResult:
    allowed: bool
    display_decision: str
    confidence: float
    failed_gates: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReadinessGate:
    """Blocks OUT/NOT OUT unless model, calibration, tracking, sync, replay, and confidence pass."""

    def __init__(
        self,
        thresholds: GateThresholds | None = None,
        calibration_path: Path = Path("data/calibration/readiness.json"),
    ) -> None:
        self.thresholds = thresholds or GateThresholds()
        self.calibration_path = calibration_path

    def calibration(self) -> CalibrationReadiness:
        if not self.calibration_path.exists():
            return CalibrationReadiness(
                reprojection_error_px=None,
                homography_error_cm=None,
                pitch_coordinate_error_cm=None,
                per_camera={},
                usable=False,
                reason="No calibration readiness file found. Run camera calibration on the match cameras.",
            )
        try:
            data = json.loads(self.calibration_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return CalibrationReadiness(None, None, None, {}, False, "Calibration readiness file is invalid JSON.")

        reproj = _float_or_none(data.get("reprojection_error_px"))
        homography = _float_or_none(data.get("homography_error_cm"))
        pitch = _float_or_none(data.get("pitch_coordinate_error_cm"))
        usable = bool(
            reproj is not None
            and homography is not None
            and reproj <= self.thresholds.calibration_reprojection_px
            and homography <= self.thresholds.homography_error_cm
        )
        reason = "Calibration passes thresholds." if usable else "Calibration missing or above accepted error thresholds."
        return CalibrationReadiness(reproj, homography, pitch, data.get("per_camera", {}), usable, reason)

    def sync(self, sync_report: dict[str, Any] | None, replay_fps: float) -> SyncReadiness:
        if sync_report is None:
            sync_error = 0.0
            dropped = 0
        else:
            sync_error = _float_or_none(sync_report.get("sync_error_ms"))
            if sync_error is None:
                frame_delta = _float_or_none(sync_report.get("frame_delta")) or 0.0
                fps = max(1.0, replay_fps)
                sync_error = frame_delta * (1000.0 / fps)
            dropped = int(sync_report.get("dropped_frames", 0))
        usable = bool(sync_error is not None and sync_error <= self.thresholds.sync_error_ms and replay_fps >= self.thresholds.replay_fps)
        reason = "Sync/replay timing passes thresholds." if usable else "Sync error or replay FPS is outside threshold."
        return SyncReadiness(round(sync_error, 3) if sync_error is not None else None, dropped, round(replay_fps, 2), usable, reason)

    def evaluate(
        self,
        proposed_decision: str,
        decision_confidence: float,
        model: dict[str, Any],
        tracking: list[dict[str, Any]],
        calibration: CalibrationReadiness,
        sync: SyncReadiness,
    ) -> DecisionGateResult:
        failed: list[str] = []
        map50 = _float_or_none(model.get("map50"))
        recall = _float_or_none(model.get("ball_recall"))
        if map50 is None or map50 < self.thresholds.model_map50:
            failed.append("model_map50")
        if recall is None or recall < self.thresholds.ball_recall:
            failed.append("ball_recall")
        if not calibration.usable:
            failed.append("calibration")
        if not sync.usable:
            failed.append("sync")
        if decision_confidence < self.thresholds.decision_confidence:
            failed.append("decision_confidence")

        if not tracking:
            failed.append("tracking")
        for idx, item in enumerate(tracking):
            score = _float_or_none(item.get("score")) or 0.0
            max_gap = int(item.get("max_missing_gap", item.get("missing_frames", 999)))
            reliability = item.get("reliability")
            if score < self.thresholds.tracking_score or max_gap > self.thresholds.max_missing_gap or reliability not in {"medium", "high"}:
                failed.append(f"tracking_camera_{idx}")

        allowed = not failed and proposed_decision in {"OUT", "NOT OUT", "UMPIRE'S CALL"}
        return DecisionGateResult(
            allowed=allowed,
            display_decision=proposed_decision if allowed else "REVIEW INCONCLUSIVE",
            confidence=round(decision_confidence, 3),
            failed_gates=failed,
            metrics={
                "model_map50": map50,
                "ball_recall": recall,
                "calibration_reprojection_error_px": calibration.reprojection_error_px,
                "homography_validation_error_cm": calibration.homography_error_cm,
                "sync_error_ms": sync.sync_error_ms,
                "replay_fps": sync.replay_fps,
            },
        )


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
