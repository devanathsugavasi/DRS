"""LBW decision engine for DRS review output."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from config.settings import PITCH_LENGTH_M, STUMP_HEIGHT_M, STUMP_WIDTH_M
from core.trajectory import TrajectoryPrediction


@dataclass(slots=True)
class LBWDecision:
    pitching_point: tuple[float, float] | None
    impact_point: tuple[float, float] | None
    line_of_impact: str
    hitting_stumps: bool
    decision: str
    confidence: float
    reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


class LBWDecisionEngine:
    """Rule-aware LBW suggestion engine fed by tracked and predicted ball path."""

    def __init__(self, stump_width_m: float = STUMP_WIDTH_M, stump_height_m: float = STUMP_HEIGHT_M) -> None:
        self.stump_half_width_m = stump_width_m / 2.0
        self.stump_height_m = stump_height_m

    def evaluate(
        self,
        trajectory: TrajectoryPrediction,
        impact_point_m: tuple[float, float] | None = None,
        batter_is_right_handed: bool = True,
    ) -> LBWDecision:
        pitching = self._detect_pitching_point(trajectory)
        line = self._line_of_impact(impact_point_m)
        reasons: list[str] = []

        if pitching is None:
            reasons.append("Pitching point not detected")
        else:
            reasons.append(f"Pitched at x={pitching[0]:.2f}m y={pitching[1]:.2f}m")

        if impact_point_m is None:
            reasons.append("Impact point not supplied")
        else:
            reasons.append(f"Impact line: {line}")

        if trajectory.wicket_collision:
            reasons.append("Predicted path intersects the stumps")
        else:
            reasons.append("Predicted path misses the stumps")

        outside_leg = False
        if pitching is not None:
            outside_leg = pitching[1] < -self.stump_half_width_m if batter_is_right_handed else pitching[1] > self.stump_half_width_m

        out = bool(trajectory.wicket_collision and not outside_leg and line in {"in_line", "umpires_call"})
        confidence = 0.82 if out else 0.68
        if outside_leg:
            reasons.append("Pitching outside leg")
        return LBWDecision(
            pitching_point=pitching,
            impact_point=impact_point_m,
            line_of_impact=line,
            hitting_stumps=trajectory.wicket_collision,
            decision="OUT" if out else "NOT OUT",
            confidence=confidence,
            reasons=reasons,
        )

    def visualize(
        self,
        frame: np.ndarray,
        decision: LBWDecision,
        stump_box: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray:
        text = f"{decision.decision} | {decision.confidence:.0%}"
        color = (0, 0, 255) if decision.decision == "OUT" else (0, 180, 255)
        cv2.rectangle(frame, (16, 52), (360, 118), (15, 15, 15), -1)
        cv2.putText(frame, text, (28, 92), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        if stump_box is not None:
            cv2.rectangle(frame, stump_box[:2], stump_box[2:], color, 2, cv2.LINE_AA)
        return frame

    def _detect_pitching_point(self, trajectory: TrajectoryPrediction) -> tuple[float, float] | None:
        if trajectory.bounce_index is None:
            return None
        point = trajectory.points[trajectory.bounce_index]
        return point.x, point.y

    def _line_of_impact(self, impact_point_m: tuple[float, float] | None) -> str:
        if impact_point_m is None:
            return "unknown"
        _, y = impact_point_m
        if abs(y) <= self.stump_half_width_m:
            return "in_line"
        if abs(y) <= self.stump_half_width_m * 1.5:
            return "umpires_call"
        return "outside_line"


def default_wicket_x() -> float:
    return PITCH_LENGTH_M / 2.0
