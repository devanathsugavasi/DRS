"""Front-foot no-ball detector interface."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class NoBallResult:
    no_ball: bool
    confidence: float
    foot_position_px: tuple[float, float] | None
    crease_line_px: tuple[float, float, float, float] | None
    reason: str


class NoBallDetector:
    """Checks front-foot position against a calibrated popping crease line."""

    def __init__(self, crease_line_px: tuple[float, float, float, float] | None = None) -> None:
        self.crease_line_px = crease_line_px

    def check_delivery(self, frame, foot_keypoints: dict[str, tuple[float, float]] | None) -> NoBallResult:
        if self.crease_line_px is None:
            return NoBallResult(False, 0.0, None, None, "No calibrated popping-crease line available.")
        if not foot_keypoints or "front_foot" not in foot_keypoints:
            return NoBallResult(False, 0.0, None, self.crease_line_px, "Front-foot keypoint missing; pose/foot model required.")
        foot = foot_keypoints["front_foot"]
        x1, y1, x2, y2 = self.crease_line_px
        line_vec = np.array([x2 - x1, y2 - y1], dtype=float)
        foot_vec = np.array([foot[0] - x1, foot[1] - y1], dtype=float)
        cross = float(np.cross(line_vec, foot_vec))
        no_ball = cross > 0.0
        return NoBallResult(no_ball, 0.75, foot, self.crease_line_px, "Front foot evaluated against calibrated crease.")
