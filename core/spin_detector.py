"""Ball spin and seam orientation estimation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class SpinAnalysis:
    rpm_estimate: float
    axis_estimate: tuple[float, float, float]
    swing_direction: str
    movement_cm_at_pitch: float
    seam_angle_deg: float | None
    confidence: float


class SpinDetector:
    def analyze(self, points: list[tuple[float, float]], fps: float, ball_patch: np.ndarray | None = None) -> SpinAnalysis:
        rpm = 0.0
        movement = 0.0
        swing = "UNKNOWN"
        confidence = 0.0
        if len(points) >= 5:
            pts = np.asarray(points, dtype=float)
            curvature = np.linalg.norm(np.diff(pts, n=2, axis=0), axis=1).mean()
            rpm = float(curvature * fps * 3.0)
            movement = float(curvature * 0.1)
            delta_x = pts[-1, 0] - pts[0, 0]
            swing = "OUTSWING" if delta_x > 0 else "INSWING"
            confidence = min(0.8, len(points) / 60.0)
        seam_angle = self.estimate_seam_angle(ball_patch) if ball_patch is not None else None
        if seam_angle is not None:
            confidence = min(1.0, confidence + 0.2)
        return SpinAnalysis(rpm, (0.0, 0.0, 1.0), swing, movement, seam_angle, confidence)

    def estimate_seam_angle(self, patch: np.ndarray) -> float | None:
        if patch.size == 0:
            return None
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
        edges = cv2.Canny(gray, 60, 160)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, max(8, min(gray.shape[:2]) // 3))
        if lines is None:
            return None
        theta = float(lines[0][0][1])
        return float(np.degrees(theta))
