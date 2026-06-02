"""Kalman-filter cricket ball tracking and trajectory drawing."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

import cv2
import numpy as np

from config.settings import (
    KALMAN_MEASUREMENT_NOISE,
    KALMAN_PROCESS_NOISE,
    MAX_MISSING_FRAMES,
    TRAJECTORY_HISTORY,
)
from core.ball_detector import DetectionResult
from utils.helpers import draw_trajectory


@dataclass(slots=True)
class TrackPoint:
    frame_id: int
    timestamp_ms: float
    camera_id: int
    x: float
    y: float
    vx: float
    vy: float
    speed_px_s: float
    direction_deg: float
    confidence: float
    predicted: bool


class CricketKalmanFilter:
    """Constant-velocity Kalman filter with state [x, y, vx, vy]."""

    def __init__(self) -> None:
        self.filter = cv2.KalmanFilter(4, 2)
        self.filter.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
        )
        self.filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.filter.processNoiseCov = np.eye(4, dtype=np.float32) * KALMAN_PROCESS_NOISE
        self.filter.measurementNoiseCov = np.eye(2, dtype=np.float32) * KALMAN_MEASUREMENT_NOISE
        self.filter.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False

    def initialize(self, x: float, y: float) -> None:
        state = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.filter.statePre = state
        self.filter.statePost = state
        self.initialized = True

    def predict(self) -> tuple[float, float, float, float]:
        state = self.filter.predict()
        return float(state[0]), float(state[1]), float(state[2]), float(state[3])

    def correct(self, x: float, y: float) -> tuple[float, float, float, float]:
        state = self.filter.correct(np.array([[x], [y]], dtype=np.float32))
        return float(state[0]), float(state[1]), float(state[2]), float(state[3])


class BallTracker:
    """Tracks the primary ball using YOLO detections and predicted gaps."""

    def __init__(self, fps: float = 60.0) -> None:
        self.fps = fps
        self.kalman = CricketKalmanFilter()
        self.history: list[TrackPoint] = []
        self.missing_frames = 0

    def update(self, result: DetectionResult) -> Optional[TrackPoint]:
        detection = result.best
        if detection is not None and not self.kalman.initialized:
            self.kalman.initialize(detection.cx, detection.cy)

        if not self.kalman.initialized:
            return None

        self.kalman.predict()
        if detection is None:
            self.missing_frames += 1
            x, y, vx, vy = self.kalman.predict()
            confidence = 0.0
            predicted = True
        else:
            self.missing_frames = 0
            x, y, vx, vy = self.kalman.correct(detection.cx, detection.cy)
            confidence = detection.confidence
            predicted = False

        if self.missing_frames > MAX_MISSING_FRAMES:
            self.reset()
            return None

        speed_px_s = math.hypot(vx, vy) * self.fps
        direction_deg = math.degrees(math.atan2(-vy, vx)) % 360.0
        point = TrackPoint(
            result.frame_id,
            result.timestamp_ms,
            result.camera_id,
            x,
            y,
            vx * self.fps,
            vy * self.fps,
            speed_px_s,
            direction_deg,
            confidence,
            predicted,
        )
        self.history.append(point)
        if len(self.history) > TRAJECTORY_HISTORY:
            self.history.pop(0)
        return point

    def draw(self, frame: np.ndarray) -> np.ndarray:
        points = [(int(point.x), int(point.y)) for point in self.history]
        draw_trajectory(frame, points)
        if self.history:
            latest = self.history[-1]
            color = (0, 110, 255) if latest.predicted else (0, 240, 255)
            center = (int(latest.x), int(latest.y))
            cv2.circle(frame, center, 8, color, 2, cv2.LINE_AA)
            end = (int(latest.x + latest.vx * 0.05), int(latest.y + latest.vy * 0.05))
            cv2.arrowedLine(frame, center, end, (255, 120, 40), 2, cv2.LINE_AA, tipLength=0.35)
            cv2.putText(frame, f"{latest.speed_px_s:.0f}px/s {latest.direction_deg:.0f}deg", (center[0] + 12, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
        return frame

    def export_rows(self) -> list[dict[str, float | int | bool]]:
        return [asdict(point) for point in self.history]

    def reset(self) -> None:
        self.kalman = CricketKalmanFilter()
        self.history.clear()
        self.missing_frames = 0
