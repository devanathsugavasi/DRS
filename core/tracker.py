"""Extended Kalman-style 3D cricket ball tracker with physics readiness limits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from config.settings import AIR_DENSITY, BALL_MASS_KG, BALL_RADIUS_M, DRAG_COEFFICIENT, GRAVITY_MPS2


@dataclass(slots=True)
class Point3D:
    timestamp_ms: float
    x: float
    y: float
    z: float


@dataclass(slots=True)
class StumpImpact:
    hit_probability: float
    impact_point: Point3D
    zone: str
    usable: bool
    reason: str


class ExtendedCricketBallTracker:
    """Tracks a 3D state vector and exposes DRS physics outputs.

    The tracker accepts image detections. Without camera calibration it uses pixel-space pseudo
    coordinates and marks stump-impact predictions as unusable.
    """

    def __init__(self, measurement_noise: float = 0.08, process_noise: float = 0.035) -> None:
        self.state = np.zeros(9, dtype=float)
        self.covariance = np.eye(9, dtype=float)
        self.measurement_noise = measurement_noise
        self.process_noise = process_noise
        self.history: list[Point3D] = []
        self.initialized = False
        self.last_timestamp_ms: float | None = None
        self.calibrated = False
        self.spin_vector = np.zeros(3, dtype=float)

    def update(self, detection_bbox: tuple[float, float, float, float], camera_id: int, timestamp_ms: float) -> Point3D:
        x1, y1, x2, y2 = detection_bbox
        measurement = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, max(0.01, (y2 - y1) / 2.0)])
        if not self.initialized:
            self.state[:3] = measurement
            self.initialized = True
            self.last_timestamp_ms = timestamp_ms
        else:
            dt = max(1e-3, (timestamp_ms - (self.last_timestamp_ms or timestamp_ms)) / 1000.0)
            self._predict(dt)
            self._correct(measurement)
            self.last_timestamp_ms = timestamp_ms
        point = Point3D(timestamp_ms, float(self.state[0]), float(self.state[1]), float(self.state[2]))
        self.history.append(point)
        return point

    def get_trajectory_3d(self) -> list[Point3D]:
        return list(self.history)

    def predict_impact_on_stumps(self) -> StumpImpact | None:
        if len(self.history) < 3:
            return None
        if not self.calibrated:
            return StumpImpact(
                hit_probability=0.0,
                impact_point=self.history[-1],
                zone="UNKNOWN",
                usable=False,
                reason="3D stump impact requires calibrated camera geometry.",
            )
        point = self.history[-1]
        probability = max(0.0, min(1.0, 1.0 - abs(point.x) / 0.25))
        return StumpImpact(probability, point, self._zone(point), probability > 0.5, "calibrated projection")

    def get_pitch_bounce_point(self) -> Point3D | None:
        if len(self.history) < 4:
            return None
        z_values = [point.z for point in self.history]
        idx = int(np.argmin(z_values))
        return self.history[idx]

    def get_ball_velocity_kph(self) -> float:
        return float(np.linalg.norm(self.state[3:6]) * 3.6)

    def get_spin_rpm(self) -> float:
        if len(self.history) < 5:
            return 0.0
        points = np.array([[p.x, p.y, p.z] for p in self.history[-5:]], dtype=float)
        curvature = np.linalg.norm(np.diff(points, n=2, axis=0), axis=1).mean()
        return float(max(0.0, curvature * 60.0))

    def reset(self) -> None:
        self.__init__(self.measurement_noise, self.process_noise)

    def _predict(self, dt: float) -> None:
        velocity = self.state[3:6]
        acceleration = self._physics_acceleration(velocity)
        self.state[:3] += velocity * dt + 0.5 * acceleration * dt * dt
        self.state[3:6] += acceleration * dt
        self.state[6:9] = acceleration
        self.covariance += np.eye(9) * self.process_noise

    def _correct(self, measurement: np.ndarray) -> None:
        residual = measurement - self.state[:3]
        gain = self.covariance[:3, :3] @ np.linalg.inv(
            self.covariance[:3, :3] + np.eye(3) * self.measurement_noise
        )
        correction = gain @ residual
        self.state[:3] += correction
        self.state[3:6] += correction * 0.12

    def _physics_acceleration(self, velocity: np.ndarray) -> np.ndarray:
        speed = max(1e-6, float(np.linalg.norm(velocity)))
        area = math.pi * BALL_RADIUS_M * BALL_RADIUS_M
        drag_mag = 0.5 * DRAG_COEFFICIENT * AIR_DENSITY * area * speed * speed / BALL_MASS_KG
        drag = -drag_mag * velocity / speed
        magnus = np.cross(self.spin_vector, velocity) * 1e-5
        return drag + magnus + np.array([0.0, -GRAVITY_MPS2, 0.0])

    def _zone(self, point: Point3D) -> str:
        if point.z > 0.55:
            return "TOP"
        if point.x < -0.05:
            return "LEG_STUMP"
        if point.x > 0.05:
            return "OFF_STUMP"
        return "MIDDLE"

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.tolist(),
            "trajectory": [asdict(point) for point in self.history],
            "calibrated": self.calibrated,
        }
