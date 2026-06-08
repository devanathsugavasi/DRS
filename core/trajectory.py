"""Projectile trajectory prediction for cricket DRS."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from scipy.integrate import solve_ivp

from config.settings import BOUNCE_RESTITUTION, GRAVITY_MPS2
from core.ball_tracker import TrackPoint


@dataclass(slots=True)
class TrajectoryPoint:
    t: float
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float


@dataclass(slots=True)
class TrajectoryPrediction:
    points: list[TrajectoryPoint]
    bounce_index: int | None
    wicket_collision: bool
    wicket_point: TrajectoryPoint | None

    def to_dict(self) -> dict:
        return {
            "points": [asdict(point) for point in self.points],
            "bounce_index": self.bounce_index,
            "wicket_collision": self.wicket_collision,
            "wicket_point": asdict(self.wicket_point) if self.wicket_point else None,
        }


class TrajectoryPredictor:
    """Predicts future ball path from tracked world coordinates."""

    def __init__(self, restitution: float = BOUNCE_RESTITUTION) -> None:
        self.restitution = restitution

    def predict_from_world_points(
        self,
        positions_m: list[tuple[float, float, float]],
        timestamps_s: list[float],
        horizon_s: float = 1.2,
        dt: float = 0.01,
        wicket_x_m: float | None = None,
        stump_half_width_m: float = 0.1143,
        stump_height_m: float = 0.711,
    ) -> TrajectoryPrediction:
        if len(positions_m) < 2:
            raise ValueError("At least two tracked positions are required")

        p0 = np.asarray(positions_m[-1], dtype=float)
        p1 = np.asarray(positions_m[-2], dtype=float)
        delta_t = max(1e-3, timestamps_s[-1] - timestamps_s[-2])
        velocity = (p0 - p1) / delta_t
        state0 = np.r_[p0, velocity]

        def ode(_t: float, state: np.ndarray) -> np.ndarray:
            return np.array([state[3], state[4], state[5], 0.0, 0.0, -GRAVITY_MPS2])

        t_eval = np.arange(0.0, horizon_s + dt, dt)
        solution = solve_ivp(ode, (0.0, horizon_s), state0, t_eval=t_eval, max_step=dt)

        points: list[TrajectoryPoint] = []
        bounce_index = None
        bounced = False
        for idx, state in enumerate(solution.y.T):
            x, y, z, vx, vy, vz = state
            if z <= 0.0 and not bounced and idx > 0:
                bounce_index = idx
                bounced = True
                vz = abs(vz) * self.restitution
                z = 0.0
            points.append(TrajectoryPoint(float(solution.t[idx]), float(x), float(y), max(0.0, float(z)), float(vx), float(vy), float(vz)))

        wicket_point = self._find_wicket_collision(points, wicket_x_m, stump_half_width_m, stump_height_m)
        return TrajectoryPrediction(points, bounce_index, wicket_point is not None, wicket_point)

    def approximate_world_from_track(
        self,
        points: list[TrackPoint],
        pixels_per_meter: float,
    ) -> tuple[list[tuple[float, float, float]], list[float]]:
        positions = [(point.x / pixels_per_meter, point.y / pixels_per_meter, 0.12) for point in points]
        t0 = points[0].timestamp_ms / 1000.0
        times = [(point.timestamp_ms / 1000.0) - t0 for point in points]
        return positions, times

    def overlay(self, frame: np.ndarray, image_points: list[tuple[int, int]]) -> np.ndarray:
        if len(image_points) >= 2:
            cv2.polylines(frame, [np.asarray(image_points, dtype=np.int32)], False, (255, 60, 40), 2, cv2.LINE_AA)
        return frame

    def _find_wicket_collision(
        self,
        points: list[TrajectoryPoint],
        wicket_x_m: float | None,
        stump_half_width_m: float,
        stump_height_m: float,
    ) -> TrajectoryPoint | None:
        if wicket_x_m is None:
            return None
        for previous, current in zip(points, points[1:]):
            prev_delta = previous.x - wicket_x_m
            curr_delta = current.x - wicket_x_m
            crossed_plane = prev_delta == 0.0 or curr_delta == 0.0 or (prev_delta < 0 < curr_delta) or (curr_delta < 0 < prev_delta)
            if not crossed_plane:
                continue

            span = current.x - previous.x
            ratio = 0.0 if abs(span) < 1e-9 else (wicket_x_m - previous.x) / span
            ratio = max(0.0, min(1.0, ratio))
            y = previous.y + ((current.y - previous.y) * ratio)
            z = previous.z + ((current.z - previous.z) * ratio)
            if abs(y) <= stump_half_width_m and 0.0 <= z <= stump_height_m:
                return TrajectoryPoint(
                    t=previous.t + ((current.t - previous.t) * ratio),
                    x=wicket_x_m,
                    y=y,
                    z=z,
                    vx=previous.vx + ((current.vx - previous.vx) * ratio),
                    vy=previous.vy + ((current.vy - previous.vy) * ratio),
                    vz=previous.vz + ((current.vz - previous.vz) * ratio),
                )

        nearest = min(points, key=lambda point: abs(point.x - wicket_x_m), default=None)
        if nearest and abs(nearest.x - wicket_x_m) <= 0.03 and abs(nearest.y) <= stump_half_width_m and 0.0 <= nearest.z <= stump_height_m:
            return nearest
        return None
