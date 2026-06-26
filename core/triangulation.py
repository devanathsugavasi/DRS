"""Multi-camera triangulation → 3D ball track (drs-3d-v1).

Path 2 of the DRS pipeline: turn 2D ball detections from two (or more)
calibrated, synced cameras into a 3D trajectory in pitch coordinates, fit a
ballistic model, and project forward to the stumps. Output matches the
drs-3d-v1 schema consumed by dashboard/drs_replay.html.

World frame (metres): origin at the striker's stumps base centre.
  x = lateral (towards leg/off), y = along the pitch (towards the bowler),
  z = height. Stumps: half-width 0.1145 m, height 0.711 m, pitch length 20.12 m.

This module is camera-agnostic: feed it intrinsics K and either known
extrinsics or anchor correspondences (crease corners, stump bases) and it
recovers each camera's projection matrix via PnP. The math is exact; accuracy
in the field depends entirely on calibration + frame sync quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

PITCH_LENGTH_Y = 20.12
STUMP_HALF_WIDTH = 0.1145
STUMP_HEIGHT = 0.711
GRAVITY = 9.81


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV-convention extrinsics (x right, y down, z forward) for a camera."""
    eye = np.asarray(eye, float); target = np.asarray(target, float); up = np.asarray(up, float)
    z = target - eye; z /= np.linalg.norm(z)
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    t = -R @ eye
    return R, t.reshape(3, 1)


def projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return K @ np.hstack([R, t.reshape(3, 1)])


def project(P: np.ndarray, pts3d: np.ndarray) -> np.ndarray:
    """Project Nx3 world points to Nx2 image points through P (3x4)."""
    pts3d = np.asarray(pts3d, float)
    h = np.hstack([pts3d, np.ones((len(pts3d), 1))])
    img = (P @ h.T).T
    return img[:, :2] / img[:, 2:3]


def calibrate_pnp(world_pts: np.ndarray, image_pts: np.ndarray, K: np.ndarray,
                  dist=None) -> tuple[np.ndarray, float]:
    """Recover a camera's projection matrix from anchor correspondences (DLT/PnP).

    Returns (P, mean_reprojection_error_px). Anchors = points with known world
    coordinates visible in the image (crease corners, stump bases, pitch corners).
    """
    world_pts = np.asarray(world_pts, np.float32)
    image_pts = np.asarray(image_pts, np.float32)
    ok, rvec, tvec = cv2.solvePnP(world_pts, image_pts, K, dist,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError("solvePnP failed — check anchor correspondences")
    R, _ = cv2.Rodrigues(rvec)
    P = projection_matrix(K, R, tvec)
    reproj = project(P, world_pts)
    err = float(np.mean(np.linalg.norm(reproj - image_pts, axis=1)))
    return P, err


def triangulate(P1: np.ndarray, P2: np.ndarray,
                pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Triangulate matched 2D points (Nx2 each) from two cameras → Nx3 world points."""
    pts1 = np.asarray(pts1, float).T
    pts2 = np.asarray(pts2, float).T
    h = cv2.triangulatePoints(P1, P2, pts1, pts2)  # 4xN homogeneous
    return (h[:3] / h[3]).T


def fit_ballistic_and_project(traj: np.ndarray, times: Sequence[float],
                              impact_idx: int, y_plane: float = 0.0,
                              n_pred: int = 6) -> list[dict]:
    """From measured 3D points, estimate velocity at impact and project to the
    stumps plane (y = y_plane) under gravity. Returns prediction points."""
    traj = np.asarray(traj, float)
    i = min(impact_idx, len(traj) - 1)
    p = traj[i]
    # velocity from the last short segment before impact
    j = max(0, i - 2)
    dt = max(times[i] - times[j], 1e-3)
    v = (traj[i] - traj[j]) / dt
    vy = v[1] if abs(v[1]) > 1e-3 else -1.0
    t_to_stumps = (y_plane - p[1]) / vy
    if t_to_stumps <= 0:
        t_to_stumps = abs((p[1] - y_plane) / vy)
    out = []
    for k in range(1, n_pred + 1):
        tt = t_to_stumps * k / n_pred
        x = p[0] + v[0] * tt
        y = p[1] + v[1] * tt
        z = p[2] + v[2] * tt - 0.5 * GRAVITY * tt * tt
        pt = {"t": round(times[i] + tt, 3), "x": round(float(x), 4),
              "y": round(float(max(y, 0.0)), 4), "z": round(float(max(z, 0.0)), 4)}
        if k == n_pred:
            pt["is_stumps"] = True
        out.append(pt)
    return out


def build_drs3d(traj: np.ndarray, times: Sequence[float],
                bounce_idx: int | None, impact_idx: int | None,
                source: str = "triangulation", delivery_id: str = "0.0") -> dict:
    """Assemble a drs-3d-v1 payload from a measured 3D trajectory."""
    traj = np.asarray(traj, float)
    trajectory = []
    for k, (pt, t) in enumerate(zip(traj, times)):
        d = {"t": round(float(t), 3), "x": round(float(pt[0]), 4),
             "y": round(float(pt[1]), 4), "z": round(float(pt[2]), 4)}
        if bounce_idx is not None and k == bounce_idx:
            d["is_bounce"] = True
        if impact_idx is not None and k == impact_idx:
            d["is_impact"] = True
        trajectory.append(d)
    prediction = []
    if impact_idx is not None:
        prediction = fit_ballistic_and_project(traj, times, impact_idx)
    return {
        "match_id": "drs", "delivery_id": delivery_id, "format": "drs-3d-v1",
        "source": source, "pitch_length_y": PITCH_LENGTH_Y,
        "stump_half_width_m": STUMP_HALF_WIDTH, "stump_height_m": STUMP_HEIGHT,
        "trajectory": trajectory, "prediction": prediction,
    }


@dataclass
class StereoRig:
    """Two calibrated cameras ready to triangulate ball detections."""
    K: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    reproj_err1: float = 0.0
    reproj_err2: float = 0.0

    @classmethod
    def from_anchors(cls, K, world_anchors, img_anchors_cam1, img_anchors_cam2):
        P1, e1 = calibrate_pnp(world_anchors, img_anchors_cam1, K)
        P2, e2 = calibrate_pnp(world_anchors, img_anchors_cam2, K)
        return cls(K=K, P1=P1, P2=P2, reproj_err1=e1, reproj_err2=e2)

    def track_3d(self, pts_cam1, pts_cam2) -> np.ndarray:
        return triangulate(self.P1, self.P2, pts_cam1, pts_cam2)
