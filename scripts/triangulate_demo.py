"""Synthetic-stereo proof of the triangulation pipeline (Path 2).

Builds a known 3D delivery, projects it into two virtual calibrated cameras,
then runs the real pipeline (PnP calibration from pitch anchors → triangulation
→ ballistic projection → drs-3d-v1) and reports how accurately the 3D ball is
recovered. This validates the math end-to-end with no footage. On real footage
the same code runs once you supply intrinsics + synced 2D ball detections.

    python scripts/triangulate_demo.py --noise-px 0.5 --out dashboard/triangulated_track.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.triangulation import (
    PITCH_LENGTH_Y, StereoRig, build_drs3d, look_at,
    projection_matrix, project,
)


def ground_truth_delivery() -> tuple[np.ndarray, list[float], int, int]:
    """A realistic LBW delivery in world metres (x lateral, y down pitch, z up)."""
    keys = [  # t, x, y, z
        (0.00, -0.21, 18.50, 2.10),
        (0.05, -0.18, 15.80, 1.70),
        (0.10, -0.15, 13.00, 1.20),
        (0.15, -0.11, 10.30, 0.55),
        (0.18, -0.09,  9.20, 0.05),   # bounce
        (0.22, -0.06,  6.40, 0.40),
        (0.26, -0.03,  3.80, 0.62),
        (0.29, -0.01,  2.10, 0.66),   # impact (pad)
    ]
    arr = np.array(keys, float)
    times = arr[:, 0].tolist()
    traj = arr[:, 1:]
    return traj, times, 4, 7  # bounce_idx, impact_idx


def pitch_anchors() -> np.ndarray:
    """Known calibration points on the pitch (stump bases + crease/pitch corners)."""
    h = 0.1145
    return np.array([
        [-h, 0.0, 0.0], [0.0, 0.0, 0.0], [h, 0.0, 0.0],            # striker stumps base
        [-h, PITCH_LENGTH_Y, 0.0], [0.0, PITCH_LENGTH_Y, 0.0], [h, PITCH_LENGTH_Y, 0.0],  # bowler stumps
        [-1.32, 0.0, 0.0], [1.32, 0.0, 0.0],                        # return crease corners (striker)
        [-1.32, PITCH_LENGTH_Y, 0.0], [1.32, PITCH_LENGTH_Y, 0.0],  # return crease (bowler)
        [0.0, 1.22, 0.0], [0.0, 18.90, 0.0],                        # popping creases on centre line
    ], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise-px", type=float, default=0.5, help="2D detection noise (px std)")
    ap.add_argument("--out", type=Path, default=Path("dashboard/triangulated_track.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(7)

    # intrinsics (1280x720, ~50deg fov)
    K = np.array([[1100.0, 0, 640.0], [0, 1100.0, 360.0], [0, 0, 1.0]])

    # ground-truth cameras: A high behind the bowler's arm, B square-leg side-on
    Ra, ta = look_at(eye=(0.0, 26.0, 7.5), target=(0.0, 8.0, 0.4))
    Rb, tb = look_at(eye=(11.0, 9.0, 3.5), target=(0.0, 9.0, 0.5))
    Pa_gt = projection_matrix(K, Ra, ta)
    Pb_gt = projection_matrix(K, Rb, tb)

    anchors = pitch_anchors()
    img_a = project(Pa_gt, anchors) + rng.normal(0, args.noise_px, (len(anchors), 2))
    img_b = project(Pb_gt, anchors) + rng.normal(0, args.noise_px, (len(anchors), 2))

    # --- Path 2 pipeline ---
    rig = StereoRig.from_anchors(K, anchors, img_a, img_b)
    print(f"calibration reprojection error  cam A: {rig.reproj_err1:.2f} px  cam B: {rig.reproj_err2:.2f} px")

    traj_gt, times, bounce_idx, impact_idx = ground_truth_delivery()
    ball_a = project(Pa_gt, traj_gt) + rng.normal(0, args.noise_px, (len(traj_gt), 2))
    ball_b = project(Pb_gt, traj_gt) + rng.normal(0, args.noise_px, (len(traj_gt), 2))

    traj_rec = rig.track_3d(ball_a, ball_b)
    err_mm = np.linalg.norm(traj_rec - traj_gt, axis=1) * 1000.0
    print(f"3D recovery error: mean {err_mm.mean():.1f} mm  max {err_mm.max():.1f} mm  "
          f"(over {len(traj_gt)} points, {args.noise_px}px detection noise)")

    payload = build_drs3d(traj_rec, times, bounce_idx, impact_idx,
                          source=f"triangulation (synthetic stereo, {args.noise_px}px noise)",
                          delivery_id="demo.1")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    sp = payload["prediction"][-1] if payload["prediction"] else {}
    print(f"projected ball at stumps: x={sp.get('x')} m  z={sp.get('z')} m  "
          f"(hitting if |x|<=0.1145 and z<=0.711)")
    print(f"wrote drs-3d-v1 → {args.out}  (load it in dashboard/drs_replay.html)")


if __name__ == "__main__":
    main()
