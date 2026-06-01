"""Write measured calibration readiness metrics after camera calibration validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Store measured DRS calibration readiness metrics")
    parser.add_argument("--reprojection-error-px", type=float, required=True)
    parser.add_argument("--homography-error-cm", type=float, required=True)
    parser.add_argument("--pitch-coordinate-error-cm", type=float, required=True)
    parser.add_argument("--per-camera-json", default="{}", help="JSON object with per-camera reprojection/sync notes")
    parser.add_argument("--out", default="data/calibration/readiness.json")
    args = parser.parse_args()

    try:
        per_camera = json.loads(args.per_camera_json)
    except json.JSONDecodeError as exc:
        raise SystemExit("--per-camera-json must be valid JSON") from exc

    payload = {
        "reprojection_error_px": args.reprojection_error_px,
        "homography_error_cm": args.homography_error_cm,
        "pitch_coordinate_error_cm": args.pitch_coordinate_error_cm,
        "per_camera": per_camera,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
