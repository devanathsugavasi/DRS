"""Run camera discovery and benchmark checks for Cricket DRS."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from discover_cameras import discover


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Cricket DRS camera readiness checks")
    parser.add_argument("--scan-limit", type=int, default=10)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()

    cameras = discover(args.scan_limit)
    camera_ids = [str(camera["id"]) for camera in cameras if camera["ok"]]
    benchmark_report = None
    if camera_ids:
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("benchmark_cameras.py")),
                "--camera-ids",
                ",".join(camera_ids),
                "--frames",
                str(args.frames),
                "--reports-dir",
                args.reports_dir,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        benchmark_report = completed.stdout

    meeting_fps = sum(1 for camera in cameras if float(camera["fps"] or 0.0) >= 25.0)
    report = {
        "created_at": datetime.utcnow().isoformat(),
        "cameras_found": len(cameras),
        "camera_ids": [camera["id"] for camera in cameras],
        "cameras_meeting_fps_target": meeting_fps,
        "production_ready": len(cameras) >= 2 and meeting_fps >= 2,
        "benchmark_output": benchmark_report,
    }
    out_dir = Path(args.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"camera_test_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report: {out_path}")


if __name__ == "__main__":
    main()
