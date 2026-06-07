"""Benchmark multi-camera capture FPS, latency, and sync variance."""

from __future__ import annotations

import argparse
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

import cv2


def benchmark_camera(camera_id: int, frames: int) -> dict[str, object]:
    cap = cv2.VideoCapture(camera_id, cv2.CAP_ANY)
    if not cap.isOpened():
        return {"id": camera_id, "status": "offline", "fps": 0.0, "drop_rate": 1.0, "latencies_ms": []}

    timestamps = []
    latencies = []
    ok_count = 0
    started = time.perf_counter()
    for _ in range(frames):
        t0 = time.perf_counter()
        ok, _frame = cap.read()
        captured = time.perf_counter()
        if ok:
            ok_count += 1
            timestamps.append(captured)
            latencies.append((captured - t0) * 1000.0)
    cap.release()

    elapsed = max(0.001, time.perf_counter() - started)
    fps = ok_count / elapsed
    drop_rate = 1.0 - (ok_count / max(1, frames))
    return {
        "id": camera_id,
        "status": "ok" if fps >= 20 and drop_rate <= 0.05 else "degraded",
        "fps": round(fps, 3),
        "drop_rate": round(drop_rate, 4),
        "max_latency_ms": round(max(latencies, default=0.0), 3),
        "mean_latency_ms": round(mean(latencies) if latencies else 0.0, 3),
        "timestamps": timestamps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark physical cameras for Cricket DRS")
    parser.add_argument("--camera-ids", default="0,1")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()

    camera_ids = [int(item.strip()) for item in args.camera_ids.split(",") if item.strip()]
    results: dict[int, dict[str, object]] = {}
    threads = [
        threading.Thread(
            target=lambda cam_id=cam_id: results.update({cam_id: benchmark_camera(cam_id, args.frames)}),
            daemon=True,
        )
        for cam_id in camera_ids
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    timestamps = [item["timestamps"] for item in results.values() if item.get("timestamps")]
    sync_delta_ms = 0.0
    if timestamps:
        min_len = min(len(items) for items in timestamps)
        deltas = [
            (max(items[idx] for items in timestamps) - min(items[idx] for items in timestamps)) * 1000.0
            for idx in range(min_len)
        ]
        sync_delta_ms = max(deltas, default=0.0)

    cameras = []
    for item in results.values():
        clean = dict(item)
        clean.pop("timestamps", None)
        cameras.append(clean)

    report = {
        "created_at": datetime.utcnow().isoformat(),
        "frames_requested": args.frames,
        "sync_delta_ms": round(sync_delta_ms, 3),
        "sync_ok": sync_delta_ms < 33.0,
        "cameras": sorted(cameras, key=lambda item: int(item["id"])),
    }
    out_dir = Path(args.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"camera_benchmark_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report: {out_path}")


if __name__ == "__main__":
    main()
