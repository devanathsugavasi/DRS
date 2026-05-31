"""Command-line entry point for the cricket DRS system."""

from __future__ import annotations

import argparse
import time

import cv2

from core.integration import DRSPipeline
from ui.dashboard import run_dashboard


def parse_camera_ids(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def list_cameras(scan_limit: int) -> None:
    print("Scanning cameras...")
    found = []
    for camera_id in range(scan_limit):
        capture = cv2.VideoCapture(camera_id, cv2.CAP_ANY)
        ok, frame = capture.read() if capture.isOpened() else (False, None)
        if ok and frame is not None:
            height, width = frame.shape[:2]
            fps = capture.get(cv2.CAP_PROP_FPS)
            print(f"camera {camera_id}: OK  {width}x{height}  fps={fps:.1f}")
            found.append(camera_id)
        else:
            print(f"camera {camera_id}: unavailable")
        capture.release()
    if found:
        print("Use: python drs_app.py --cameras " + ",".join(str(item) for item in found))
    else:
        print("No cameras opened. Check USB/capture-card permissions and cables.")


def run_headless(camera_ids: list[int], seconds: float, record: bool) -> None:
    pipeline = DRSPipeline(camera_ids=camera_ids, record=record)
    pipeline.start()
    started = time.perf_counter()
    try:
        while time.perf_counter() - started < seconds:
            state = pipeline.process_once()
            for camera_id, output in state.frames.items():
                cv2.imshow(f"DRS cam {camera_id}", output.annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        pipeline.export_tracking()
        pipeline.detector.flush("csv")
        pipeline.stop()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Cricket DRS prototype")
    parser.add_argument("--cameras", default="0,1", help="Comma-separated camera indices")
    parser.add_argument("--record", action="store_true", help="Save synchronized camera streams")
    parser.add_argument("--headless", action="store_true", help="Use OpenCV windows instead of Tkinter dashboard")
    parser.add_argument("--seconds", type=float, default=60.0, help="Runtime for headless mode")
    parser.add_argument("--list-cameras", action="store_true", help="Scan available camera indices and exit")
    parser.add_argument("--scan-limit", type=int, default=10, help="Number of camera indices to scan")
    parser.add_argument("--api", action="store_true", help="Run FastAPI backend for Electron dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="API host")
    parser.add_argument("--port", type=int, default=8765, help="API port")
    args = parser.parse_args()

    if args.list_cameras:
        list_cameras(args.scan_limit)
        return

    camera_ids = parse_camera_ids(args.cameras)
    if args.api:
        from core.api_server import run_api

        run_api(camera_ids, args.record, args.host, args.port)
    elif args.headless:
        run_headless(camera_ids, args.seconds, args.record)
    else:
        run_dashboard(camera_ids, args.record)


if __name__ == "__main__":
    main()
