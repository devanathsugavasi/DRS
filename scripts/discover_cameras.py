"""Discover local OpenCV camera devices for Cricket DRS."""

from __future__ import annotations

import argparse

import cv2


def discover(scan_limit: int) -> list[dict[str, object]]:
    cameras = []
    for camera_id in range(scan_limit):
        cap = cv2.VideoCapture(camera_id, cv2.CAP_ANY)
        if cap.isOpened():
            ret, frame = cap.read()
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            cameras.append(
                {
                    "id": camera_id,
                    "width": width,
                    "height": height,
                    "fps": round(fps, 3),
                    "ok": bool(ret and frame is not None),
                }
            )
        cap.release()
    return cameras


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover physical cameras for Cricket DRS")
    parser.add_argument("--scan-limit", type=int, default=10)
    args = parser.parse_args()

    cameras = discover(args.scan_limit)
    if not cameras:
        print("No cameras opened.")
        return
    for camera in cameras:
        status = "OK" if camera["ok"] else "no frame"
        print(
            f"Camera {camera['id']}: {camera['width']}x{camera['height']} "
            f"@ {camera['fps']:.1f} FPS -- {status}"
        )


if __name__ == "__main__":
    main()
