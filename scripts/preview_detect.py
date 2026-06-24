"""Preview cricket-ball detections on any video clip.

Pulls N frames from a video, runs the current YOLO model on each, draws the
boxes, and writes annotated JPGs you can open in VS Code / Finder. Use it to
eyeball whether a clip is usable for DRS (tight side-on delivery) before
committing it to the training set.

Examples
--------
    # 24 frames spread across a clip, save to ./preview_out
    python scripts/preview_detect.py --video /Volumes/Lexar/PRIVATE/AVCHD/BDMV/STREAM/00001.MTS

    # sample a 20s window starting at 3 min, every 0.5s
    python scripts/preview_detect.py --video CLIP.MTS --start 180 --duration 20 --fps 2
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preview ball detections on a clip")
    p.add_argument("--video", type=Path, required=True, help="Path to a video file (.MTS/.mp4/...)")
    p.add_argument("--model", type=Path, default=Path("models/cricket_ball_yolov8.pt"))
    p.add_argument("--out", type=Path, default=Path("preview_out"))
    p.add_argument("--frames", type=int, default=24, help="Max frames to sample")
    p.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    p.add_argument("--duration", type=float, default=None, help="Window length in seconds (default: whole clip)")
    p.add_argument("--fps", type=float, default=2.0, help="Frames sampled per second of the window")
    p.add_argument("--conf", type=float, default=0.15, help="Min detection confidence")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    t0 = time.time()
    model = YOLO(str(args.model))
    print(f"model loaded in {time.time() - t0:.1f}s", flush=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {args.video}")
    vfps = cap.get(cv2.CAP_PROP_FPS) or 50.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(vfps / args.fps)))
    start_frame = int(args.start * vfps)
    end_frame = total if args.duration is None else min(total, start_frame + int(args.duration * vfps))

    saved = hits = 0
    fidx = start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    while fidx < end_frame and saved < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        if (fidx - start_frame) % step == 0:
            r = model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
            confs = []
            for b in r.boxes:
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                c = float(b.conf[0])
                confs.append(round(c, 2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, f"{c:.2f}", (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            t_sec = fidx / vfps
            name = f"{args.video.stem}_t{t_sec:07.2f}.jpg"
            cv2.imwrite(str(args.out / name), frame)
            saved += 1
            if confs:
                hits += 1
                print(f"  {name}: {len(confs)} det conf={confs}", flush=True)
        fidx += 1
    cap.release()
    print(f"\nsaved {saved} frames to {args.out}/  ({hits} had detections)")
    print(f"open the folder in VS Code to inspect: code {args.out}")


if __name__ == "__main__":
    main()
