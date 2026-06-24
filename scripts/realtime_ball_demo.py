"""Real-time ball-flight tracking demo (single fixed camera).

Streams a video (or live camera) frame-by-frame, isolates the fast-moving ball
with 3-frame differencing inside the fixed pitch ROI, smooths it with a Kalman
filter, and draws a live fading trail of the ball's path. This is the honest,
working demo for the current wide footage: it tracks ball flight (shots /
outfield). It does NOT do LBW — that needs the camera re-shoot (docs/CAMERA_SPEC.md).

Run:
    # on a clip (windowed live playback)
    python scripts/realtime_ball_demo.py --video training/good_clips/00006.MTS

    # jump to an action window and save an annotated mp4 (no window needed)
    python scripts/realtime_ball_demo.py --video training/good_clips/00006.MTS \
        --start 389 --duration 4 --save demo_out.mp4 --no-display

    # live USB camera
    python scripts/realtime_ball_demo.py --camera 0
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.motion_ball_finder import DEFAULT_ROI, roi_box, gray_crop, fast_blobs
from core.ball_tracker import CricketKalmanFilter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time ball-flight tracking demo")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path, help="video file to stream")
    src.add_argument("--camera", type=int, help="live camera index (e.g. 0)")
    p.add_argument("--roi", type=str, default=None, help='ROI fractions "x1,y1,x2,y2"')
    p.add_argument("--start", type=float, default=0.0, help="start second (video only)")
    p.add_argument("--duration", type=float, default=None, help="seconds to run (video only)")
    p.add_argument("--save", type=Path, default=None, help="write annotated mp4 here")
    p.add_argument("--no-display", action="store_true", help="don't open a window")
    p.add_argument("--trail", type=int, default=18, help="trail length in frames")
    p.add_argument("--gate", type=float, default=60.0, help="Kalman match gate (px)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    roi = DEFAULT_ROI if not args.roi else tuple(float(v) for v in args.roi.split(","))

    cap = cv2.VideoCapture(str(args.video) if args.video else args.camera)
    if not cap.isOpened():
        raise SystemExit("could not open source")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if args.video and args.start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start * fps))
    end_frame = None
    if args.video and args.duration:
        end_frame = int((args.start + args.duration) * fps)

    ok, frame = cap.read()
    if not ok:
        raise SystemExit("empty source")
    box = roi_box(frame.shape[1], frame.shape[0], roi)
    cw, ch = box[2] - box[0], box[3] - box[1]

    writer = None
    if args.save:
        writer = cv2.VideoWriter(str(args.save), cv2.VideoWriter_fourcc(*"mp4v"), fps, (cw, ch))

    g_prev2 = None
    g_prev1 = gray_crop(frame, box)
    kf = CricketKalmanFilter()
    trail: deque = deque(maxlen=args.trail)
    miss = 0
    have_track = False
    fidx = int(args.start * fps) + 1 if args.video else 1
    t0 = time.time()
    n = 0

    while True:
        ok, frame = cap.read()
        if not ok or (end_frame and fidx >= end_frame):
            break
        crop = frame[box[1]:box[3], box[0]:box[2]].copy()
        g = gray_crop(frame, box)

        ball = None
        if g_prev2 is not None:
            blobs = fast_blobs(g_prev2, g_prev1, g)
            if have_track:
                px, py, _, _ = kf.predict()
                best, bd = None, 1e9
                for (bx, by, _a) in blobs:
                    d = np.hypot(bx - px, by - py)
                    if d < bd and d <= args.gate:
                        best, bd = (bx, by), d
                if best is not None:
                    x, y, _, _ = kf.correct(best[0], best[1]); ball = (int(x), int(y)); miss = 0
                else:
                    ball = (int(px), int(py)); miss += 1
                    if miss > 8:
                        have_track = False; trail.clear()
            else:
                # seed on the fastest isolated blob (largest jump from frame centre of mass not needed:
                # pick the blob with the biggest area as a simple seed, then Kalman locks on)
                if blobs:
                    bx, by, _a = max(blobs, key=lambda b: b[2])
                    kf.initialize(bx, by); have_track = True; miss = 0; ball = (bx, by)

        if ball is not None:
            trail.append(ball)

        # draw fading trail
        for i in range(1, len(trail)):
            a = i / len(trail)
            cv2.line(crop, trail[i-1], trail[i], (0, int(255*a), 255), 2)
        if ball is not None:
            cv2.circle(crop, ball, 7, (0, 0, 255), 2)

        n += 1
        live_fps = n / max(time.time() - t0, 1e-6)
        status = "TRACKING" if have_track and miss == 0 else ("COASTING" if have_track else "searching")
        cv2.rectangle(crop, (0, 0), (cw, 28), (0, 0, 0), -1)
        cv2.putText(crop, f"ball-flight demo  {status}  {live_fps:4.1f} fps",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if writer:
            writer.write(crop)
        if not args.no_display:
            cv2.imshow("DRS ball-flight (q to quit)", crop)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        g_prev2, g_prev1 = g_prev1, g
        fidx += 1

    cap.release()
    if writer:
        writer.release()
        print(f"saved {args.save}")
    if not args.no_display:
        cv2.destroyAllWindows()
    print(f"processed {n} frames at {n/max(time.time()-t0,1e-6):.1f} fps")


if __name__ == "__main__":
    main()
