"""Motion-based cricket-ball finder for a fixed wide camera.

The match camera is fixed, so the ball is the FASTEST small object in frame
during a delivery. We isolate it with 3-frame differencing (catches things that
move a lot between consecutive frames), filter blobs by size, and report
candidates. No training / labels needed.

Two modes:

  scan   - sweep a whole clip cheaply, log per-second motion-candidate counts so
           you can find which seconds contain a delivery.
             python scripts/motion_ball_finder.py scan --video CLIP.MTS

  track  - full-rate pass over a time window, draw ball candidates, write an
           annotated mp4 + a candidates json.
             python scripts/motion_ball_finder.py track --video CLIP.MTS \
                 --start 120 --duration 30 --out preview_track
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# Fixed pitch-zone ROI (fractions of full frame) - same as extract_pitch_crops.
DEFAULT_ROI = (0.27, 0.28, 0.73, 0.90)
# Ball blob area in the CROP, in px^2. Tune per footage.
MIN_AREA, MAX_AREA = 3, 350


def roi_box(w: int, h: int, roi) -> tuple[int, int, int, int]:
    return int(roi[0]*w), int(roi[1]*h), int(roi[2]*w), int(roi[3]*h)


def fast_blobs(prev_g, cur_g, next_g) -> list[tuple[int, int, float]]:
    """3-frame differencing -> small fast blobs. Returns [(cx, cy, area)]."""
    d1 = cv2.absdiff(prev_g, cur_g)
    d2 = cv2.absdiff(cur_g, next_g)
    mask = cv2.bitwise_and(d1, d2)
    mask = cv2.threshold(mask, 18, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if not (MIN_AREA <= a <= MAX_AREA):
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        ar = bw / max(bh, 1)
        if ar < 0.3 or ar > 3.0:        # roughly compact
            continue
        out.append((x + bw // 2, y + bh // 2, a))
    return out


def gray_crop(frame, box):
    x1, y1, x2, y2 = box
    g = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(g, (3, 3), 0)


def scan(args, roi):
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok, f = cap.read()
    if not ok:
        raise SystemExit("cannot read video")
    box = roi_box(f.shape[1], f.shape[0], roi)
    prev = gray_crop(f, box)
    buf, sec_hist = [prev], {}
    fidx = 1
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if fidx % args.stride == 0:
            g = gray_crop(f, box)
            buf.append(g)
            if len(buf) >= 3:
                n = len(fast_blobs(buf[-3], buf[-2], buf[-1]))
                s = int(fidx / fps)
                sec_hist[s] = sec_hist.get(s, 0) + n
        fidx += 1
    cap.release()
    # report the top seconds by candidate count
    ranked = sorted(sec_hist.items(), key=lambda kv: kv[1], reverse=True)
    print(f"scanned {fidx} frames, {total} total, fps={fps:.0f}")
    print("top active seconds (sec: candidate count) - likely deliveries/action:")
    for s, n in ranked[:25]:
        print(f"  {s:5d}s : {n}")


def track(args, roi):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
    start_f = int(args.start * fps)
    end_f = start_f + int(args.duration * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    ok, f = cap.read()
    if not ok:
        raise SystemExit("cannot read window")
    H, W = f.shape[:2]
    box = roi_box(W, H, roi)
    cw, ch = box[2]-box[0], box[3]-box[1]
    writer = cv2.VideoWriter(str(out / "track.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (cw, ch))
    frames = [f]
    fidx = start_f + 1
    while fidx < end_f:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
        fidx += 1
    cap.release()

    grays = [gray_crop(f, box) for f in frames]
    candidates = []
    for i in range(1, len(frames) - 1):
        crop = frames[i][box[1]:box[3], box[0]:box[2]].copy()
        blobs = fast_blobs(grays[i-1], grays[i], grays[i+1])
        for (cx, cy, a) in blobs:
            cv2.circle(crop, (cx, cy), 9, (0, 0, 255), 2)
        if blobs:
            candidates.append({"frame": start_f + i,
                               "t": round((start_f + i) / fps, 3),
                               "blobs": [[int(x), int(y), float(ar)] for x, y, ar in blobs]})
        cv2.putText(crop, f"t={(start_f+i)/fps:.2f}s  cand={len(blobs)}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        writer.write(crop)
    writer.release()
    (out / "candidates.json").write_text(json.dumps(candidates, indent=2))
    print(f"wrote {out/'track.mp4'} ({len(frames)} frames)")
    print(f"frames with candidates: {len(candidates)}")
    print(f"candidates json: {out/'candidates.json'}")


def _eval_tracklet(pts):
    """pts = [(frame, x, y)]. Return (is_ball, net_disp, straightness, speed)."""
    if len(pts) < 6:
        return False, 0.0, 0.0, 0.0
    xs = np.array([p[1] for p in pts], float)
    ys = np.array([p[2] for p in pts], float)
    seg = np.hypot(np.diff(xs), np.diff(ys))
    path_len = float(seg.sum())
    net = float(np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))
    if path_len < 1e-3:
        return False, 0.0, 0.0, 0.0
    straightness = net / path_len            # 1.0 = perfectly straight
    nframes = pts[-1][0] - pts[0][0]
    speed = path_len / max(nframes, 1)       # px per frame
    is_ball = net >= 60 and straightness >= 0.80 and speed >= 6.0 and seg.max() <= 120
    return is_ball, net, straightness, speed


def find(args, roi):
    """Stream a whole clip, link fast blobs into tracklets, keep ball-like arcs."""
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
    ok, f = cap.read()
    if not ok:
        raise SystemExit("cannot read video")
    box = roi_box(f.shape[1], f.shape[0], roi)
    cw = box[2] - box[0]                      # crop width in px
    # pitch corridor = central x-band of the crop (where the stumps/pitch line are)
    corr = tuple(float(v) for v in args.corridor.split(",")) if args.corridor else (0.32, 0.62)
    corr_lo, corr_hi = corr[0] * cw, corr[1] * cw
    g_prev2 = None
    g_prev1 = gray_crop(f, box)
    fidx = 1
    active = []          # each: {"pts":[(frame,x,y)], "miss":int}
    found = []
    MAX_JUMP, MAX_MISS = 70, 3
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = gray_crop(f, box)
        if g_prev2 is not None:
            blobs = fast_blobs(g_prev2, g_prev1, g)
            used = set()
            for tr in active:
                fr, lx, ly = tr["pts"][-1]
                best, bd = None, 1e9
                for j, (bx, by, _a) in enumerate(blobs):
                    if j in used:
                        continue
                    d = np.hypot(bx - lx, by - ly)
                    if d < bd and d <= MAX_JUMP:
                        best, bd = j, d
                if best is None:
                    tr["miss"] += 1
                else:
                    used.add(best)
                    bx, by, _a = blobs[best]
                    tr["pts"].append((fidx, bx, by))
                    tr["miss"] = 0
            for j, (bx, by, _a) in enumerate(blobs):
                if j not in used:
                    active.append({"pts": [(fidx, bx, by)], "miss": 0})
            still = []
            for tr in active:
                if tr["miss"] > MAX_MISS:
                    ib, net, st, sp = _eval_tracklet(tr["pts"])
                    if ib:
                        p = tr["pts"]
                        sx, sy = p[0][1], p[0][2]
                        ex, ey = p[-1][1], p[-1][2]
                        dx, dy = ex - sx, ey - sy
                        mid_x = float(np.median([q[1] for q in p]))
                        # delivery = vertical-dominant + inside the pitch corridor +
                        # ball-like speed/length (rejects edge player-walks & long tracks)
                        vertical = abs(dy) > 1.2 * abs(dx)
                        in_corr = corr_lo <= mid_x <= corr_hi
                        ball_like = 6 <= len(p) <= 20 and sp >= 12.0 and st >= 0.85
                        if vertical and in_corr and ball_like:
                            kind = "delivery"
                        elif vertical:
                            kind = "vertical"
                        else:
                            kind = "across"
                        found.append({"t_start": round(p[0][0]/fps, 2),
                                      "t_end": round(p[-1][0]/fps, 2),
                                      "frames": [p[0][0], p[-1][0]],
                                      "len": len(p), "net_px": round(net, 1),
                                      "straightness": round(st, 2), "speed_px": round(sp, 1),
                                      "start": [int(sx), int(sy)], "end": [int(ex), int(ey)],
                                      "dx": int(dx), "dy": int(dy), "mid_x": int(mid_x),
                                      "kind": kind,
                                      "pts": [[fr, int(x), int(y)] for fr, x, y in p]})
                else:
                    still.append(tr)
            active = still
        g_prev2, g_prev1 = g_prev1, g
        fidx += 1
    cap.release()
    found.sort(key=lambda d: d["net_px"], reverse=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "deliveries.json").write_text(json.dumps(found, indent=2))
    deliveries = [d for d in found if d["kind"] == "delivery"]
    print(f"scanned {fidx} frames @ {fps:.0f}fps")
    print(f"ball-like arcs: {len(found)}  (in-corridor deliveries: {len(deliveries)})")
    print("top deliveries (vertical + pitch corridor + ball-like speed):")
    for d in deliveries[:20]:
        print(f"  t={d['t_start']:6.1f}-{d['t_end']:.1f}s  len={d['len']:2d}  "
              f"net={d['net_px']:5.0f}px  straight={d['straightness']:.2f}  "
              f"spd={d['speed_px']:.1f}  dy={d['dy']:+5d} dx={d['dx']:+5d}")
    print(f"\nfull list: {out/'deliveries.json'}")


def export(args, roi):
    """Track the dominant ball arc in a window and write pipeline-format tracking JSON.

    Output matches test_delivery.tracking.json: a list of
    {frame, x, y, confidence, speed_kmh}. This makes the motion tracker a
    drop-in ball source for the existing DRS pipeline. Best on open-field arcs;
    delivery-zone tracking is limited by the footage (see docs/MASTER_PLAN.md).
    """
    import sys
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from core.ball_tracker import CricketKalmanFilter

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
    start_f = int(args.start * fps)
    end_f = start_f + int(args.duration * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    ok, f = cap.read()
    if not ok:
        raise SystemExit("cannot read window")
    box = roi_box(f.shape[1], f.shape[0], roi)
    frames = [f]
    fidx = start_f + 1
    while fidx < end_f:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
        fidx += 1
    cap.release()
    grays = [gray_crop(fr, box) for fr in frames]
    cand = [[] for _ in frames]
    for i in range(1, len(frames) - 1):
        cand[i] = [(x, y) for (x, y, a) in fast_blobs(grays[i-1], grays[i], grays[i+1])]

    # greedy link to find the best seed tracklet in the window
    active, tracks = [], []
    for i in range(1, len(frames) - 1):
        used = set()
        for tr in active:
            lx, ly = tr[-1][1], tr[-1][2]
            best, bd = None, 1e9
            for j, (cx, cy) in enumerate(cand[i]):
                if j in used:
                    continue
                d = np.hypot(cx - lx, cy - ly)
                if d < bd and d <= 70:
                    best, bd = j, d
            if best is None:
                tr.append((i, tr[-1][1], tr[-1][2], True))
            else:
                used.add(best); cx, cy = cand[i][best]; tr.append((i, cx, cy, False))
        for j, (cx, cy) in enumerate(cand[i]):
            if j not in used:
                active.append([(i, cx, cy, False)])
        keep = []
        for tr in active:
            if sum(1 for p in tr[-4:] if p[3]) > 3:
                tracks.append(tr)
            else:
                keep.append(tr)
        active = keep
    tracks += active

    def net(tr):
        pts = [(p[1], p[2]) for p in tr if not p[3]]
        return 0 if len(pts) < 6 else np.hypot(pts[-1][0]-pts[0][0], pts[-1][1]-pts[0][1])
    seed = max(tracks, key=net) if tracks else None
    if seed is None or net(seed) < 40:
        raise SystemExit("no ball-like arc found in window")
    spts = [(p[0], p[1], p[2]) for p in seed if not p[3]]

    # Kalman-stitch from the seed, coasting through gaps
    kf = CricketKalmanFilter(); kf.initialize(spts[0][1], spts[0][2])
    kf.filter.statePost = np.array([[spts[0][1]], [spts[0][2]],
                                    [spts[1][1]-spts[0][1]], [spts[1][2]-spts[0][2]]], dtype=np.float32)
    kf.filter.statePre = kf.filter.statePost.copy()
    px_per_m = args.px_per_m
    out_pts, coast = [], 0
    for i in range(spts[0][0], len(frames)):
        pxp, pyp, vx, vy = kf.predict()
        best, bd = None, 1e9
        for (cx, cy) in cand[i]:
            d = np.hypot(cx - pxp, cy - pyp)
            if d < bd and d <= 55:
                best, bd = (cx, cy), d
        if best is not None:
            x, y, vx, vy = kf.correct(best[0], best[1]); coast = 0; conf = 1.0
        else:
            x, y = pxp, pyp; coast += 1; conf = round(max(0.2, 0.9 - 0.1*coast), 2)
        speed_px = float(np.hypot(vx, vy)) * fps
        speed_kmh = round(speed_px / px_per_m * 3.6, 2) if px_per_m else 0.0
        out_pts.append({"frame": start_f + i, "x": int(round(x)), "y": int(round(y)),
                        "confidence": conf, "speed_kmh": speed_kmh})
        if coast > 10:
            break

    out = Path(args.out)
    payload = {"source": str(args.video), "roi": list(roi), "fps": fps,
               "calibrated": bool(px_per_m), "px_per_m": px_per_m,
               "window": [args.start, args.start + args.duration],
               "tracking": out_pts}
    out.write_text(json.dumps(payload, indent=2))
    measured = sum(1 for p in out_pts if p["confidence"] >= 1.0)
    print(f"exported {len(out_pts)} track points ({measured} measured) -> {out}")
    print(f"window {args.start}-{args.start+args.duration}s, speed_kmh "
          f"{'computed' if px_per_m else 'left 0 (no --px-per-m calibration)'}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("scan"); s.add_argument("--video", type=Path, required=True)
    s.add_argument("--stride", type=int, default=2)
    t = sub.add_parser("track"); t.add_argument("--video", type=Path, required=True)
    t.add_argument("--start", type=float, required=True)
    t.add_argument("--duration", type=float, default=20.0)
    t.add_argument("--out", type=Path, default=Path("preview_track"))
    fd = sub.add_parser("find"); fd.add_argument("--video", type=Path, required=True)
    fd.add_argument("--out", type=Path, default=Path("deliveries_out"))
    fd.add_argument("--corridor", type=str, default=None,
                    help='pitch corridor x-band as crop-width fractions, e.g. "0.32,0.62"')
    ex = sub.add_parser("export"); ex.add_argument("--video", type=Path, required=True)
    ex.add_argument("--start", type=float, required=True)
    ex.add_argument("--duration", type=float, default=5.0)
    ex.add_argument("--out", type=Path, default=Path("ball_track.json"))
    ex.add_argument("--px-per-m", type=float, default=0.0,
                    help="pixels per metre in the crop (from calibration) to fill speed_kmh")
    for p in (s, t, fd, ex):
        p.add_argument("--roi", type=str, default=None)
    args = ap.parse_args()
    roi = DEFAULT_ROI if not args.roi else tuple(float(v) for v in args.roi.split(","))
    if args.mode == "scan":
        scan(args, roi)
    elif args.mode == "find":
        find(args, roi)
    elif args.mode == "export":
        export(args, roi)
    else:
        track(args, roi)


if __name__ == "__main__":
    main()
