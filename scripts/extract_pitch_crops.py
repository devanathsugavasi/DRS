"""Extract fixed pitch-zone crops from the fixed-camera match clips.

The match footage is a single wide stand camera, so the pitch sits in a fixed
sub-rectangle of every frame. Cropping to that rectangle makes the ball large
enough (~10-15px) to detect and track, and removes distant fielders that cause
false positives. Output is a YOLO detection dataset (images + labels + review)
ready for manual correction.

The crop ROI is given as fractions of the full frame (auto guess, tuned from
the usable clips). Override with --roi "x1,y1,x2,y2" if needed.

Example
-------
    python scripts/extract_pitch_crops.py \
        --source training/good_clips --output training_crops \
        --frame-stride 15 --max-frames-per-video 250
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2

# Fixed pitch-zone ROI as fractions of the full 1920x1080 frame (auto guess).
DEFAULT_ROI = (0.27, 0.28, 0.73, 0.90)
BALL_CLASS_NAME = "cricket_ball"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract fixed pitch-zone crops + auto labels")
    p.add_argument("--source", type=Path, default=Path("training/good_clips"))
    p.add_argument("--output", type=Path, default=Path("training_crops"))
    p.add_argument("--model", type=Path, default=Path("models/cricket_ball_yolov8.pt"))
    p.add_argument("--roi", type=str, default=None, help='x1,y1,x2,y2 fractions, e.g. "0.27,0.28,0.73,0.90"')
    p.add_argument("--frame-stride", type=int, default=15, help="sample every Nth frame")
    p.add_argument("--max-frames-per-video", type=int, default=250)
    p.add_argument("--train-ratio", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--conf", type=float, default=0.15, help="min confidence to keep an auto label")
    p.add_argument("--uncertain-conf", type=float, default=0.50, help="below this -> copy to review/")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default="cpu")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def prepare_dirs(out: Path, overwrite: bool) -> dict[str, Path]:
    paths = {
        "images_train": out / "images" / "train",
        "images_val": out / "images" / "val",
        "labels_train": out / "labels" / "train",
        "labels_val": out / "labels" / "val",
        "review": out / "review",
    }
    if overwrite:
        for pth in paths.values():
            if pth.exists():
                shutil.rmtree(pth)
    for pth in paths.values():
        pth.mkdir(parents=True, exist_ok=True)
    return paths


def is_ball(box, names: dict) -> bool:
    cid = int(box.cls[0]) if box.cls is not None else -1
    cname = str(names.get(cid, "")).lower()
    return cid == 0 or "ball" in cname


def main() -> None:
    args = parse_args()
    roi = DEFAULT_ROI
    if args.roi:
        roi = tuple(float(v) for v in args.roi.split(","))
        assert len(roi) == 4, "--roi needs 4 comma-separated fractions"

    videos = sorted(args.source.glob("*.MTS")) + sorted(args.source.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"no videos in {args.source}")
    paths = prepare_dirs(args.output, args.overwrite)

    from ultralytics import YOLO
    print("loading model (cold start can take minutes)...", flush=True)
    model = YOLO(str(args.model))
    print(f"loaded. ROI fractions={roi}", flush=True)

    rng = random.Random(args.seed)
    stats = {"extracted": 0, "with_ball": 0, "review": 0, "train": 0, "val": 0}

    for vp in videos:
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            print(f"WARN open fail {vp.name}", flush=True)
            continue
        fidx = taken = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if fidx % args.frame_stride == 0:
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = int(roi[0]*w), int(roi[1]*h), int(roi[2]*w), int(roi[3]*h)
                crop = frame[y1:y2, x1:x2]
                split = "val" if rng.random() >= args.train_ratio else "train"
                stem = f"{vp.stem}_f{fidx:06d}"
                img_path = paths[f"images_{split}"] / f"{stem}.jpg"
                lbl_path = paths[f"labels_{split}"] / f"{stem}.txt"

                r = model.predict(crop, imgsz=args.imgsz, conf=args.conf,
                                  device=args.device, verbose=False)[0]
                names = getattr(r, "names", {}) or {}
                labels, best = [], 0.0
                for b in r.boxes:
                    if not is_ball(b, names):
                        continue
                    conf = float(b.conf[0])
                    xc, yc, bw, bh = [float(v) for v in b.xywhn[0]]
                    labels.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                    best = max(best, conf)

                cv2.imwrite(str(img_path), crop)
                lbl_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")

                stats["extracted"] += 1
                stats[split] += 1
                if labels:
                    stats["with_ball"] += 1
                if not labels or best < args.uncertain_conf:
                    tag = "no_ball" if not labels else f"conf{best:.2f}"
                    shutil.copy2(img_path, paths["review"] / f"{stem}_{tag}.jpg")
                    stats["review"] += 1

                taken += 1
                if stats["extracted"] % 100 == 0:
                    print(f"  {stats['extracted']} crops | ball {stats['with_ball']} | review {stats['review']}", flush=True)
                if taken >= args.max_frames_per_video:
                    break
            fidx += 1
        cap.release()
        print(f"done {vp.name}: {taken} crops", flush=True)

    yaml = args.output / "data.yaml"
    yaml.write_text("\n".join([
        f"path: {args.output.resolve().as_posix()}",
        "train: images/train", "val: images/val", "",
        "nc: 1", "names:", f"  0: {BALL_CLASS_NAME}", "",
    ]), encoding="utf-8")
    (args.output / "extract_stats.json").write_text(
        json.dumps({**stats, "roi": roi}, indent=2), encoding="utf-8")

    print(f"\nDONE. {stats['extracted']} crops "
          f"(train {stats['train']}/val {stats['val']}), "
          f"ball auto-labeled on {stats['with_ball']}, review {stats['review']}")
    print(f"dataset: {args.output}  yaml: {yaml}")


if __name__ == "__main__":
    main()
