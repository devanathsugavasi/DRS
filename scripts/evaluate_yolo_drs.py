"""Evaluate a trained DRS YOLO model before using it for decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLO DRS model quality")
    parser.add_argument("--model", default="models/cricket_ball_yolov8.pt")
    parser.add_argument("--data", default="training/drs_yolo_dataset.yaml")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default=0)
    parser.add_argument("--min-map50", type=float, default=0.88)
    parser.add_argument("--min-ball-map50", type=float, default=0.92)
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device, split="val")
    summary = {
        "model": args.model,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "decision_ready": float(metrics.box.map50) >= args.min_map50,
    }
    out = Path("models/model_evaluation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["decision_ready"]:
        raise SystemExit("Model is not accurate enough for reliable DRS testing yet.")


if __name__ == "__main__":
    main()
