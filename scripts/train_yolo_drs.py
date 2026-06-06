"""Train a cricket-specific YOLOv8 model for DRS objects."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 for cricket DRS objects")
    parser.add_argument("--data", default="training/drs_yolo_dataset.yaml", help="YOLO data YAML")
    parser.add_argument("--base-model", default="yolo11l.pt", help="YOLO starting model; use local YOLO11x/YOLO11l when possible")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=0, help="CUDA device id or cpu")
    parser.add_argument("--project", default="models/training_runs")
    parser.add_argument("--name", default="drs_yolov8")
    parser.add_argument(
        "--export-best",
        default="",
        help="Optional destination for best.pt after training, for example models/cricket_ball_yolov8.pt",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    project_path = Path(args.project)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")
    project_path.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    model = YOLO(args.base_model)
    result = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=30,
        cos_lr=True,
        close_mosaic=15,
        hsv_h=0.015,
        hsv_s=0.55,
        hsv_v=0.35,
        degrees=2.0,
        translate=0.04,
        scale=0.35,
        shear=1.0,
        fliplr=0.5,
        mosaic=0.65,
        mixup=0.05,
        cache=False,
    )

    best = Path(result.save_dir) / "weights" / "best.pt"
    print(f"Training complete. Best model: {best}")
    if args.export_best:
        import shutil

        destination = Path(args.export_best)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, destination)
        print(f"Best model copied to: {destination}")
    else:
        print("Copy the best model to models/cricket_ball_yolov8.pt after validation.")


if __name__ == "__main__":
    main()
