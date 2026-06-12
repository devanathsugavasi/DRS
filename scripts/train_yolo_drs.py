"""Train and validate a cricket-ball YOLO model for DRS."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def auto_device() -> str:
    """Use the RTX/CUDA GPU when available, otherwise fall back to CPU."""
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


def metrics_to_dict(metrics: object) -> dict[str, float | str]:
    """Extract stable validation metrics from an Ultralytics metrics object."""
    box = getattr(metrics, "box", None)
    results = {
        "map50": float(getattr(box, "map50", 0.0) or 0.0),
        "map50_95": float(getattr(box, "map", 0.0) or 0.0),
        "precision": float(getattr(box, "mp", 0.0) or 0.0),
        "recall": float(getattr(box, "mr", 0.0) or 0.0),
    }
    save_dir = getattr(metrics, "save_dir", None)
    if save_dir:
        results["validation_dir"] = str(save_dir)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO11l cricket-ball detector for DRS")
    parser.add_argument("--data", default="training/data.yaml", help="YOLO data YAML")
    parser.add_argument("--base-model", default="yolo11l.pt", help="YOLO11l starting model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="auto", help="CUDA device id, cpu, or auto")
    parser.add_argument("--project", default="models/training_runs")
    parser.add_argument("--name", default="cricket_ball_yolo11l")
    parser.add_argument(
        "--export-best",
        default="models/cricket_ball_yolov8.pt",
        help="Destination for best.pt after validation",
    )
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--exist-ok", action="store_true", help="Allow reusing an existing run directory")
    args = parser.parse_args()

    data_path = Path(args.data)
    project_path = Path(args.project)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")
    project_path.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    device = auto_device() if args.device == "auto" else str(args.device)
    print(f"Training device: {device}")
    print(f"Dataset: {data_path}")
    print(f"Base model: {args.base_model}")

    model = YOLO(args.base_model)
    result = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
        patience=args.patience,
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
        workers=args.workers,
        plots=True,
        exist_ok=args.exist_ok,
    )

    best = Path(result.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Training finished but best.pt was not found: {best}")

    print(f"Training complete. Best model: {best}")
    trained_model = YOLO(str(best))
    validation = trained_model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=f"{args.name}_validation",
        plots=True,
        exist_ok=args.exist_ok,
    )

    metrics = metrics_to_dict(validation)
    metrics_path = Path(result.save_dir) / "validation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    destination = Path(args.export_best)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, destination)

    print("Validation complete")
    print(f"mAP50: {metrics['map50']:.4f}")
    print(f"mAP50-95: {metrics['map50_95']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Validation metrics: {metrics_path}")
    print(f"Confusion matrix and plots: {metrics.get('validation_dir', Path(args.project) / (args.name + '_validation'))}")
    print(f"Best model copied to: {destination}")


if __name__ == "__main__":
    main()
