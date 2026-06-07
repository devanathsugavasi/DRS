"""Evaluate a trained DRS YOLO model before using it for decisions."""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path


GATES = {
    "map50": 0.88,
    "map50_95": 0.65,
    "precision": 0.85,
    "recall": 0.82,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLO DRS model quality")
    parser.add_argument("--model", default="models/cricket_ball_yolov8.pt")
    parser.add_argument("--data", default="training/drs_yolo_dataset.yaml")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default=0)
    parser.add_argument("--dataset-source", default="local")
    parser.add_argument("--reports-dir", default="reports")
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
        "ball_recall": float(metrics.box.mr),
        "inference_ms": float(getattr(metrics, "speed", {}).get("inference", 0.0)) if hasattr(metrics, "speed") else None,
    }
    gate_results = {
        name: {
            "value": round(float(summary[source_key]), 6),
            "threshold": threshold,
            "status": "PASS" if float(summary[source_key]) >= threshold else "FAIL",
        }
        for name, threshold in GATES.items()
        for source_key in [name if name != "recall" else "ball_recall"]
    }
    summary["gates"] = gate_results
    summary["decision_ready"] = all(item["status"] == "PASS" for item in gate_results.values())
    out = Path("models/model_evaluation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if out.exists():
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    existing[Path(args.model).name] = summary
    existing[args.model] = summary
    out.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    report = {
        "evaluated_at": datetime.datetime.utcnow().isoformat(),
        "dataset_source": args.dataset_source,
        "dataset_config": args.data,
        "image_count": None,
        "cricket_model": summary,
        "gates_passed": sum(1 for item in gate_results.values() if item["status"] == "PASS"),
        "production_ready": summary["decision_ready"],
    }
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"model_validation_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report: {report_path}")
    if not summary["decision_ready"]:
        raise SystemExit("Model is not accurate enough for reliable DRS testing yet.")


if __name__ == "__main__":
    main()
