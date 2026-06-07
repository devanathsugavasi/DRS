#!/usr/bin/env python
"""Bootstrap public cricket DRS baseline assets.

The script downloads a cricket-ball detector that can make the project run now.
It also writes an explicit readiness record showing that public weights still need
local validation before the LBW engine may show final OUT/NOT OUT decisions.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path


HF_CRICKET_BALL_MODEL_URL = "https://huggingface.co/ashishgimekar/cricket-ball-yolo/resolve/main/best.pt"
HF_CRICKET_BALL_MODEL_PAGE = "https://huggingface.co/ashishgimekar/cricket-ball-yolo"
KAGGLE_CRICKET_BALL_DATASET = "kushagra3204/cricket-ball-dataset-for-yolo"
KAGGLE_CRICKET_BALL_PAGE = "https://www.kaggle.com/datasets/kushagra3204/cricket-ball-dataset-for-yolo"
ROBOFLOW_TRACKING_PAGE = "https://universe.roboflow.com/cricket-ball-tracking-dataset"


@dataclass(frozen=True)
class PublicAsset:
    name: str
    source: str
    destination: str
    status: str
    note: str


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    temporary.replace(destination)


def _write_readiness(model_path: Path, force: bool) -> Path:
    metrics_path = Path("models/model_evaluation.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, object] = {}
    if metrics_path.exists() and not force:
        try:
            existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}

    existing[model_path.name] = {
        "source": HF_CRICKET_BALL_MODEL_PAGE,
        "map50": None,
        "map50_95": None,
        "ball_recall": None,
        "precision": None,
        "inference_ms": None,
        "usable": False,
        "reason": (
            "Public cricket-ball weights are installed for working demos, but local validation metrics "
            "are not available yet. Run scripts/evaluate_yolo_drs.py on a held-out cricket dataset."
        ),
    }
    metrics_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return metrics_path


def _write_manifest(model_path: Path, metrics_path: Path) -> Path:
    manifest_path = Path("models/public_assets_manifest.json")
    assets = [
        PublicAsset(
            name="Cricket ball YOLOv8 public baseline",
            source=HF_CRICKET_BALL_MODEL_PAGE,
            destination=str(model_path),
            status="installed" if model_path.exists() else "missing",
            note="Single-class cricket ball detector for immediate pipeline testing; not enough by itself for LBW decisions.",
        ),
        PublicAsset(
            name="Cricket Ball Dataset for YOLO",
            source=KAGGLE_CRICKET_BALL_PAGE,
            destination="data/datasets/cricket_ball_data",
            status="manual_download_required",
            note="1778 annotated YOLOv8 images, CC0/Public Domain. Use Kaggle credentials to download.",
        ),
        PublicAsset(
            name="Roboflow Cricket Ball Tracking DATASET",
            source=ROBOFLOW_TRACKING_PAGE,
            destination="data/datasets/roboflow_cricket_ball_tracking",
            status="manual_download_required",
            note="Large public tracking/detection dataset family; verify license/API key before training.",
        ),
    ]
    manifest_path.write_text(
        json.dumps(
            {
                "readiness_metrics": str(metrics_path),
                "assets": [asdict(asset) for asset in assets],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Install public cricket DRS baseline assets")
    parser.add_argument("--model-destination", default="models/cricket_ball_yolov8.pt")
    parser.add_argument("--force", action="store_true", help="Re-download and overwrite existing public baseline model")
    parser.add_argument("--skip-model", action="store_true", help="Only write dataset/model manifest and readiness metadata")
    args = parser.parse_args()

    model_path = Path(args.model_destination)
    if not args.skip_model:
        if model_path.exists() and not args.force:
            print(f"Model already exists: {model_path}")
        else:
            print(f"Downloading public cricket ball detector from {HF_CRICKET_BALL_MODEL_PAGE}")
            _download(HF_CRICKET_BALL_MODEL_URL, model_path)
            print(f"Installed model: {model_path}")

    metrics_path = _write_readiness(model_path, args.force)
    manifest_path = _write_manifest(model_path, metrics_path)
    print(f"Wrote readiness metadata: {metrics_path}")
    print(f"Wrote public asset manifest: {manifest_path}")
    print()
    print("Optional dataset commands:")
    print(f"  kaggle datasets download -d {KAGGLE_CRICKET_BALL_DATASET} -p data/datasets --unzip")
    print("  Roboflow download requires an API key; see docs/PUBLIC_BASELINES.md")
    print()
    print("Important: installed weights enable demos, not final LBW decisions. Run local evaluation first.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
