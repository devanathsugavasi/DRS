"""Prepare YOLO cricket-ball training data from MTS match footage.

The script extracts frames from AVCHD .MTS videos, pre-annotates cricket balls
with an existing cricket model when available, falls back to COCO sports-ball
detections, and writes YOLO-format labels plus review assets.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2


DEFAULT_SOURCE = Path(r"E:\PRIVATE\AVCHD\BDMV\STREAM")
DEFAULT_OUTPUT = Path("training")
DEFAULT_CRICKET_MODEL = Path("models/cricket_ball_yolov8.pt")
DEFAULT_FALLBACK_MODEL = Path("yolo11l.pt")
BALL_CLASS_NAME = "cricket_ball"
COCO_SPORTS_BALL_CLASS_ID = 32


@dataclass(slots=True)
class DatasetStats:
    total_videos: int = 0
    videos_processed: int = 0
    total_frames_seen: int = 0
    total_frames_extracted: int = 0
    train_frames: int = 0
    val_frames: int = 0
    frames_with_ball_detected: int = 0
    frames_requiring_manual_review: int = 0
    labels_written: int = 0
    review_images_written: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare YOLO cricket-ball data from .MTS videos")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Folder containing .MTS videos")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Training output folder")
    parser.add_argument("--frame-stride", type=int, default=5, help="Extract every Nth frame")
    parser.add_argument("--train-ratio", type=float, default=0.82, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic train/val split seed")
    parser.add_argument("--conf", type=float, default=0.18, help="Minimum detector confidence to write a label")
    parser.add_argument("--uncertain-conf", type=float, default=0.40, help="Below this confidence, copy frame to review")
    parser.add_argument("--model", type=Path, default=None, help="Optional model path for pre-annotation")
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. 0 or cpu. Defaults to auto.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    parser.add_argument("--image-ext", default="jpg", choices=("jpg", "png"), help="Extracted image extension")
    parser.add_argument("--max-videos", type=int, default=None, help="Optional cap for smoke tests")
    parser.add_argument("--max-frames-per-video", type=int, default=None, help="Optional cap for smoke tests")
    parser.add_argument("--overwrite", action="store_true", help="Clear generated images, labels, and review folders first")
    return parser.parse_args()


def find_videos(source: Path, max_videos: int | None) -> list[Path]:
    if not source.exists():
        raise FileNotFoundError(f"Video source folder not found: {source}")
    videos = sorted(source.glob("*.MTS"))
    if max_videos is not None:
        videos = videos[: max(0, max_videos)]
    if not videos:
        raise FileNotFoundError(f"No .MTS videos found in: {source}")
    return videos


def count_video_frames(videos: Iterable[Path]) -> int:
    total = 0
    for video_path in videos:
        capture = cv2.VideoCapture(str(video_path))
        if capture.isOpened():
            total += int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        capture.release()
    return total


def prepare_directories(output: Path, overwrite: bool) -> dict[str, Path]:
    paths = {
        "images_train": output / "images" / "train",
        "images_val": output / "images" / "val",
        "labels_train": output / "labels" / "train",
        "labels_val": output / "labels" / "val",
        "review": output / "review",
    }
    if overwrite:
        for path in paths.values():
            if path.exists():
                shutil.rmtree(path)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def resolve_model_path(requested: Path | None) -> tuple[Path, bool]:
    """Return the pre-annotation model path and whether it is cricket-specific."""
    candidates = []
    if requested is not None:
        candidates.append(requested)
    candidates.extend(
        [
            DEFAULT_CRICKET_MODEL,
            Path("models/best.pt"),
            DEFAULT_FALLBACK_MODEL,
            Path("yolo11n.pt"),
            Path("yolov8n.pt"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate, candidate.name in {"cricket_ball_yolov8.pt", "best.pt"} and candidate.parent.name == "models"
    return DEFAULT_FALLBACK_MODEL, False


def load_yolo_model(model_path: Path):
    from ultralytics import YOLO

    return YOLO(str(model_path))


def should_use_val(video_index: int, frame_index: int, train_ratio: float, seed: int) -> bool:
    rng = random.Random(f"{seed}:{video_index}:{frame_index}")
    return rng.random() >= train_ratio


def detections_to_yolo_labels(result, cricket_model: bool, conf_threshold: float) -> tuple[list[str], float]:
    """Convert one Ultralytics result into YOLO labels for class 0 cricket_ball."""
    labels: list[str] = []
    best_conf = 0.0
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return labels, best_conf

    names = getattr(result, "names", {}) or {}
    for box in boxes:
        confidence = float(box.conf[0]) if box.conf is not None else 0.0
        if confidence < conf_threshold:
            continue
        class_id = int(box.cls[0]) if box.cls is not None else -1
        class_name = str(names.get(class_id, "")).lower()
        if cricket_model:
            is_ball = class_id == 0 or "ball" in class_name
        else:
            is_ball = class_id == COCO_SPORTS_BALL_CLASS_ID or class_name in {"sports ball", "ball"}
        if not is_ball:
            continue

        x_center, y_center, width, height = [float(value) for value in box.xywhn[0]]
        labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        best_conf = max(best_conf, confidence)
    return labels, best_conf


def write_data_yaml(output: Path) -> Path:
    yaml_path = output / "data.yaml"
    yaml_text = "\n".join(
        [
            f"path: {output.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "",
            "nc: 1",
            "names:",
            f"  0: {BALL_CLASS_NAME}",
            "",
        ]
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


def write_stats(output: Path, stats: DatasetStats, model_path: Path, cricket_model: bool, data_yaml: Path) -> Path:
    stats_path = output / "dataset_stats.json"
    payload = {
        **asdict(stats),
        "model_used": str(model_path),
        "model_type": "cricket" if cricket_model else "coco_sports_ball_fallback",
        "data_yaml": str(data_yaml),
        "manual_review_folder": str((output / "review").resolve()),
    }
    stats_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return stats_path


def iter_extracted_frames(
    videos: Iterable[Path],
    frame_stride: int,
    max_frames_per_video: int | None,
) -> Iterable[tuple[int, Path, int, object]]:
    for video_index, video_path in enumerate(videos):
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            print(f"WARNING: could not open video: {video_path}")
            continue
        frame_index = 0
        extracted_for_video = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_stride == 0:
                yield video_index, video_path, frame_index, frame
                extracted_for_video += 1
                if max_frames_per_video is not None and extracted_for_video >= max_frames_per_video:
                    break
            frame_index += 1
        capture.release()


def main() -> None:
    args = parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1")

    videos = find_videos(args.source, args.max_videos)
    paths = prepare_directories(args.output, args.overwrite)
    model_path, cricket_model = resolve_model_path(args.model)
    model = load_yolo_model(model_path)

    stats = DatasetStats(
        total_videos=len(sorted(args.source.glob("*.MTS"))),
        videos_processed=len(videos),
        total_frames_seen=count_video_frames(videos),
    )
    print(f"Preparing data from {len(videos)} video(s)")
    print(f"Pre-annotation model: {model_path} ({'cricket' if cricket_model else 'COCO sports-ball fallback'})")

    for video_index, video_path, frame_index, frame in iter_extracted_frames(
        videos,
        args.frame_stride,
        args.max_frames_per_video,
    ):
        split = "val" if should_use_val(video_index, frame_index, args.train_ratio, args.seed) else "train"
        stem = f"{video_path.stem}_f{frame_index:06d}"
        image_path = paths[f"images_{split}"] / f"{stem}.{args.image_ext}"
        label_path = paths[f"labels_{split}"] / f"{stem}.txt"

        result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
        labels, best_conf = detections_to_yolo_labels(result, cricket_model, args.conf)

        cv2.imwrite(str(image_path), frame)
        label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")

        stats.total_frames_extracted += 1
        if split == "train":
            stats.train_frames += 1
        else:
            stats.val_frames += 1
        if labels:
            stats.frames_with_ball_detected += 1
            stats.labels_written += len(labels)

        needs_review = not labels or best_conf < args.uncertain_conf
        if needs_review:
            review_name = f"{stem}_{'no_ball' if not labels else f'conf_{best_conf:.2f}'}.{args.image_ext}"
            shutil.copy2(image_path, paths["review"] / review_name)
            stats.frames_requiring_manual_review += 1
            stats.review_images_written += 1

        if stats.total_frames_extracted % 250 == 0:
            print(
                f"Extracted {stats.total_frames_extracted} frames | "
                f"ball labels on {stats.frames_with_ball_detected} | "
                f"review {stats.frames_requiring_manual_review}"
            )

    data_yaml = write_data_yaml(args.output)
    stats_path = write_stats(args.output, stats, model_path, cricket_model, data_yaml)

    print("\nDataset preparation complete")
    print(f"Total videos: {stats.total_videos}")
    print(f"Videos processed: {stats.videos_processed}")
    print(f"Total frames extracted: {stats.total_frames_extracted}")
    print(f"Frames with ball detected: {stats.frames_with_ball_detected}")
    print(f"Frames requiring manual review: {stats.frames_requiring_manual_review}")
    print(f"YOLO data YAML: {data_yaml}")
    print(f"Stats JSON: {stats_path}")
    print(f"Review folder: {args.output / 'review'}")


if __name__ == "__main__":
    main()
