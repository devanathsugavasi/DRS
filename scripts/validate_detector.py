"""Validate cricket-ball detector performance on real video footage."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import cv2
import numpy as np
from ultralytics import YOLO

from utils.inference_device import resolve_device, run_with_cpu_fallback


BALL_CLASS_ID = 0


@dataclass(slots=True)
class VideoMetrics:
    video: str
    duration_seconds: float
    frames_sampled: int
    detection_rate: float
    avg_confidence: float
    max_confidence: float
    min_confidence: float
    false_positive_estimate: float
    tracking_continuity: float
    verdict: str
    verdict_reason: str
    recommendation: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate YOLO cricket-ball detection on video footage")
    parser.add_argument("--video", type=Path, help="Single .MTS/.mp4 video file")
    parser.add_argument("--video-dir", type=Path, help="Folder of videos to process")
    parser.add_argument("--model", type=Path, default=Path("models/best.pt"), help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation_results"), help="Output directory")
    parser.add_argument("--sample-rate", type=int, default=3, help="Process every N frames")
    parser.add_argument("--save-frames", action="store_true", help="Save annotated detected and worst frames")
    parser.add_argument("--device", type=str, default="cpu", help="Inference device: 'cpu' or 'cuda' (default: cpu)")
    parser.add_argument("--synthetic", action="store_true", help="Use test videos from data/test_videos/ if no video is specified")
    return parser.parse_args()


def find_videos(video: Path | None, video_dir: Path | None, synthetic: bool = False) -> list[Path]:
    if video:
        if not video.exists():
            raise FileNotFoundError(video)
        return [video]
    if video_dir and video_dir.exists():
        extensions = {".mts", ".mp4", ".avi", ".mov"}
        videos = sorted(path for path in video_dir.iterdir() if path.suffix.lower() in extensions)
        if videos:
            return videos
    if synthetic:
        test_video_dir = Path("data/test_videos")
        if test_video_dir.exists():
            extensions = {".mts", ".mp4", ".avi", ".mov"}
            videos = sorted(path for path in test_video_dir.iterdir() if path.suffix.lower() in extensions)
            if videos:
                print(f"[synthetic] Using {len(videos)} test video(s) from {test_video_dir}")
                return videos
        raise FileNotFoundError(f"No test videos found in {test_video_dir}")
    raise ValueError("Provide --video, an existing --video-dir, or --synthetic")


def extract_ball_detections(result, frame_shape: tuple[int, int, int]) -> list[dict[str, float]]:
    detections: list[dict[str, float]] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections
    h, w = frame_shape[:2]
    for box in boxes:
        class_id = int(box.cls[0]) if box.cls is not None else -1
        if class_id != BALL_CLASS_ID:
            continue
        confidence = float(box.conf[0]) if box.conf is not None else 0.0
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0]]
        detections.append(
            {
                "confidence": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cx": (x1 + x2) / 2.0,
                "cy": (y1 + y2) / 2.0,
                "width": max(1.0, x2 - x1),
                "height": max(1.0, y2 - y1),
                "frame_width": float(w),
                "frame_height": float(h),
            }
        )
    return detections


def is_implausible_detection(detection: dict[str, float]) -> bool:
    width = detection["frame_width"]
    height = detection["frame_height"]
    cx = detection["cx"]
    cy = detection["cy"]
    box_area = detection["width"] * detection["height"]
    frame_area = width * height
    outside_pitch_band = cy < height * 0.05 or cy > height * 0.95 or cx < width * 0.02 or cx > width * 0.98
    too_large = box_area > frame_area * 0.02
    too_tiny = box_area < 4.0
    return outside_pitch_band or too_large or too_tiny


def draw_detections(frame: np.ndarray, detections: list[dict[str, float]], frame_index: int) -> np.ndarray:
    annotated = frame.copy()
    for detection in detections:
        start = int(detection["x1"]), int(detection["y1"])
        end = int(detection["x2"]), int(detection["y2"])
        cv2.rectangle(annotated, start, end, (0, 220, 255), 2)
        cv2.putText(
            annotated,
            f"ball {detection['confidence']:.2f}",
            (start[0], max(20, start[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(annotated, f"frame {frame_index}", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return annotated


def verdict_for(detection_rate: float, avg_confidence: float, continuity: float) -> tuple[str, str, str]:
    if detection_rate >= 0.65 and avg_confidence >= 0.55 and continuity >= 0.75:
        return "GOOD", "Model is ready for live testing.", "Proceed to calibrated pitch validation."
    if detection_rate >= 0.45 or avg_confidence >= 0.40:
        return "MARGINAL", "Detector works intermittently but is not match-ready.", "Add labels from missed/low-confidence frames and retrain."
    return "POOR", "Detector is not reliable enough for DRS decisions.", "Label more real footage and retrain before live use."


def validate_video(model: YOLO, video_path: Path, args: argparse.Namespace) -> VideoMetrics:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if fps else 0.0
    video_dir = args.output_dir / video_path.stem
    worst_dir = video_dir / "worst"
    if args.save_frames:
        video_dir.mkdir(parents=True, exist_ok=True)
        worst_dir.mkdir(parents=True, exist_ok=True)

    sampled = 0
    frames_with_detection = 0
    confidences: list[float] = []
    implausible_count = 0
    continuity_total = 0
    continuity_hits = 0
    previous_center: tuple[float, float] | None = None
    worst_frames: list[tuple[float, int, np.ndarray]] = []

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % args.sample_rate != 0:
            frame_index += 1
            continue

        sampled += 1

        def _predict(dev: str):
            return model(frame, conf=args.conf, verbose=False, device=dev)[0]

        result = run_with_cpu_fallback(_predict, resolve_device(args.device))
        detections = extract_ball_detections(result, frame.shape)
        if detections:
            frames_with_detection += 1
            best = max(detections, key=lambda item: item["confidence"])
            confidences.extend(item["confidence"] for item in detections)
            implausible_count += sum(1 for item in detections if is_implausible_detection(item))
            if len(detections) > 1:
                implausible_count += len(detections) - 1
            if previous_center is not None:
                continuity_total += 1
                distance = float(np.linalg.norm(np.array([best["cx"], best["cy"]]) - np.array(previous_center)))
                if distance <= 150.0:
                    continuity_hits += 1
            previous_center = best["cx"], best["cy"]
            if args.save_frames:
                annotated = draw_detections(frame, detections, frame_index)
                cv2.imwrite(str(video_dir / f"frame_{frame_index:06d}.jpg"), annotated)
                worst_frames.append((best["confidence"], frame_index, annotated))
        frame_index += 1

    cap.release()
    detection_rate = frames_with_detection / max(1, sampled)
    avg_confidence = mean(confidences) if confidences else 0.0
    max_confidence = max(confidences) if confidences else 0.0
    min_confidence = min(confidences) if confidences else 0.0
    false_positive_estimate = implausible_count / max(1, len(confidences))
    continuity = continuity_hits / max(1, continuity_total)
    verdict, reason, recommendation = verdict_for(detection_rate, avg_confidence, continuity)

    if args.save_frames:
        for confidence, worst_frame_index, annotated in sorted(worst_frames, key=lambda item: item[0])[:5]:
            cv2.imwrite(str(worst_dir / f"frame_{worst_frame_index:06d}_conf_{confidence:.2f}.jpg"), annotated)

    metrics = VideoMetrics(
        video=video_path.name,
        duration_seconds=round(duration, 3),
        frames_sampled=sampled,
        detection_rate=round(detection_rate, 4),
        avg_confidence=round(avg_confidence, 4),
        max_confidence=round(max_confidence, 4),
        min_confidence=round(min_confidence, 4),
        false_positive_estimate=round(false_positive_estimate, 4),
        tracking_continuity=round(continuity, 4),
        verdict=verdict,
        verdict_reason=reason,
        recommendation=recommendation,
    )
    report_path = args.output_dir / f"{video_path.stem}_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")
    print_summary(args.model, video_path, total_frames, args.sample_rate, metrics, report_path, worst_dir if args.save_frames else None)
    return metrics


def print_summary(
    model_path: Path,
    video_path: Path,
    total_frames: int,
    sample_rate: int,
    metrics: VideoMetrics,
    report_path: Path,
    worst_dir: Path | None,
) -> None:
    print("=" * 44)
    print("DETECTOR VALIDATION SUMMARY")
    print("=" * 44)
    print(f"Model: {model_path}")
    print(f"Video: {video_path.name} ({metrics.duration_seconds:.1f}s, {total_frames} frames)")
    print(f"Sampled: {metrics.frames_sampled} frames (every {sample_rate}rd)")
    print()
    print(f"Detection Rate:      {metrics.detection_rate * 100:.1f}%")
    print(f"Avg Confidence:      {metrics.avg_confidence:.3f}")
    print(f"Tracking Continuity: {metrics.tracking_continuity * 100:.1f}%")
    print(f"False Positives est: {metrics.false_positive_estimate * 100:.1f}%")
    print()
    print(f"VERDICT: {metrics.verdict} - {metrics.verdict_reason}")
    if worst_dir:
        print(f"Worst frames saved to: {worst_dir}")
    print(f"Full report: {report_path}")
    print("=" * 44)


def main() -> None:
    args = parse_args()
    if args.sample_rate < 1:
        raise ValueError("--sample-rate must be >= 1")
    videos = find_videos(args.video, args.video_dir, synthetic=args.synthetic)
    if not args.model.exists():
        fallback = Path("models/cricket_ball_yolov8.pt")
        if fallback.exists():
            args.model = fallback
        else:
            raise FileNotFoundError(f"No model found at {args.model} or {fallback}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Inference device: {device}")
    model = YOLO(str(args.model))
    model.to(device)
    all_metrics = [validate_video(model, video, args) for video in videos]
    if len(all_metrics) > 1:
        summary_path = args.output_dir / "summary.json"
        summary_path.write_text(json.dumps([asdict(item) for item in all_metrics], indent=2), encoding="utf-8")
        print(f"Combined summary: {summary_path}")


if __name__ == "__main__":
    main()
