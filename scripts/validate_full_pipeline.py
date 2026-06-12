"""Run full offline DRS pipeline validation on a video."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.testing_pipeline import AnalysisOptions, DeliveryTestingPipeline, OUTPUT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the full offline DRS pipeline")
    parser.add_argument("--video", type=Path, required=True, help="Video file to analyze")
    parser.add_argument("--model", type=Path, default=Path("models/cricket_ball_yolov8.pt"), help="Detector model path")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame cap for quick validation")
    parser.add_argument("--edge-detection", action="store_true", help="Enable audio/HotSpot proxy evidence")
    parser.add_argument("--confidence-threshold", type=float, default=0.25, help="Detector confidence threshold")
    return parser.parse_args()


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> None:
    args = parse_args()
    if not args.video.exists():
        raise FileNotFoundError(args.video)
    model_path = args.model if args.model.exists() else None
    pipeline = DeliveryTestingPipeline(model_path=model_path)
    options = AnalysisOptions(
        edge_detection=args.edge_detection,
        max_frames=args.max_frames,
        confidence_threshold=args.confidence_threshold,
        replay_generation=True,
    )
    job_id = f"validation_{uuid.uuid4().hex[:8]}"
    result = pipeline.process(job_id, [args.video], options)
    camera = result["cameras"][0] if result.get("cameras") else {}
    frames = int(camera.get("frames_processed") or 0)
    detections = camera.get("detections") or []
    tracks = camera.get("tracking_points") or []
    detected_count = sum(1 for item in detections if item.get("confidence", 0.0) > 0)
    tracked_count = len(tracks)
    detection_rate = detected_count / max(1, frames)
    tracking_rate = tracked_count / max(1, frames)
    summary = result.get("summary", {})
    gate = summary.get("gate", {})
    decision = summary.get("lbw_recommendation") or summary.get("raw_lbw_recommendation") or "UNKNOWN"
    confidence = float(summary.get("confidence_score") or 0.0)
    status = "WORKING" if result.get("status") == "completed" else "FAILED"
    report_path = OUTPUT_DIR / job_id / "full_pipeline_validation.json"
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("=" * 44)
    print("FULL PIPELINE VALIDATION")
    print("=" * 44)
    print(f"Frames processed:     {frames}")
    print(f"Ball detected:        {percent(detection_rate)}")
    print(f"Successfully tracked: {percent(tracking_rate)}")
    print(f"Trajectory predicted: {'YES' if summary.get('predicted_wicket_impact') else 'NO'}")
    print(f"LBW analysis:         {'YES' if summary else 'NO'}")
    print()
    print(f"DECISION: {decision} (confidence: {confidence:.2f})")
    print(f"Pitching:  {summary.get('pitching_location', 'unknown')}")
    print(f"Impact:    {summary.get('impact_location', 'unknown')}")
    print(f"Wickets:   {summary.get('predicted_wicket_impact', 'unknown')}")
    if gate.get("failed_gates"):
        print(f"Failed gates: {', '.join(gate['failed_gates'])}")
    print()
    print(f"Pipeline STATUS: {status}")
    print(f"Report: {report_path}")
    print("=" * 44)


if __name__ == "__main__":
    main()
