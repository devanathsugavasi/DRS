"""Offline single/dual-camera cricket delivery DRS analysis pipeline."""

from __future__ import annotations

import csv
import json
import math
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.ball_detector import DetectionResult, BallDetector
from core.ball_tracker import BallTracker, TrackPoint
from core.lbw import LBWDecisionEngine
from core.tracking_quality import TrackingQualityAnalyzer
from core.trajectory import TrajectoryPredictor

TESTING_DATA_DIR = Path("data/testing")
UPLOAD_DIR = TESTING_DATA_DIR / "uploads"
OUTPUT_DIR = TESTING_DATA_DIR / "outputs"


@dataclass(slots=True)
class AnalysisOptions:
    ball_detection: bool = True
    ball_tracking: bool = True
    trajectory_prediction: bool = True
    lbw_analysis: bool = True
    edge_detection: bool = False
    replay_generation: bool = True
    max_frames: int | None = None
    confidence_threshold: float = 0.25


@dataclass(slots=True)
class ObjectEstimate:
    label: str
    bbox: tuple[int, int, int, int] | None
    confidence: float
    source: str


class DeliveryTestingPipeline:
    """Processes uploaded cricket delivery clips into DRS-style evidence."""

    def __init__(self, model_path: Path | str = Path("models/cricket_ball_yolov8.pt")) -> None:
        self.detector = BallDetector(model_path=model_path, export_results=False)
        self.trajectory = TrajectoryPredictor()
        self.lbw = LBWDecisionEngine()
        self.quality = TrackingQualityAnalyzer()

    def process(self, job_id: str, video_paths: list[Path], options: AnalysisOptions) -> dict[str, Any]:
        if len(video_paths) not in {1, 2}:
            raise ValueError("Testing platform supports one or two uploaded videos")

        job_dir = OUTPUT_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        camera_results = []
        all_tracks: list[dict[str, Any]] = []
        for camera_id, path in enumerate(video_paths):
            result = self._process_camera(job_id, camera_id, path, job_dir, options)
            camera_results.append(result)
            all_tracks.extend(result["tracking_points"])

        sync = self._synchronize(camera_results) if len(camera_results) == 2 else None
        fused = self._fuse_tracks(camera_results)
        decision = self._build_decision(fused, camera_results, bool(sync))
        animation_path = self._write_clean_drs_animation(job_dir, job_id, fused, decision)
        report_path = self._write_report(job_dir, job_id, camera_results, decision, sync)
        json_path = self._write_json(job_dir, job_id, camera_results, decision, sync)
        csv_path = self._write_csv(job_dir, job_id, all_tracks)

        return {
            "job_id": job_id,
            "mode": "dual_camera" if len(video_paths) == 2 else "single_camera",
            "status": "complete",
            "summary": decision,
            "sync": sync,
            "cameras": camera_results,
            "exports": {
                "json": str(json_path),
                "csv": str(csv_path),
                "pdf": str(report_path),
                "analyzed_video": camera_results[0]["analyzed_video"],
                "animation_video": str(animation_path),
                "screenshots": [item for cam in camera_results for item in cam["screenshots"]],
            },
            "calibration_status": {
                "ready_for_testing": True,
                "production_ready": False,
                "message": "Camera calibration code exists, but real DRS accuracy needs checkerboard captures from the exact match cameras and pitch.",
            },
            "model_status": {
                "ready_for_testing": self.detector.model is not None,
                "production_ready": False,
                "message": "YOLO model is loadable, but tournament-grade accuracy needs training/validation on your red and white ball footage.",
            },
        }

    def _process_camera(
        self,
        job_id: str,
        camera_id: int,
        video_path: Path,
        job_dir: Path,
        options: AnalysisOptions,
    ) -> dict[str, Any]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        tracker = BallTracker(fps=fps)
        output_path = job_dir / f"camera_{camera_id}_analyzed.mp4"
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        frame_id = 0
        detections: list[dict[str, Any]] = []
        object_estimates: dict[str, ObjectEstimate] = {}
        screenshots: list[str] = []
        bounce_point_px: tuple[int, int] | None = None
        impact_point_px: tuple[int, int] | None = None

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if options.max_frames is not None and frame_id >= options.max_frames:
                break

            timestamp_ms = (frame_id / fps) * 1000.0
            detection_result = self._detect_ball(frame, frame_id, timestamp_ms, camera_id, options)
            track_point = tracker.update(detection_result) if options.ball_tracking else None
            estimates = self._estimate_static_objects(frame)
            object_estimates.update({item.label: item for item in estimates})

            annotated = frame.copy()
            if options.ball_detection:
                annotated = self.detector.annotate(annotated, detection_result)
            if options.ball_tracking:
                annotated = tracker.draw(annotated)
            if track_point:
                bounce_point_px = self._estimate_bounce(tracker.history) or bounce_point_px
                impact_point_px = self._estimate_impact(tracker.history, object_estimates.get("pads")) or impact_point_px
            self._draw_drs_overlay(annotated, object_estimates, bounce_point_px, impact_point_px)
            writer.write(annotated)

            if frame_id in {0, max(0, total_frames // 2), max(0, total_frames - 1)}:
                shot_path = job_dir / f"camera_{camera_id}_frame_{frame_id}.jpg"
                cv2.imwrite(str(shot_path), annotated)
                screenshots.append(str(shot_path))

            best = detection_result.best
            detections.append(
                {
                    "frame_id": frame_id,
                    "timestamp_ms": timestamp_ms,
                    "camera_id": camera_id,
                    "bbox": best.bbox if best else None,
                    "center": [best.cx, best.cy] if best else None,
                    "confidence": best.confidence if best else 0.0,
                }
            )
            frame_id += 1

        cap.release()
        writer.release()
        tracks = [asdict(point) for point in tracker.history]
        quality = self.quality.evaluate(detections, tracks)
        speed_px_s = float(np.median([point["speed_px_s"] for point in tracks])) if tracks else 0.0
        pixels_per_meter = max(25.0, width / 20.12)
        speed_kmh = (speed_px_s / pixels_per_meter) * 3.6

        return {
            "camera_id": camera_id,
            "source_video": str(video_path),
            "analyzed_video": str(output_path),
            "fps": fps,
            "width": width,
            "height": height,
            "frames_processed": frame_id,
            "detections": detections,
            "tracking_points": tracks,
            "object_estimates": {key: asdict(value) for key, value in object_estimates.items()},
            "ball_speed_kmh": round(speed_kmh, 2),
            "bounce_point_px": list(bounce_point_px) if bounce_point_px else None,
            "impact_point_px": list(impact_point_px) if impact_point_px else None,
            "screenshots": screenshots,
            "confidence": quality.score,
            "tracking_quality": quality.to_dict(),
        }

    def _detect_ball(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_ms: float,
        camera_id: int,
        options: AnalysisOptions,
    ) -> DetectionResult:
        if not options.ball_detection:
            return DetectionResult(frame_id, timestamp_ms, camera_id, [], 0.0)
        result = self.detector.detect(frame, frame_id, timestamp_ms, camera_id)
        filtered = [item for item in result.detections if item.confidence >= options.confidence_threshold]
        return DetectionResult(frame_id, timestamp_ms, camera_id, filtered, result.inference_ms)

    def _estimate_static_objects(self, frame: np.ndarray) -> list[ObjectEstimate]:
        h, w = frame.shape[:2]
        stumps = ObjectEstimate("stumps", (int(w * 0.78), int(h * 0.38), int(w * 0.84), int(h * 0.82)), 0.35, "geometry_fallback")
        pads = ObjectEstimate("pads", (int(w * 0.46), int(h * 0.42), int(w * 0.55), int(h * 0.86)), 0.25, "geometry_fallback")
        bat = ObjectEstimate("bat", (int(w * 0.38), int(h * 0.35), int(w * 0.45), int(h * 0.84)), 0.20, "geometry_fallback")
        return [stumps, pads, bat]

    def _estimate_bounce(self, points: list[TrackPoint]) -> tuple[int, int] | None:
        if len(points) < 5:
            return None
        velocities = np.array([point.vy for point in points], dtype=float)
        changes = np.diff(np.sign(velocities))
        candidates = np.where(changes < 0)[0]
        if candidates.size == 0:
            return None
        point = points[int(candidates[0])]
        return int(point.x), int(point.y)

    def _estimate_impact(self, points: list[TrackPoint], pads: ObjectEstimate | None) -> tuple[int, int] | None:
        if not points or pads is None or pads.bbox is None:
            return None
        x1, y1, x2, y2 = pads.bbox
        for point in points:
            if x1 <= point.x <= x2 and y1 <= point.y <= y2:
                return int(point.x), int(point.y)
        return None

    def _draw_drs_overlay(
        self,
        frame: np.ndarray,
        objects: dict[str, ObjectEstimate],
        bounce: tuple[int, int] | None,
        impact: tuple[int, int] | None,
    ) -> None:
        colors = {"stumps": (60, 255, 120), "pads": (255, 210, 80), "bat": (80, 180, 255)}
        for label, item in objects.items():
            if item.bbox is None:
                continue
            cv2.rectangle(frame, item.bbox[:2], item.bbox[2:], colors.get(label, (220, 220, 220)), 2)
            cv2.putText(frame, f"{label} {item.confidence:.0%}", (item.bbox[0], item.bbox[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors.get(label, (220, 220, 220)), 1)
        if bounce:
            cv2.circle(frame, bounce, 9, (0, 220, 255), 2)
            cv2.putText(frame, "Bounce", (bounce[0] + 10, bounce[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
        if impact:
            cv2.circle(frame, impact, 10, (0, 80, 255), 2)
            cv2.putText(frame, "Impact", (impact[0] + 10, impact[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 2)

    def _synchronize(self, camera_results: list[dict[str, Any]]) -> dict[str, Any]:
        frame_delta = abs(camera_results[0]["frames_processed"] - camera_results[1]["frames_processed"])
        fps_delta = abs(camera_results[0]["fps"] - camera_results[1]["fps"])
        confidence = max(0.35, 1.0 - (frame_delta * 0.01) - (fps_delta * 0.05))
        return {
            "method": "software_timestamp_alignment",
            "frame_delta": frame_delta,
            "fps_delta": round(fps_delta, 3),
            "confidence": round(confidence, 3),
        }

    def _fuse_tracks(self, camera_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        tracks = [cam["tracking_points"] for cam in camera_results if cam["tracking_points"]]
        if not tracks:
            return []
        if len(tracks) == 1:
            return tracks[0]
        limit = min(len(track) for track in tracks)
        fused = []
        for idx in range(limit):
            items = [track[idx] for track in tracks]
            fused.append(
                {
                    **items[0],
                    "x": float(np.mean([item["x"] for item in items])),
                    "y": float(np.mean([item["y"] for item in items])),
                    "confidence": float(np.mean([item["confidence"] for item in items])),
                    "source": "dual_camera_fusion",
                }
            )
        return fused

    def _build_decision(self, fused_tracks: list[dict[str, Any]], camera_results: list[dict[str, Any]], dual: bool) -> dict[str, Any]:
        avg_conf = float(np.mean([cam["confidence"] for cam in camera_results])) if camera_results else 0.0
        ball_speed = float(np.mean([cam["ball_speed_kmh"] for cam in camera_results])) if camera_results else 0.0
        main = camera_results[0]
        reliability_boost = 0.12 if dual else 0.0
        if fused_tracks:
            recent_confidence = float(np.mean([item.get("confidence", 0.0) for item in fused_tracks[-12:]]))
            avg_conf = (avg_conf * 0.65) + (recent_confidence * 0.35)
        hit_probability = min(0.96, max(0.05, avg_conf + reliability_boost))
        hitting = hit_probability >= 0.62 and main.get("impact_point_px") is not None
        decision = "OUT" if hitting else "NOT OUT"
        quality = [cam.get("tracking_quality", {}) for cam in camera_results]
        reliability = "high" if hit_probability >= 0.78 and all(item.get("reliability") == "high" for item in quality) else "medium" if hit_probability >= 0.58 else "low"
        return {
            "ball_speed_kmh": round(ball_speed, 2),
            "pitching_location": main.get("bounce_point_px") or "unknown",
            "impact_location": main.get("impact_point_px") or "unknown",
            "predicted_wicket_impact": "hitting" if hitting else "missing/uncertain",
            "lbw_recommendation": decision,
            "confidence_score": round(hit_probability, 3),
            "uncertainty": round(1.0 - hit_probability, 3),
            "reliability": reliability,
            "tracking_quality": quality,
            "notes": [
                "Single-camera mode estimates depth approximately; dual-camera mode improves confidence after calibration.",
                "Bat, pad, and stump detections use fallback geometry unless a dedicated multi-class YOLO model is trained.",
                "Use high-FPS side-on and stump-line cameras for serious LBW testing.",
            ],
        }

    def _write_clean_drs_animation(self, job_dir: Path, job_id: str, tracks: list[dict[str, Any]], decision: dict[str, Any]) -> Path:
        path = job_dir / "clean_drs_animation.mp4"
        width, height, fps = 1280, 720, 30
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        pitch_left, pitch_right = 210, 1070
        pitch_top, pitch_bottom = 120, 620
        crease_x = int(pitch_right - 120)
        stump_x = int(pitch_right - 70)
        center_y = (pitch_top + pitch_bottom) // 2

        normalized = self._normalize_track_for_animation(tracks, pitch_left, pitch_right, pitch_top, pitch_bottom)
        frames = max(90, len(normalized) * 3)
        for frame_idx in range(frames):
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            canvas[:] = (10, 35, 24)
            for y in range(0, height, 42):
                cv2.rectangle(canvas, (0, y), (width, y + 21), (14, 48, 31), -1)
            cv2.rectangle(canvas, (pitch_left, pitch_top), (pitch_right, pitch_bottom), (75, 105, 76), -1)
            cv2.rectangle(canvas, (pitch_left, pitch_top), (pitch_right, pitch_bottom), (160, 190, 170), 2)
            cv2.line(canvas, (crease_x, pitch_top), (crease_x, pitch_bottom), (220, 230, 210), 2)
            self._draw_animation_stumps(canvas, stump_x, center_y)

            upto = min(len(normalized), max(1, frame_idx // 3))
            visible = normalized[:upto]
            if len(visible) > 1:
                cv2.polylines(canvas, [np.array(visible, dtype=np.int32)], False, (60, 255, 140), 4, cv2.LINE_AA)
                glow = np.array(visible, dtype=np.int32)
                cv2.polylines(canvas, [glow], False, (30, 150, 255), 1, cv2.LINE_AA)
            if visible:
                cv2.circle(canvas, visible[-1], 10, (245, 245, 245), -1, cv2.LINE_AA)
                cv2.circle(canvas, visible[-1], 16, (60, 255, 140), 2, cv2.LINE_AA)

            self._draw_animation_panel(canvas, decision)
            writer.write(canvas)
        writer.release()
        return path

    def _normalize_track_for_animation(
        self,
        tracks: list[dict[str, Any]],
        left: int,
        right: int,
        top: int,
        bottom: int,
    ) -> list[tuple[int, int]]:
        if not tracks:
            return [(left + 70, bottom - 80), (right - 90, (top + bottom) // 2)]
        pts = np.array([[float(item["x"]), float(item["y"])] for item in tracks], dtype=float)
        min_xy = pts.min(axis=0)
        span = np.maximum(pts.max(axis=0) - min_xy, 1.0)
        normalized = (pts - min_xy) / span
        x = left + normalized[:, 0] * (right - left - 120) + 40
        y = bottom - normalized[:, 1] * (bottom - top - 80) - 40
        return [(int(px), int(py)) for px, py in zip(x, y)]

    def _draw_animation_stumps(self, canvas: np.ndarray, x: int, y: int) -> None:
        for offset in (-18, 0, 18):
            cv2.line(canvas, (x + offset, y - 76), (x + offset, y + 76), (235, 228, 190), 7, cv2.LINE_AA)
        cv2.line(canvas, (x - 27, y - 82), (x + 27, y - 82), (235, 228, 190), 4, cv2.LINE_AA)

    def _draw_animation_panel(self, canvas: np.ndarray, decision: dict[str, Any]) -> None:
        cv2.rectangle(canvas, (36, 36), (490, 180), (4, 12, 28), -1)
        cv2.rectangle(canvas, (36, 36), (490, 180), (60, 255, 140), 1)
        result = decision.get("lbw_recommendation", "PENDING")
        color = (60, 80, 255) if result == "OUT" else (50, 205, 255)
        cv2.putText(canvas, "CLEAN DRS ANIMATION", (58, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (225, 238, 255), 2)
        cv2.putText(canvas, result, (58, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.55, color, 3, cv2.LINE_AA)
        confidence = int(float(decision.get("confidence_score", 0.0)) * 100)
        cv2.putText(canvas, f"CONF {confidence}% | {decision.get('reliability', 'low').upper()}", (58, 164), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (60, 255, 140), 2)

    def _confidence(self, detections: list[dict[str, Any]], tracks: list[dict[str, Any]]) -> float:
        if not detections:
            return 0.0
        detected = [item["confidence"] for item in detections if item["confidence"] > 0]
        detection_rate = len(detected) / max(1, len(detections))
        avg_detection = float(np.mean(detected)) if detected else 0.0
        track_rate = len(tracks) / max(1, len(detections))
        return round(min(1.0, (avg_detection * 0.55) + (detection_rate * 0.3) + (track_rate * 0.15)), 3)

    def _write_json(self, job_dir: Path, job_id: str, cameras: list[dict[str, Any]], decision: dict[str, Any], sync: dict[str, Any] | None) -> Path:
        path = job_dir / "tracking_data.json"
        path.write_text(json.dumps({"job_id": job_id, "decision": decision, "sync": sync, "cameras": cameras}, indent=2), encoding="utf-8")
        return path

    def _write_csv(self, job_dir: Path, job_id: str, points: list[dict[str, Any]]) -> Path:
        path = job_dir / "tracking_data.csv"
        fields = ["job_id", "camera_id", "frame_id", "timestamp_ms", "x", "y", "vx", "vy", "speed_px_s", "direction_deg", "confidence", "predicted"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for point in points:
                writer.writerow({"job_id": job_id, **point})
        return path

    def _write_report(self, job_dir: Path, job_id: str, cameras: list[dict[str, Any]], decision: dict[str, Any], sync: dict[str, Any] | None) -> Path:
        path = job_dir / "drs_report.pdf"
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas

            pdf = canvas.Canvas(str(path), pagesize=A4)
            y = 800
            pdf.setFont("Helvetica-Bold", 16)
            pdf.drawString(42, y, "Cricket DRS Testing Report")
            y -= 32
            pdf.setFont("Helvetica", 10)
            for label, value in [
                ("Job ID", job_id),
                ("Mode", "Dual Camera" if len(cameras) == 2 else "Single Camera"),
                ("LBW Recommendation", decision["lbw_recommendation"]),
                ("Confidence", f"{decision['confidence_score']:.0%}"),
                ("Ball Speed", f"{decision['ball_speed_kmh']} km/h"),
                ("Pitching", str(decision["pitching_location"])),
                ("Impact", str(decision["impact_location"])),
                ("Wicket Impact", decision["predicted_wicket_impact"]),
                ("Sync", json.dumps(sync) if sync else "single camera"),
            ]:
                pdf.drawString(42, y, f"{label}: {value}")
                y -= 20
            pdf.save()
        except Exception:
            path.write_text(json.dumps({"job_id": job_id, "decision": decision, "sync": sync}, indent=2), encoding="utf-8")
        return path


def stage_uploads(files: list[Path]) -> tuple[str, list[Path]]:
    job_id = uuid.uuid4().hex[:12]
    job_upload_dir = UPLOAD_DIR / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)
    staged = []
    for idx, source in enumerate(files):
        suffix = source.suffix or ".mp4"
        dest = job_upload_dir / f"camera_{idx}{suffix}"
        shutil.copy2(source, dest)
        staged.append(dest)
    return job_id, staged
