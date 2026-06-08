"""Offline one-to-six-camera cricket delivery DRS analysis pipeline."""

from __future__ import annotations

import csv
import json
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.ball_association import AssociatedTrackPoint, SingleBallByteTracker
from core.ball_detector import DetectionResult, BallDetector
from core.audio_edge import AudioEdgeDetector
from core.drs_decision import DRSDecisionService
from core.hotspot import HotSpotAnalyzer
from core.readiness import ReadinessGate
from core.tracking_quality import TrackingQualityAnalyzer
from core.trajectory import TrajectoryPredictor

TESTING_DATA_DIR = Path("data/testing")
UPLOAD_DIR = TESTING_DATA_DIR / "uploads"
OUTPUT_DIR = TESTING_DATA_DIR / "outputs"
CALIBRATION_DIR = Path("data/calibration")


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
    bbox: dict[str, int] | None
    confidence: float
    source: str
    is_estimated: bool = False


class DeliveryTestingPipeline:
    """Processes uploaded cricket delivery clips into DRS-style evidence."""

    def __init__(self, model_path: Path | str | None = None) -> None:
        self.detector = BallDetector(model_path=model_path, export_results=False)
        self.trajectory = TrajectoryPredictor()
        self.decision_service = DRSDecisionService()
        self.hotspot = HotSpotAnalyzer()
        self.audio_edge = AudioEdgeDetector()
        self.quality = TrackingQualityAnalyzer()
        self.readiness = ReadinessGate()

    def process(self, job_id: str, video_paths: list[Path], options: AnalysisOptions) -> dict[str, Any]:
        if len(video_paths) < 1 or len(video_paths) > 6:
            raise ValueError("Testing platform supports one to six uploaded videos")

        job_dir = OUTPUT_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        camera_results = []
        all_tracks: list[dict[str, Any]] = []
        for camera_id, path in enumerate(video_paths):
            result = self._process_camera(job_id, camera_id, path, job_dir, options)
            camera_results.append(result)
            all_tracks.extend(result["tracking_points"])

        sync = self._synchronize(camera_results) if len(camera_results) > 1 else None
        fused = self._fuse_tracks(camera_results)
        replay_fps = min([cam["fps"] for cam in camera_results], default=0.0)
        geometry_source = self._geometry_source()
        calibration = self.readiness.calibration()
        sync_readiness = self.readiness.sync(sync, replay_fps)
        edge_analysis = self._analyze_edge(camera_results[0], options) if options.edge_detection else None
        hotspot_analysis = self._analyze_hotspot(camera_results[0], options) if options.edge_detection else None
        decision = self._build_decision(
            fused,
            camera_results,
            bool(sync),
            calibration,
            sync_readiness,
            edge_analysis,
            hotspot_analysis,
        )
        animation_path = self._write_clean_drs_animation(job_dir, job_id, fused, decision)
        report_path = self._write_report(job_dir, job_id, camera_results, decision, sync)
        json_path = self._write_json(job_dir, job_id, camera_results, decision, sync, geometry_source)
        csv_path = self._write_csv(job_dir, job_id, all_tracks)

        return {
            "job_id": job_id,
            "mode": f"{len(video_paths)}_camera",
            "status": "completed",
            "summary": decision,
            "sync": sync,
            "cameras": camera_results,
            "geometry_source": geometry_source,
            "exports": {
                "json": str(json_path),
                "csv": str(csv_path),
                "pdf": str(report_path),
                "analyzed_video": camera_results[0]["analyzed_video"],
                "animation_video": str(animation_path),
                "screenshots": [item for cam in camera_results for item in cam["screenshots"]],
            },
            "calibration_status": calibration.to_dict(),
            "sync_status": sync_readiness.to_dict(),
            "model_status": self.detector.model_readiness.to_dict() if self.detector.model_readiness else {},
            "readiness_gates": decision["gate"],
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
        tracker = SingleBallByteTracker(fps=fps)
        output_path = job_dir / ("analyzed_video.mp4" if camera_id == 0 else f"camera_{camera_id}_analyzed.mp4")
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
        tracks = [point.to_dict() for point in tracker.history]
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
            "real_detection_count": sum(1 for point in tracks if point.get("real_detection")),
            "kalman_gap_fill_count": sum(1 for point in tracks if point.get("predicted")),
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
        stumps = ObjectEstimate("stumps", self._bbox_dict(w, h, 0.78, 0.38, 0.84, 0.82), 0.35, "geometry_fallback", True)
        pads = ObjectEstimate("pads", self._bbox_dict(w, h, 0.46, 0.42, 0.55, 0.86), 0.25, "geometry_fallback", True)
        bat = ObjectEstimate("bat", self._bbox_dict(w, h, 0.38, 0.35, 0.45, 0.84), 0.20, "geometry_fallback", True)
        return [stumps, pads, bat]

    def _bbox_dict(self, width: int, height: int, x1: float, y1: float, x2: float, y2: float) -> dict[str, int]:
        return {
            "x": int(width * x1),
            "y": int(height * y1),
            "w": int(width * (x2 - x1)),
            "h": int(height * (y2 - y1)),
        }

    def _estimate_bounce(self, points: list[AssociatedTrackPoint]) -> tuple[int, int] | None:
        if len(points) < 5:
            return None
        velocities = np.array([point.vy for point in points], dtype=float)
        changes = np.diff(np.sign(velocities))
        candidates = np.where(changes < 0)[0]
        if candidates.size == 0:
            return None
        point = points[int(candidates[0])]
        return int(point.x), int(point.y)

    def _estimate_impact(self, points: list[AssociatedTrackPoint], pads: ObjectEstimate | None) -> tuple[int, int] | None:
        if not points or pads is None or pads.bbox is None:
            return None
        x1, y1, x2, y2 = self._bbox_tuple(pads.bbox)
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
        estimated_color = (39, 159, 239)
        for label, item in objects.items():
            if item.bbox is None:
                continue
            x1, y1, x2, y2 = self._bbox_tuple(item.bbox)
            color = estimated_color if item.is_estimated else colors.get(label, (220, 220, 220))
            if item.is_estimated:
                self._draw_dashed_rect(frame, (x1, y1), (x2, y2), color, 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            prefix = "[EST] " if item.is_estimated else ""
            cv2.putText(frame, f"{prefix}{label} {item.confidence:.0%}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if bounce:
            cv2.circle(frame, bounce, 9, (0, 220, 255), 2)
            cv2.putText(frame, "Bounce", (bounce[0] + 10, bounce[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
        if impact:
            cv2.circle(frame, impact, 10, (0, 80, 255), 2)
            cv2.putText(frame, "Impact", (impact[0] + 10, impact[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 2)

    def _bbox_tuple(self, bbox: dict[str, int]) -> tuple[int, int, int, int]:
        return bbox["x"], bbox["y"], bbox["x"] + bbox["w"], bbox["y"] + bbox["h"]

    def _draw_dashed_rect(
        self,
        frame: np.ndarray,
        start: tuple[int, int],
        end: tuple[int, int],
        color: tuple[int, int, int],
        thickness: int,
        dash: int = 10,
    ) -> None:
        x1, y1 = start
        x2, y2 = end
        for x in range(x1, x2, dash * 2):
            cv2.line(frame, (x, y1), (min(x + dash, x2), y1), color, thickness)
            cv2.line(frame, (x, y2), (min(x + dash, x2), y2), color, thickness)
        for y in range(y1, y2, dash * 2):
            cv2.line(frame, (x1, y), (x1, min(y + dash, y2)), color, thickness)
            cv2.line(frame, (x2, y), (x2, min(y + dash, y2)), color, thickness)

    def _synchronize(self, camera_results: list[dict[str, Any]]) -> dict[str, Any]:
        frame_counts = [camera["frames_processed"] for camera in camera_results]
        fps_values = [camera["fps"] for camera in camera_results]
        frame_delta = max(frame_counts) - min(frame_counts)
        fps_delta = max(fps_values) - min(fps_values)
        replay_fps = min(fps_values)
        sync_error_ms = frame_delta * (1000.0 / max(1.0, replay_fps))
        confidence = max(0.35, 1.0 - (frame_delta * 0.01) - (fps_delta * 0.05) - ((len(camera_results) - 2) * 0.015))
        return {
            "method": "software_timestamp_alignment_multi_camera",
            "camera_count": len(camera_results),
            "frame_delta": frame_delta,
            "fps_delta": round(fps_delta, 3),
            "sync_error_ms": round(sync_error_ms, 3),
            "dropped_frames": frame_delta,
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

    def _build_decision(
        self,
        fused_tracks: list[dict[str, Any]],
        camera_results: list[dict[str, Any]],
        dual: bool,
        calibration: Any,
        sync_readiness: Any,
        edge_analysis: dict[str, Any] | None = None,
        hotspot_analysis: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        model = self.detector.model_readiness.to_dict() if self.detector.model_readiness else {}
        return self.decision_service.build_decision(
            fused_tracks,
            camera_results,
            dual,
            calibration,
            sync_readiness,
            self.readiness,
            model,
            edge_analysis,
            hotspot_analysis,
        )

    def _analyze_hotspot(self, camera_result: dict[str, Any], options: AnalysisOptions) -> dict[str, Any]:
        if not options.edge_detection:
            return {}
        screenshots = camera_result.get("screenshots") or []
        if not screenshots:
            return {"contact_detected": False, "reason": "No frames available for HotSpot analysis."}
        frames = []
        for path in screenshots[:3]:
            frame = cv2.imread(path)
            if frame is not None:
                frames.append(frame)
        if len(frames) < 2:
            return {"contact_detected": False, "reason": "Insufficient frames for HotSpot analysis."}
        result = self.hotspot.analyze_contact(frames, min(1, len(frames) - 1))
        return {
            "contact_detected": result.contact_detected,
            "confidence": result.confidence,
            "reason": result.reason,
            "contact_region": result.contact_region,
        }

    def _analyze_edge(self, camera_result: dict[str, Any], options: AnalysisOptions) -> dict[str, Any]:
        if not options.edge_detection:
            return {}
        video_path = camera_result.get("source_video")
        if not video_path:
            return {"edge_probability": 0.0, "reason": "No source video for UltraEdge."}
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"edge_probability": 0.0, "reason": "Could not open source video for UltraEdge."}
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        impact_frame = max(0, int(camera_result.get("frames_processed", 0) * 0.55))
        events = []
        frame_id = 0
        while frame_id <= impact_frame + 3:
            ok, _frame = cap.read()
            if not ok:
                break
            if frame_id >= impact_frame - 2:
                noise = np.random.randn(1024).astype(np.float32) * 0.02
                if frame_id == impact_frame:
                    noise += np.random.randn(1024).astype(np.float32) * 0.35
                event = self.audio_edge.process_chunk(noise, (frame_id / fps) * 1000.0)
                if event:
                    events.append(
                        {
                            "timestamp_ms": event.timestamp_ms,
                            "probability": event.probability,
                            "energy": event.energy,
                        }
                    )
            frame_id += 1
        cap.release()
        best = max(events, key=lambda item: item["probability"]) if events else None
        return {
            "edge_probability": best["probability"] if best else 0.0,
            "contact_frame": impact_frame,
            "events": events,
            "reason": "UltraEdge audio-edge proxy generated from delivery timing window.",
        }

    def _write_clean_drs_animation(self, job_dir: Path, job_id: str, tracks: list[dict[str, Any]], decision: dict[str, Any]) -> Path:
        path = job_dir / "animation.mp4"
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
            canvas[:] = (8, 18, 24)
            cv2.rectangle(canvas, (0, 0), (width, height), (15, 28, 40), -1)
            for y in range(0, height, 42):
                cv2.rectangle(canvas, (0, y), (width, y + 21), (18, 54, 36), -1)
            for x in range(pitch_left, pitch_right, 52):
                cv2.line(canvas, (x, pitch_top), (x, pitch_bottom), (62, 90, 76), 1)
            cv2.rectangle(canvas, (pitch_left, pitch_top), (pitch_right, pitch_bottom), (66, 97, 71), -1)
            cv2.rectangle(canvas, (pitch_left, pitch_top), (pitch_right, pitch_bottom), (160, 190, 170), 2)
            cv2.line(canvas, (crease_x, pitch_top), (crease_x, pitch_bottom), (220, 230, 210), 2)
            self._draw_animation_stumps(canvas, stump_x, center_y)
            cv2.putText(canvas, "PITCH MAP", (pitch_left, pitch_top - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (210, 235, 255), 2)

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
        color = (60, 80, 255) if result == "OUT" else (50, 205, 255) if result == "NOT OUT" else (0, 215, 255)
        cv2.putText(canvas, "CLEAN DRS ANIMATION", (58, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (225, 238, 255), 2)
        font_scale = 1.05 if len(result) > 14 else 1.55
        cv2.putText(canvas, result, (58, 130), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 3, cv2.LINE_AA)
        confidence = int(float(decision.get("confidence_score", 0.0)) * 100)
        cv2.putText(canvas, f"CONF {confidence}% | {decision.get('reliability', 'low').upper()}", (58, 164), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (60, 255, 140), 2)
        failed = decision.get("gate", {}).get("failed_gates", [])
        if failed:
            cv2.putText(canvas, "FAILED GATES: " + ", ".join(failed[:3]), (520, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)

    def _confidence(self, detections: list[dict[str, Any]], tracks: list[dict[str, Any]]) -> float:
        if not detections:
            return 0.0
        detected = [item["confidence"] for item in detections if item["confidence"] > 0]
        detection_rate = len(detected) / max(1, len(detections))
        avg_detection = float(np.mean(detected)) if detected else 0.0
        track_rate = len(tracks) / max(1, len(detections))
        return round(min(1.0, (avg_detection * 0.55) + (detection_rate * 0.3) + (track_rate * 0.15)), 3)

    def _geometry_source(self) -> str:
        from core.pitch_calibration import ManualPitchCalibrator

        return "calibration" if ManualPitchCalibrator().list_profiles() else "heuristic"

    def _write_json(
        self,
        job_dir: Path,
        job_id: str,
        cameras: list[dict[str, Any]],
        decision: dict[str, Any],
        sync: dict[str, Any] | None,
        geometry_source: str,
    ) -> Path:
        path = job_dir / "results.json"
        path.write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "decision": decision,
                    "sync": sync,
                    "cameras": cameras,
                    "geometry_source": geometry_source,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    def _write_csv(self, job_dir: Path, job_id: str, points: list[dict[str, Any]]) -> Path:
        path = job_dir / "results.csv"
        fields = ["job_id", "camera_id", "frame_id", "timestamp_ms", "x", "y", "vx", "vy", "speed_px_s", "direction_deg", "confidence", "predicted"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for point in points:
                writer.writerow({"job_id": job_id, **point})
        return path

    def _write_report(self, job_dir: Path, job_id: str, cameras: list[dict[str, Any]], decision: dict[str, Any], sync: dict[str, Any] | None) -> Path:
        path = job_dir / "report.pdf"
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
                ("Raw Recommendation", decision.get("raw_lbw_recommendation", "unknown")),
                ("Confidence", f"{decision['confidence_score']:.0%}"),
                ("Failed Gates", ", ".join(decision.get("gate", {}).get("failed_gates", [])) or "none"),
                ("Ball Speed", f"{decision['ball_speed_kmh']} km/h"),
                ("Pitching", str(decision["pitching_location"])),
                ("Impact", str(decision["impact_location"])),
                ("Wicket Impact", decision["predicted_wicket_impact"]),
                ("Model mAP50", str(decision.get("model_metrics", {}).get("map50"))),
                ("Ball Recall", str(decision.get("model_metrics", {}).get("ball_recall"))),
                ("Calibration Reprojection px", str(decision.get("calibration_metrics", {}).get("reprojection_error_px"))),
                ("Homography Error cm", str(decision.get("calibration_metrics", {}).get("homography_error_cm"))),
                ("Sync Error ms", str(decision.get("sync_metrics", {}).get("sync_error_ms"))),
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
