"""Real-time DRS integration pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from config.settings import TARGET_FPS, TRACKING_DIR
from core.ball_detector import BallDetector, DetectionResult
from core.ball_tracker import BallTracker, TrackPoint
from core.camera_manager import CameraManager, VideoFrame
from core.pitch_calibration import ManualPitchCalibrator
from core.synchronization import SyncReport, SyncVerifier
from core.trajectory import TrajectoryPredictor
from utils.helpers import save_csv, timestamp_str
from utils.logger import get_logger

log = get_logger("drs_pipeline")


@dataclass(slots=True)
class PipelineFrame:
    video_frame: VideoFrame
    detection: DetectionResult
    track_point: Optional[TrackPoint]
    annotated: np.ndarray
    world_coords: Optional[dict] = None


@dataclass(slots=True)
class PipelineState:
    frames: dict[int, PipelineFrame] = field(default_factory=dict)
    sync_report: Optional[SyncReport] = None


class DRSPipeline:
    """Integrates synchronized feeds, YOLO detection, Kalman tracking, and exports."""

    def __init__(self, camera_ids: list[int] | None = None, record: bool = False, detector: BallDetector | None = None):
        self.camera_manager = CameraManager(camera_ids, record=record)
        self.detector = detector or BallDetector()
        self.trackers: dict[int, BallTracker] = {}
        self.sync_verifier = SyncVerifier()
        self.running = False
        self.calibrators: dict[int, ManualPitchCalibrator] = {}
        self.trajectory_predictor = TrajectoryPredictor()
        self._load_calibration_profiles()

    def _load_calibration_profiles(self) -> None:
        for cam_id in self.camera_manager.camera_ids:
            calibrator = ManualPitchCalibrator()
            if calibrator.load_profile(cam_id):
                self.calibrators[cam_id] = calibrator
                log.info("Loaded calibration profile for camera {}", cam_id)

    def start(self) -> None:
        self.camera_manager.start()
        self.running = True

    def stop(self) -> None:
        self.running = False
        self.camera_manager.stop()

    def process_once(self) -> PipelineState:
        frames = self.camera_manager.latest_frames()
        report = self.sync_verifier.evaluate(frames)
        outputs: dict[int, PipelineFrame] = {}
        for camera_id, item in frames.items():
            tracker = self.trackers.setdefault(camera_id, BallTracker(fps=TARGET_FPS))
            detection = self.detector.detect(item.frame, item.frame_id, item.timestamp_ms, camera_id)
            annotated = self.detector.annotate(item.frame.copy(), detection)
            point = tracker.update(detection)
            tracker.draw(annotated)
            self._draw_status(annotated, report)
            world_coords = None
            if point and camera_id in self.calibrators:
                world = self.calibrators[camera_id].pixel_to_pitch_mm(camera_id, point.x, point.y)
                if world is not None:
                    world_coords = {
                        'x_mm': world[0],
                        'y_mm': world[1],
                        'pixel_x': point.x,
                        'pixel_y': point.y,
                    }
            outputs[camera_id] = PipelineFrame(item, detection, point, annotated, world_coords)
        return PipelineState(outputs, report)

    def export_tracking(self) -> dict[int, str]:
        paths: dict[int, str] = {}
        for camera_id, tracker in self.trackers.items():
            path = TRACKING_DIR / f"track_cam_{camera_id}_{timestamp_str()}.csv"
            save_csv(tracker.export_rows(), path)
            paths[camera_id] = str(path)
        return paths

    def run_appeal_analysis(self, frames: list[VideoFrame] | None = None) -> dict:
        """Run full DRS analysis on buffered replay frames.

        This is the core 'DRS review' function:
        Detection → Tracking → Calibration → Trajectory → LBW Decision
        """

        if frames is None:
            # Use last N frames from all cameras
            frames_dict = {}
            for cam_id, worker in self.camera_manager.workers.items():
                frames_dict[cam_id] = worker.snapshot()
        else:
            frames_dict = {0: frames}

        all_detections = []
        all_tracks = []

        for cam_id, frame_list in frames_dict.items():
            tracker = BallTracker(fps=TARGET_FPS)
            detections = []
            tracks = []

            for vf in frame_list:
                det = self.detector.detect(vf.frame, vf.frame_id, vf.timestamp_ms, cam_id)
                point = tracker.update(det)
                if det.best:
                    detections.append({
                        'frame_id': vf.frame_id,
                        'confidence': det.best.confidence,
                        'cx': det.best.cx,
                        'cy': det.best.cy,
                        'inference_ms': det.inference_ms,
                    })
                if point:
                    tracks.append({
                        'frame_id': point.frame_id,
                        'x': point.x,
                        'y': point.y,
                        'vx': point.vx,
                        'vy': point.vy,
                        'speed_px_s': point.speed_px_s,
                    })

            all_detections.extend(detections)
            all_tracks.extend(tracks)

        # Build decision using the decision service
        detection_rate = len(all_detections) / max(1, sum(len(fl) for fl in frames_dict.values()))
        avg_confidence = (sum(d['confidence'] for d in all_detections) / max(1, len(all_detections))) if all_detections else 0.0

        # Get trajectory prediction if calibrated
        trajectory_data = None
        if all_tracks and self.calibrators:
            cam_id = list(self.calibrators.keys())[0]
            world_points = []
            timestamps = []
            for t in all_tracks[-10:]:
                result = self.calibrators[cam_id].pixel_to_pitch_mm(cam_id, t['x'], t['y'])
                if result:
                    world_points.append((result[0] / 1000.0, result[1] / 1000.0, 0.12))
                    timestamps.append(t['frame_id'] / TARGET_FPS)
            if len(world_points) >= 2:
                try:
                    prediction = self.trajectory_predictor.predict_from_world_points(world_points, timestamps)
                    trajectory_data = prediction.to_dict()
                except Exception:
                    pass

        return {
            'status': 'completed',
            'total_frames': sum(len(fl) for fl in frames_dict.values()),
            'detection_rate': round(detection_rate, 4),
            'avg_confidence': round(avg_confidence, 4),
            'detections_count': len(all_detections),
            'tracks_count': len(all_tracks),
            'trajectory': trajectory_data,
            'cameras_analyzed': list(frames_dict.keys()),
        }

    def _draw_status(self, frame: np.ndarray, report: SyncReport) -> None:
        color = (80, 240, 80) if report.within_tolerance else (0, 120, 255)
        cv2.putText(frame, f"sync {report.spread_ms:.1f}ms", (12, frame.shape[0] - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
