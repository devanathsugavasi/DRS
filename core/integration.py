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
from core.synchronization import SyncReport, SyncVerifier
from utils.helpers import save_csv, timestamp_str
from utils.logger import get_logger

log = get_logger("drs_pipeline")


@dataclass(slots=True)
class PipelineFrame:
    video_frame: VideoFrame
    detection: DetectionResult
    track_point: Optional[TrackPoint]
    annotated: np.ndarray


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
            outputs[camera_id] = PipelineFrame(item, detection, point, annotated)
        return PipelineState(outputs, report)

    def export_tracking(self) -> dict[int, str]:
        paths: dict[int, str] = {}
        for camera_id, tracker in self.trackers.items():
            path = TRACKING_DIR / f"track_cam_{camera_id}_{timestamp_str()}.csv"
            save_csv(tracker.export_rows(), path)
            paths[camera_id] = str(path)
        return paths

    def _draw_status(self, frame: np.ndarray, report: SyncReport) -> None:
        color = (80, 240, 80) if report.within_tolerance else (0, 120, 255)
        cv2.putText(frame, f"sync {report.spread_ms:.1f}ms", (12, frame.shape[0] - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
