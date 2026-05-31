"""Timestamp alignment, dropped-frame detection, and flash sync support."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config.settings import SYNC_TOLERANCE_MS
from core.camera_manager import VideoFrame


@dataclass(slots=True)
class SyncReport:
    frame_id: int
    camera_count: int
    spread_ms: float
    within_tolerance: bool
    dropped_frames: dict[int, int]


class SyncVerifier:
    """Measures frame alignment and reports dropped sequence numbers."""

    def __init__(self, tolerance_ms: float = SYNC_TOLERANCE_MS) -> None:
        self.tolerance_ms = tolerance_ms
        self.last_frame_ids: dict[int, int] = {}

    def evaluate(self, frames: dict[int, VideoFrame]) -> SyncReport:
        if not frames:
            return SyncReport(0, 0, 0.0, True, {})
        timestamps = [item.timestamp_ms for item in frames.values()]
        spread_ms = max(timestamps) - min(timestamps)
        dropped: dict[int, int] = {}
        for camera_id, item in frames.items():
            previous = self.last_frame_ids.get(camera_id)
            if previous is not None and item.frame_id > previous + 1:
                dropped[camera_id] = item.frame_id - previous - 1
            self.last_frame_ids[camera_id] = item.frame_id
        return SyncReport(
            frame_id=min(item.frame_id for item in frames.values()),
            camera_count=len(frames),
            spread_ms=spread_ms,
            within_tolerance=spread_ms <= self.tolerance_ms,
            dropped_frames=dropped,
        )


class FlashSynchronizer:
    """Detects a bright flash pulse for optional field synchronization."""

    def __init__(self, brightness_delta: float = 45.0) -> None:
        self.brightness_delta = brightness_delta
        self.baseline: dict[int, float] = {}

    def detect_flash(self, frame: VideoFrame) -> bool:
        gray = cv2.cvtColor(frame.frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        previous = self.baseline.get(frame.camera_id, brightness)
        self.baseline[frame.camera_id] = 0.9 * previous + 0.1 * brightness
        return brightness - previous >= self.brightness_delta
