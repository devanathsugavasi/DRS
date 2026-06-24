"""Camera synchronization manager with flash/audio anchors and dropped-frame stats."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class SyncFrame:
    camera_id: int
    timestamp_ms: float
    frame_id: int
    frame: np.ndarray


@dataclass(slots=True)
class CameraSyncHealth:
    camera_id: int
    offset_ms: float
    dropped_frames: int
    sync_health_percent: float
    last_timestamp_ms: float | None = None


class SyncManager:
    """Aligns camera frames by measured offsets and reports health."""

    def __init__(self, tolerance_ms: float = 2.0) -> None:
        self.tolerance_ms = tolerance_ms
        self.offsets_ms: dict[int, float] = {}
        self.frames: dict[int, list[SyncFrame]] = {}
        self.dropped_frames: dict[int, int] = {}

    def add_frame(self, item: SyncFrame) -> None:
        buffer = self.frames.setdefault(item.camera_id, [])
        if buffer and item.frame_id > buffer[-1].frame_id + 1:
            self.dropped_frames[item.camera_id] = self.dropped_frames.get(item.camera_id, 0) + (item.frame_id - buffer[-1].frame_id - 1)
        buffer.append(item)
        if len(buffer) > 600:
            del buffer[:-600]

    def detect_flash_anchor(self, frames_by_camera: dict[int, list[SyncFrame]]) -> dict[int, float]:
        anchors: dict[int, float] = {}
        for camera_id, frames in frames_by_camera.items():
            if not frames:
                continue
            brightness = [float(np.mean(cv2.cvtColor(item.frame, cv2.COLOR_BGR2GRAY))) for item in frames]
            if len(brightness) < 3:
                continue
            idx = int(np.argmax(np.diff(brightness, prepend=brightness[0])))
            anchors[camera_id] = frames[idx].timestamp_ms
        self._update_offsets_from_anchors(anchors)
        return anchors

    def detect_audio_click_anchor(self, audio_energy_by_camera: dict[int, list[tuple[float, float]]]) -> dict[int, float]:
        anchors: dict[int, float] = {}
        for camera_id, samples in audio_energy_by_camera.items():
            if not samples:
                continue
            anchors[camera_id] = max(samples, key=lambda item: item[1])[0]
        self._update_offsets_from_anchors(anchors)
        return anchors

    def get_sync_offset(self, camera_id: int) -> float:
        return self.offsets_ms.get(camera_id, 0.0)

    def get_aligned_frame(self, camera_id: int, target_timestamp: float) -> SyncFrame | None:
        buffer = self.frames.get(camera_id, [])
        if not buffer:
            return None
        corrected_target = target_timestamp + self.get_sync_offset(camera_id)
        nearest = min(buffer, key=lambda item: abs(item.timestamp_ms - corrected_target))
        frame_interval = self._frame_interval_ms(buffer)
        if abs(nearest.timestamp_ms - corrected_target) <= frame_interval:
            return nearest
        return None

    def health(self) -> list[CameraSyncHealth]:
        rows: list[CameraSyncHealth] = []
        for camera_id, buffer in self.frames.items():
            offset = self.get_sync_offset(camera_id)
            dropped = self.dropped_frames.get(camera_id, 0)
            health = max(0.0, 100.0 - abs(offset) * 3.0 - dropped * 2.0)
            rows.append(CameraSyncHealth(camera_id, offset, dropped, round(health, 2), buffer[-1].timestamp_ms if buffer else None))
        return rows

    def _update_offsets_from_anchors(self, anchors: dict[int, float]) -> None:
        if not anchors:
            return
        reference = min(anchors.values())
        for camera_id, timestamp in anchors.items():
            self.offsets_ms[camera_id] = reference - timestamp

    def _frame_interval_ms(self, buffer: list[SyncFrame]) -> float:
        if len(buffer) < 2:
            return 40.0
        deltas = np.diff([item.timestamp_ms for item in buffer[-30:]])
        return float(np.median(np.abs(deltas))) if deltas.size else 40.0
