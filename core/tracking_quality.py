"""Tracking quality scoring and DRS reliability gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class TrackingQuality:
    detection_rate: float
    detection_coverage: float
    mean_confidence: float
    track_coverage: float
    jitter_px: float
    trajectory_smoothness: float
    missing_frames: int
    max_missing_gap: int
    jump_rejections: int
    reliability: str
    score: float
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TrackingQualityAnalyzer:
    """Converts raw tracking output into a conservative confidence signal."""

    def evaluate(self, detections: list[dict[str, Any]], tracks: list[dict[str, Any]]) -> TrackingQuality:
        frame_count = max(1, len(detections))
        detected = [item for item in detections if item.get("confidence", 0.0) > 0]
        confidences = [float(item["confidence"]) for item in detected]
        detection_rate = len(detected) / frame_count
        detection_coverage = detection_rate
        mean_confidence = float(np.mean(confidences)) if confidences else 0.0
        track_coverage = len(tracks) / frame_count
        missing_frames = frame_count - len(detected)
        jitter_px = self._jitter(tracks)
        trajectory_smoothness = round(float(max(0.0, min(1.0, 1.0 - jitter_px / 80.0))), 3)
        max_missing_gap = self._max_missing_gap(detections)
        jump_rejections = sum(1 for point in tracks if point.get("rejected_jump"))

        jitter_penalty = min(0.35, jitter_px / 90.0)
        missing_penalty = min(0.25, max_missing_gap / max(1, frame_count) * 1.2)
        jump_penalty = min(0.15, jump_rejections * 0.025)
        score = (
            detection_rate * 0.34
            + mean_confidence * 0.34
            + min(1.0, track_coverage) * 0.22
            + max(0.0, 1.0 - jitter_penalty) * 0.10
            - missing_penalty
            - jump_penalty
        )
        score = round(float(max(0.0, min(1.0, score))), 3)

        warnings: list[str] = []
        if detection_rate < 0.55:
            warnings.append("Ball detection rate is low; train with more match footage or improve camera angle.")
        if mean_confidence < 0.45:
            warnings.append("Detector confidence is low; results should be treated as testing evidence only.")
        if jitter_px > 24:
            warnings.append("Tracking jitter is high; verify FPS, shutter speed, and calibration.")
        if missing_frames > frame_count * 0.35:
            warnings.append("Many frames are missing detections; Kalman prediction is filling gaps.")
        if max_missing_gap > 5:
            warnings.append("A missing-detection gap is too long for reliable DRS ball tracking.")
        if jump_rejections:
            warnings.append(f"{jump_rejections} implausible detection jump(s) were rejected.")

        if score >= 0.78 and not warnings:
            reliability = "high"
        elif score >= 0.58:
            reliability = "medium"
        else:
            reliability = "low"

        return TrackingQuality(
            detection_rate=round(detection_rate, 3),
            detection_coverage=round(detection_coverage, 3),
            mean_confidence=round(mean_confidence, 3),
            track_coverage=round(track_coverage, 3),
            jitter_px=round(jitter_px, 2),
            trajectory_smoothness=trajectory_smoothness,
            missing_frames=missing_frames,
            max_missing_gap=max_missing_gap,
            jump_rejections=jump_rejections,
            reliability=reliability,
            score=score,
            warnings=warnings,
        )

    def _jitter(self, tracks: list[dict[str, Any]]) -> float:
        if len(tracks) < 4:
            return 0.0
        points = np.array([[float(item["x"]), float(item["y"])] for item in tracks], dtype=float)
        velocities = np.diff(points, axis=0)
        accelerations = np.diff(velocities, axis=0)
        if len(accelerations) == 0:
            return 0.0
        return float(np.median(np.linalg.norm(accelerations, axis=1)))

    def _max_missing_gap(self, detections: list[dict[str, Any]]) -> int:
        longest = 0
        current = 0
        for item in detections:
            if item.get("confidence", 0.0) > 0:
                longest = max(longest, current)
                current = 0
            else:
                current += 1
        return max(longest, current)
