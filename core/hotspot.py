"""HotSpot-style contact visualization using optical-flow heatmaps."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class HotSpotResult:
    contact_detected: bool
    contact_region: tuple[int, int, int, int] | None
    heatmap: np.ndarray
    confidence: float
    reason: str


class HotSpotAnalyzer:
    """Approximates HotSpot visuals from motion energy when thermal data is unavailable."""

    def analyze_contact(self, frames: list[np.ndarray], contact_frame_idx: int) -> HotSpotResult:
        if len(frames) < 2 or not (0 <= contact_frame_idx < len(frames)):
            return HotSpotResult(False, None, np.empty((0, 0, 3), dtype=np.uint8), 0.0, "Need at least two frames around contact.")
        idx0 = max(0, contact_frame_idx - 1)
        idx1 = min(len(frames) - 1, contact_frame_idx + 1)
        prev = cv2.cvtColor(frames[idx0], cv2.COLOR_BGR2GRAY)
        curr = cv2.cvtColor(frames[idx1], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude = cv2.normalize(np.linalg.norm(flow, axis=2), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(magnitude, cv2.COLORMAP_JET)
        threshold = float(np.mean(magnitude) + 2.5 * np.std(magnitude))
        mask = (magnitude > threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return HotSpotResult(False, None, heatmap, 0.0, "No concentrated contact motion detected.")
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        confidence = min(1.0, cv2.contourArea(contour) / max(1.0, frames[0].shape[0] * frames[0].shape[1] * 0.02))
        cv2.ellipse(heatmap, (x + w // 2, y + h // 2), (max(8, w // 2), max(8, h // 2)), 0, 0, 360, (255, 255, 255), 2)
        return HotSpotResult(confidence > 0.15, (x, y, x + w, y + h), heatmap, float(confidence), "Optical-flow contact proxy; not real thermal imaging.")
