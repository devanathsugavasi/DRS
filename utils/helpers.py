"""Shared helpers for video annotation and structured exports."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


def now_ms() -> float:
    return time.time() * 1000.0


def timestamp_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def draw_timestamp(frame: np.ndarray, timestamp_ms: float, camera_id: int) -> np.ndarray:
    out = frame.copy()
    text = f"CAM {camera_id} | {timestamp_ms / 1000.0:.3f}s"
    cv2.putText(out, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def draw_bounding_box(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int] = (0, 220, 255),
) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        y0 = max(0, y1 - th - 8)
        cv2.rectangle(frame, (x1, y0), (x1 + tw + 8, y1), color, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)
    return frame


def draw_trajectory(
    frame: np.ndarray,
    points: Sequence[tuple[int, int]],
    color: tuple[int, int, int] = (0, 165, 255),
) -> np.ndarray:
    if len(points) < 2:
        return frame
    total = max(1, len(points) - 1)
    for idx in range(1, len(points)):
        alpha = idx / total
        faded = tuple(max(35, int(channel * alpha)) for channel in color)
        cv2.line(frame, points[idx - 1], points[idx], faded, 2, cv2.LINE_AA)
    return frame


def resize_keep_aspect(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == width:
        return frame
    ratio = width / float(w)
    return cv2.resize(frame, (width, int(h * ratio)), interpolation=cv2.INTER_AREA)


def save_json(data: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def save_csv(rows: Iterable[Mapping[str, Any]], path: Path) -> Path:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
