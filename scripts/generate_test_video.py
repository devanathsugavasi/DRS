#!/usr/bin/env python
"""Generate synthetic cricket delivery videos for offline DRS pipeline checks.

These videos are useful for verifying uploads, overlays, tracking animations, replay
controls, and dual-camera synchronization. They are not accuracy validation data.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class DeliveryConfig:
    duration_seconds: float
    fps: int
    width: int
    height: int
    ball_color: str
    camera_angle: str
    seed: int


def _ball_bgr(ball_color: str) -> tuple[int, int, int]:
    if ball_color.lower() == "white":
        return (245, 245, 235)
    if ball_color.lower() == "pink":
        return (185, 55, 235)
    return (32, 32, 210)


def _pitch_points(width: int, height: int, angle: str) -> np.ndarray:
    if angle == "side":
        return np.array(
            [
                [int(width * 0.10), int(height * 0.70)],
                [int(width * 0.92), int(height * 0.70)],
                [int(width * 0.78), int(height * 0.42)],
                [int(width * 0.20), int(height * 0.42)],
            ],
            dtype=np.int32,
        )
    if angle == "end":
        return np.array(
            [
                [int(width * 0.34), int(height * 0.90)],
                [int(width * 0.66), int(height * 0.90)],
                [int(width * 0.58), int(height * 0.22)],
                [int(width * 0.42), int(height * 0.22)],
            ],
            dtype=np.int32,
        )
    return np.array(
        [
            [int(width * 0.16), int(height * 0.86)],
            [int(width * 0.86), int(height * 0.80)],
            [int(width * 0.66), int(height * 0.22)],
            [int(width * 0.34), int(height * 0.24)],
        ],
        dtype=np.int32,
    )


def _draw_field(frame: np.ndarray, frame_id: int, total_frames: int, config: DeliveryConfig) -> None:
    height, width = frame.shape[:2]
    grass_a = np.array([18, 92, 44], dtype=np.uint8)
    grass_b = np.array([24, 112, 52], dtype=np.uint8)
    frame[:, :] = grass_a
    stripe_height = max(18, height // 22)
    for y in range(0, height, stripe_height):
        if (y // stripe_height) % 2 == 0:
            frame[y : y + stripe_height, :] = grass_b

    pitch = _pitch_points(width, height, config.camera_angle)
    cv2.fillConvexPoly(frame, pitch, (108, 126, 88))
    cv2.polylines(frame, [pitch], True, (185, 205, 165), 2, cv2.LINE_AA)

    cv2.line(frame, (int(width * 0.20), int(height * 0.58)), (int(width * 0.88), int(height * 0.56)), (210, 230, 210), 2)
    cv2.line(frame, (int(width * 0.28), int(height * 0.47)), (int(width * 0.72), int(height * 0.47)), (210, 230, 210), 2)

    cv2.putText(
        frame,
        f"SYNTHETIC DRS CHECK | CAM {config.camera_angle.upper()} | frame {frame_id}/{total_frames}",
        (22, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (230, 245, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Pipeline demo only - not validation footage",
        (22, height - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (60, 255, 165),
        2,
        cv2.LINE_AA,
    )


def _trajectory_position(t: float, width: int, height: int, angle: str) -> tuple[int, int, float]:
    release_x = width * 0.12
    release_y = height * 0.28
    bounce_t = 0.58
    impact_t = 0.78

    if angle == "end":
        x = width * (0.50 + 0.04 * math.sin(t * math.pi * 1.4))
        y = height * (0.18 + 0.70 * t)
        apparent_radius = 5 + 9 * t
    elif angle == "side":
        x = release_x + width * 0.76 * t
        y = release_y + height * (0.52 * abs(math.sin(t * math.pi * 0.76)) + 0.10 * t)
        if t > bounce_t:
            y -= height * 0.22 * (t - bounce_t)
        apparent_radius = 7
    else:
        x = release_x + width * 0.74 * t
        y = release_y + height * (0.64 * t + 0.20 * math.sin(t * math.pi))
        if t > impact_t:
            y -= height * 0.14 * (t - impact_t)
        apparent_radius = 6 + 2 * t

    return int(x), int(y), apparent_radius


def _draw_stumps(frame: np.ndarray, angle: str) -> None:
    height, width = frame.shape[:2]
    if angle == "end":
        base_x, base_y, stump_h, gap = int(width * 0.50), int(height * 0.78), int(height * 0.24), int(width * 0.035)
    elif angle == "side":
        base_x, base_y, stump_h, gap = int(width * 0.86), int(height * 0.61), int(height * 0.20), int(width * 0.010)
    else:
        base_x, base_y, stump_h, gap = int(width * 0.78), int(height * 0.60), int(height * 0.18), int(width * 0.014)

    for offset in (-gap, 0, gap):
        cv2.line(frame, (base_x + offset, base_y), (base_x + offset, base_y - stump_h), (242, 234, 202), 5, cv2.LINE_AA)
    cv2.line(frame, (base_x - gap - 7, base_y - stump_h), (base_x + gap + 7, base_y - stump_h - 4), (242, 234, 202), 3, cv2.LINE_AA)


def _draw_player_removed_overlay(frame: np.ndarray, angle: str) -> None:
    height, width = frame.shape[:2]
    if angle == "end":
        center = (int(width * 0.50), int(height * 0.64))
        size = (int(width * 0.18), int(height * 0.34))
    else:
        center = (int(width * 0.70), int(height * 0.58))
        size = (int(width * 0.16), int(height * 0.26))
    overlay = frame.copy()
    cv2.ellipse(overlay, center, size, 0, 0, 360, (38, 92, 65), -1)
    cv2.addWeighted(overlay, 0.34, frame, 0.66, 0, frame)
    cv2.putText(
        frame,
        "batter removed overlay",
        (center[0] - size[0], center[1] - size[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (60, 255, 165),
        1,
        cv2.LINE_AA,
    )


def _draw_trajectory(frame: np.ndarray, frame_id: int, total_frames: int, config: DeliveryConfig) -> dict[str, float]:
    height, width = frame.shape[:2]
    progress = frame_id / max(1, total_frames - 1)
    points: list[tuple[int, int]] = []
    for idx in range(max(2, frame_id - 42), frame_id + 1):
        pt = idx / max(1, total_frames - 1)
        x, y, _ = _trajectory_position(pt, width, height, config.camera_angle)
        points.append((x, y))

    for p0, p1 in zip(points, points[1:]):
        cv2.line(frame, p0, p1, (58, 255, 154), 3, cv2.LINE_AA)

    x, y, radius = _trajectory_position(progress, width, height, config.camera_angle)
    ball_bgr = _ball_bgr(config.ball_color)
    cv2.circle(frame, (x, y), int(radius + 10), (40, 255, 170), 1, cv2.LINE_AA)
    cv2.circle(frame, (x, y), int(radius), ball_bgr, -1, cv2.LINE_AA)
    cv2.circle(frame, (x, y), max(2, int(radius * 0.35)), (255, 255, 255), -1, cv2.LINE_AA)

    bounce_x, bounce_y, _ = _trajectory_position(0.58, width, height, config.camera_angle)
    impact_x, impact_y, _ = _trajectory_position(0.78, width, height, config.camera_angle)
    cv2.drawMarker(frame, (bounce_x, bounce_y), (35, 170, 255), cv2.MARKER_TILTED_CROSS, 24, 2, cv2.LINE_AA)
    cv2.putText(frame, "bounce", (bounce_x + 12, bounce_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (35, 170, 255), 1, cv2.LINE_AA)
    cv2.circle(frame, (impact_x, impact_y), 15, (74, 148, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "impact", (impact_x + 14, impact_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (74, 148, 255), 1, cv2.LINE_AA)

    speed_kmh = 132.0 - 18.0 * progress + 3.0 * math.sin(progress * math.pi * 2)
    cv2.putText(frame, f"speed est {speed_kmh:.1f} km/h", (22, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (60, 255, 165), 2, cv2.LINE_AA)
    return {"frame": frame_id, "x": x, "y": y, "confidence": 1.0, "speed_kmh": round(speed_kmh, 2)}


def generate_test_delivery(output_path: Path, config: DeliveryConfig) -> Path:
    """Generate a synthetic cricket delivery video with DRS-style overlays."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_frames = int(config.duration_seconds * config.fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, config.fps, (config.width, config.height))
    rng = np.random.default_rng(config.seed)
    tracking_rows: list[dict[str, float]] = []

    print(f"Generating {output_path} ({config.camera_angle}, {config.width}x{config.height}, {config.fps} FPS)")
    for frame_id in range(total_frames):
        frame = np.zeros((config.height, config.width, 3), dtype=np.uint8)
        _draw_field(frame, frame_id, total_frames, config)
        _draw_stumps(frame, config.camera_angle)
        _draw_player_removed_overlay(frame, config.camera_angle)

        noise = rng.integers(0, 10, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)
        tracking_rows.append(_draw_trajectory(frame, frame_id, total_frames, config))

        writer.write(frame)
        if (frame_id + 1) % max(1, total_frames // 10) == 0:
            print(f"  progress {int((frame_id + 1) / total_frames * 100)}%")

    writer.release()
    metadata_path = output_path.with_suffix(".tracking.json")
    metadata_path.write_text(json.dumps({"synthetic": True, "tracking": tracking_rows}, indent=2), encoding="utf-8")
    print(f"Created video: {output_path}")
    print(f"Created synthetic tracking metadata: {metadata_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic cricket delivery test videos")
    parser.add_argument("--output", default="data/testing/demo_delivery.mp4", help="Output video file")
    parser.add_argument("--duration", type=float, default=6, help="Video duration in seconds")
    parser.add_argument("--fps", type=int, default=60, help="Frames per second")
    parser.add_argument("--width", type=int, default=1280, help="Video width in pixels")
    parser.add_argument("--height", type=int, default=720, help="Video height in pixels")
    parser.add_argument("--ball-color", choices=["red", "white", "pink"], default="red")
    parser.add_argument("--camera-angle", choices=["broadcast", "side", "end"], default="broadcast")
    parser.add_argument("--dual", action="store_true", help="Generate two synchronized camera angles")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.dual:
        base = Path(args.output)
        for suffix, angle in (("cam0", "broadcast"), ("cam1", "side")):
            generate_test_delivery(
                base.with_name(f"{base.stem}_{suffix}{base.suffix}"),
                DeliveryConfig(args.duration, args.fps, args.width, args.height, args.ball_color, angle, args.seed),
            )
        return

    generate_test_delivery(
        Path(args.output),
        DeliveryConfig(args.duration, args.fps, args.width, args.height, args.ball_color, args.camera_angle, args.seed),
    )


if __name__ == "__main__":
    main()
