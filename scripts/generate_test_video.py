#!/usr/bin/env python
"""Generate synthetic cricket delivery test videos."""

import argparse
from pathlib import Path
import cv2
import numpy as np


def generate_test_delivery(output_path: Path, duration_seconds: float = 10, fps: int = 30, width: int = 1280, height: int = 720):
    """Generate a synthetic cricket delivery video with moving ball."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_frames = int(duration_seconds * fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    print(f"Generating {duration_seconds}s test video at {fps} FPS ({total_frames} frames)...")
    print(f"Resolution: {width}x{height}")
    print(f"Output: {output_path}")
    
    for frame_id in range(total_frames):
        # Create frame with gradient background (cricket field green)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = [34, 139, 34]  # Dark green (BGR)
        
        # Add some texture/noise to look more realistic
        noise = np.random.randint(0, 20, (height, width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        # Simulate ball trajectory (parabolic motion from left to right)
        t = frame_id / total_frames  # 0 to 1
        ball_x = int(100 + width * 0.7 * t)
        ball_y = int(height * 0.3 + 300 * np.sin(t * np.pi))
        ball_radius = 8
        
        # Draw ball (red/dark red like cricket ball)
        cv2.circle(frame, (ball_x, ball_y), ball_radius, (0, 0, 139), -1)  # Dark red
        cv2.circle(frame, (ball_x, ball_y), ball_radius, (0, 0, 255), 2)   # Bright red outline
        
        # Draw stumps on right side
        stump_x = width - 150
        stump_y = int(height * 0.6)
        stump_width = 3
        stump_height = 100
        
        # Three vertical stumps
        for i in range(3):
            x = stump_x + i * 15
            cv2.line(frame, (x, stump_y - stump_height // 2), (x, stump_y + stump_height // 2), (255, 255, 255), stump_width)
        
        # Draw bails (horizontal bars)
        cv2.line(frame, (stump_x - 5, stump_y - stump_height // 2 - 5), (stump_x + 40, stump_y - stump_height // 2 - 5), (255, 255, 255), 2)
        cv2.line(frame, (stump_x - 5, stump_y - stump_height // 2 + 5), (stump_x + 40, stump_y - stump_height // 2 + 5), (255, 255, 255), 2)
        
        # Add frame info
        cv2.putText(frame, f"Frame: {frame_id} / {total_frames}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {frame_id / fps:.2f}s", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Ball speed indicator
        speed_indicator = int(100 + 50 * np.sin(t * np.pi * 2))
        cv2.putText(frame, f"Speed: ~{speed_indicator} km/h", (20, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        writer.write(frame)
        
        # Progress
        if (frame_id + 1) % max(1, total_frames // 10) == 0:
            progress = int((frame_id + 1) / total_frames * 100)
            print(f"  Progress: {progress}%")
    
    writer.release()
    print(f"✅ Test video created: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic cricket delivery test videos")
    parser.add_argument("--output", default="test_delivery.mp4", help="Output video file")
    parser.add_argument("--duration", type=float, default=10, help="Video duration in seconds")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--width", type=int, default=1280, help="Video width in pixels")
    parser.add_argument("--height", type=int, default=720, help="Video height in pixels")
    parser.add_argument("--count", type=int, default=1, help="Number of videos to generate")
    
    args = parser.parse_args()
    
    for i in range(args.count):
        if args.count > 1:
            output = Path(args.output).stem + f"_{i+1}.mp4"
        else:
            output = args.output
        
        generate_test_delivery(
            Path(output),
            duration_seconds=args.duration,
            fps=args.fps,
            width=args.width,
            height=args.height
        )


if __name__ == "__main__":
    main()
