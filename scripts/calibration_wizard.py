"""Interactive pitch calibration wizard.

Usage:
    python scripts/calibration_wizard.py --camera 0
    python scripts/calibration_wizard.py --image data/calibration/frame.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from core.pitch_calibration import ManualPitchCalibrator, CALIBRATION_DIR

MARKER_NAMES = [
    "Off Stump",
    "Middle Stump",
    "Leg Stump",
    "Bowling Crease Center",
    "Popping Crease Center",
]

MARKER_KEYS = [
    "off_stump",
    "middle_stump",
    "leg_stump",
    "bowling_crease",
    "popping_crease",
]

MARKER_COLORS = [
    (0, 0, 255),    # red
    (0, 255, 0),    # green
    (255, 0, 0),    # blue
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
]


class CalibrationWizard:
    def __init__(self, camera_id: int | None, image_path: Path | None):
        self.camera_id = camera_id
        self.image_path = image_path
        self.points: list[tuple[float, float]] = []
        self.frame: np.ndarray | None = None
        self.original_frame: np.ndarray | None = None
        self.calibrator = ManualPitchCalibrator()
        self.done = False

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 5:
            self.points.append((float(x), float(y)))
            idx = len(self.points) - 1
            print(f"  [{idx+1}/5] {MARKER_NAMES[idx]}: ({x}, {y})")
            if len(self.points) == 5:
                self._calibrate()

    def _calibrate(self):
        markers = {
            key: {"x": pt[0], "y": pt[1]}
            for key, pt in zip(MARKER_KEYS, self.points)
        }
        h, w = self.original_frame.shape[:2]
        cam_id = self.camera_id if self.camera_id is not None else 0
        try:
            profile = self.calibrator.save_profile(cam_id, markers, (w, h))
            print(f"\n✓ Calibration saved for camera {cam_id}")
            print(f"  Homography error: {profile.homography_error_cm:.3f} cm")
            print(f"  Profile dir: {CALIBRATION_DIR}")
            # Test pixel→world on all 5 points
            print("\n  Verification (pixel → world mm):")
            for name, (px, py) in zip(MARKER_NAMES, self.points):
                result = self.calibrator.pixel_to_pitch_mm(cam_id, px, py)
                if result:
                    print(f"    {name}: ({px:.0f}, {py:.0f}) → ({result[0]:.1f} mm, {result[1]:.1f} mm)")
            self.done = True
        except Exception as exc:
            print(f"\n✗ Calibration failed: {exc}")
            self.points.clear()

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        display = frame.copy()
        # Draw placed points
        for i, (px, py) in enumerate(self.points):
            color = MARKER_COLORS[i]
            cv2.circle(display, (int(px), int(py)), 8, color, -1, cv2.LINE_AA)
            cv2.circle(display, (int(px), int(py)), 10, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display, MARKER_NAMES[i], (int(px) + 14, int(py) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        # Draw lines between stumps if 3+ points
        if len(self.points) >= 3:
            pts = [tuple(map(int, p)) for p in self.points[:3]]
            cv2.line(display, pts[0], pts[2], (255, 255, 255), 2, cv2.LINE_AA)
        # Draw crease line if 4+ points
        if len(self.points) >= 5:
            bc = tuple(map(int, self.points[3]))
            pc = tuple(map(int, self.points[4]))
            cv2.line(display, bc, pc, (200, 200, 200), 1, cv2.LINE_AA)
        # Instructions
        if len(self.points) < 5:
            next_marker = MARKER_NAMES[len(self.points)]
            cv2.putText(display, f"Click: {next_marker} ({len(self.points)+1}/5)",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        elif self.done:
            cv2.putText(display, "CALIBRATED - Press Q to quit, R to redo",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, "R=reset  Q=quit", (20, display.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        return display

    def run(self):
        if self.image_path:
            self.original_frame = cv2.imread(str(self.image_path))
            if self.original_frame is None:
                print(f"Error: cannot read {self.image_path}")
                return
        else:
            cap = cv2.VideoCapture(self.camera_id or 0)
            if not cap.isOpened():
                print(f"Error: cannot open camera {self.camera_id}")
                return
            ok, self.original_frame = cap.read()
            cap.release()
            if not ok:
                print("Error: cannot read frame from camera")
                return

        print("\n=== DRS Pitch Calibration Wizard ===")
        print(f"Frame size: {self.original_frame.shape[1]}x{self.original_frame.shape[0]}")
        print("Click the 5 pitch markers in order:\n")
        for i, name in enumerate(MARKER_NAMES):
            print(f"  {i+1}. {name}")
        print()

        cv2.namedWindow("DRS Calibration", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("DRS Calibration", self._mouse_callback)

        while True:
            display = self._draw_overlay(self.original_frame)
            cv2.imshow("DRS Calibration", display)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.points.clear()
                self.done = False
                print("\nReset — click markers again.")

        cv2.destroyAllWindows()
        if self.done:
            print("\nCalibration complete.")
        else:
            print("\nCalibration cancelled.")


def main():
    parser = argparse.ArgumentParser(description="Interactive DRS pitch calibration")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--camera", type=int, help="Camera ID to capture from")
    group.add_argument("--image", type=Path, help="Path to a frame image")
    args = parser.parse_args()
    wizard = CalibrationWizard(args.camera, args.image)
    wizard.run()


if __name__ == "__main__":
    main()
