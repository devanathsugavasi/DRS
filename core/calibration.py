"""Multi-camera calibration and pixel-to-world mapping utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config.settings import CALIBRATION_DIR, CHECKERBOARD_SIZE, SQUARE_SIZE_MM
from utils.helpers import save_json


@dataclass(slots=True)
class CameraCalibration:
    camera_id: int
    image_size: tuple[int, int]
    rms_error: float
    camera_matrix: list[list[float]]
    distortion_coeffs: list[list[float]]
    rotation_vectors: list[list[float]]
    translation_vectors: list[list[float]]
    homography: Optional[list[list[float]]] = None


class MultiCameraCalibrator:
    """Calibrates cameras from checkerboard images and estimates pitch-plane homography."""

    def __init__(
        self,
        checkerboard_size: tuple[int, int] = CHECKERBOARD_SIZE,
        square_size_mm: float = SQUARE_SIZE_MM,
    ) -> None:
        self.checkerboard_size = checkerboard_size
        self.square_size_mm = square_size_mm
        self.object_template = self._make_object_template()

    def calibrate_camera(self, camera_id: int, image_paths: list[Path]) -> CameraCalibration:
        object_points: list[np.ndarray] = []
        image_points: list[np.ndarray] = []
        image_size: Optional[tuple[int, int]] = None

        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_size = (gray.shape[1], gray.shape[0])
            found, corners = cv2.findChessboardCorners(gray, self.checkerboard_size)
            if not found:
                continue
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
            refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            object_points.append(self.object_template)
            image_points.append(refined)

        if not object_points or image_size is None:
            raise ValueError(f"No checkerboard detections found for camera {camera_id}")

        rms, camera_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, image_size, None, None
        )
        calibration = CameraCalibration(
            camera_id=camera_id,
            image_size=image_size,
            rms_error=float(rms),
            camera_matrix=camera_matrix.tolist(),
            distortion_coeffs=dist.tolist(),
            rotation_vectors=[item.tolist() for item in rvecs],
            translation_vectors=[item.tolist() for item in tvecs],
        )
        return calibration

    def set_pitch_homography(
        self,
        calibration: CameraCalibration,
        image_points: np.ndarray,
        world_points_m: np.ndarray,
    ) -> CameraCalibration:
        homography, _ = cv2.findHomography(image_points.astype(np.float32), world_points_m.astype(np.float32))
        calibration.homography = homography.tolist()
        return calibration

    def undistort(self, frame: np.ndarray, calibration: CameraCalibration) -> np.ndarray:
        return cv2.undistort(
            frame,
            np.asarray(calibration.camera_matrix, dtype=np.float32),
            np.asarray(calibration.distortion_coeffs, dtype=np.float32),
        )

    def pixel_to_world(self, x: float, y: float, calibration: CameraCalibration) -> tuple[float, float]:
        if calibration.homography is None:
            raise ValueError("Calibration has no pitch-plane homography")
        point = np.array([[[x, y]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(point, np.asarray(calibration.homography, dtype=np.float32))
        return float(mapped[0, 0, 0]), float(mapped[0, 0, 1])

    def save(self, calibrations: list[CameraCalibration], path: Path | None = None) -> Path:
        path = path or CALIBRATION_DIR / "camera_calibration.json"
        return save_json([asdict(item) for item in calibrations], path)

    def draw_reprojection(
        self,
        frame: np.ndarray,
        calibration: CameraCalibration,
        object_points: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> np.ndarray:
        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            np.asarray(calibration.camera_matrix, dtype=np.float32),
            np.asarray(calibration.distortion_coeffs, dtype=np.float32),
        )
        for point in projected.reshape(-1, 2):
            cv2.circle(frame, tuple(point.astype(int)), 4, (0, 255, 255), -1, cv2.LINE_AA)
        return frame

    def _make_object_template(self) -> np.ndarray:
        cols, rows = self.checkerboard_size
        points = np.zeros((rows * cols, 3), np.float32)
        points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        points *= self.square_size_mm
        return points
