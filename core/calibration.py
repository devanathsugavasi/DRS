"""Multi-camera calibration and pixel-to-world mapping utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config.settings import CALIBRATION_DIR, CHECKERBOARD_SIZE, SQUARE_SIZE_MM
from utils.helpers import save_json
from utils.helpers import load_json


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

    def pixel_to_pitch_mm(self, camera_id: int, px: float, py: float, calibration: CameraCalibration) -> tuple[float, float]:
        if calibration.camera_id != camera_id:
            raise ValueError(f"Calibration belongs to camera {calibration.camera_id}, not {camera_id}")
        x, y = self.pixel_to_world(px, py, calibration)
        return x * 1000.0, y * 1000.0

    def triangulate_3d(
        self,
        observations: dict[int, tuple[float, float]],
        calibrations: dict[int, CameraCalibration],
    ) -> tuple[float, float, float]:
        if len(observations) < 2:
            raise ValueError("At least two calibrated camera observations are required")
        camera_ids = list(observations.keys())[:2]
        projections = []
        points = []
        for camera_id in camera_ids:
            calibration = calibrations[camera_id]
            camera_matrix = np.asarray(calibration.camera_matrix, dtype=np.float64)
            rvec = np.asarray(calibration.rotation_vectors[0], dtype=np.float64)
            tvec = np.asarray(calibration.translation_vectors[0], dtype=np.float64)
            rotation, _ = cv2.Rodrigues(rvec)
            projection = camera_matrix @ np.hstack([rotation, tvec.reshape(3, 1)])
            projections.append(projection)
            points.append(np.asarray(observations[camera_id], dtype=np.float64).reshape(2, 1))
        point_h = cv2.triangulatePoints(projections[0], projections[1], points[0], points[1])
        denom = float(point_h[3, 0]) if abs(float(point_h[3, 0])) > 1e-9 else 1e-9
        point = point_h[:3, 0] / denom
        return float(point[0]), float(point[1]), float(point[2])

    def homography_validation_error_cm(
        self,
        calibration: CameraCalibration,
        image_points: np.ndarray,
        world_points_m: np.ndarray,
    ) -> float:
        if calibration.homography is None:
            raise ValueError("Calibration has no pitch-plane homography")
        projected = cv2.perspectiveTransform(
            image_points.reshape(-1, 1, 2).astype(np.float32),
            np.asarray(calibration.homography, dtype=np.float32),
        ).reshape(-1, 2)
        error_m = np.linalg.norm(projected - world_points_m.reshape(-1, 2), axis=1)
        return float(np.mean(error_m) * 100.0)

    def save(self, calibrations: list[CameraCalibration], path: Path | None = None) -> Path:
        path = path or CALIBRATION_DIR / "camera_calibration.json"
        return save_json([asdict(item) for item in calibrations], path)

    def save_per_camera(self, calibration: CameraCalibration) -> Path:
        return save_json(asdict(calibration), CALIBRATION_DIR / f"calibration_{calibration.camera_id}.json")

    def load_per_camera(self, camera_id: int) -> CameraCalibration:
        data = load_json(CALIBRATION_DIR / f"calibration_{camera_id}.json")
        return CameraCalibration(**data)

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
