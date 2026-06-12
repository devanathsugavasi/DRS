"""Multi-camera calibration and pixel-to-world mapping utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config.settings import CALIBRATION_DIR, CHECKERBOARD_SIZE, SQUARE_SIZE_MM
from utils.helpers import save_json
from utils.helpers import load_json


PROFILE_DIR = Path("config/calibration_profiles")
PITCH_WORLD_POINTS = np.array(
    [
        [-1.32, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.32, 0.0, 0.0],
        [-1.32, 1.22, 0.0],
        [0.0, 1.22, 0.0],
        [1.32, 1.22, 0.0],
        [-0.1143, 20.12, 0.711],
        [0.0, 20.12, 0.711],
        [0.1143, 20.12, 0.711],
    ],
    dtype=np.float32,
)
PITCH_POINT_LABELS = [
    "Bowling crease - left edge",
    "Bowling crease - center",
    "Bowling crease - right edge",
    "Popping crease - left edge",
    "Popping crease - center",
    "Popping crease - right edge",
    "Striker stumps - left top",
    "Striker stumps - center top",
    "Striker stumps - right top",
]


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


class PitchCalibrator:
    """Single-camera pitch calibration using nine clicked pitch landmarks."""

    def __init__(self, profile_dir: Path = PROFILE_DIR) -> None:
        self.profile_dir = profile_dir
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self.profile: dict | None = None
        self.image_points: list[list[float]] = []
        self.world_points = PITCH_WORLD_POINTS.copy()

    def calibrate_interactive(
        self,
        camera_frame: np.ndarray,
        camera_id: int,
        profile_name: str,
        ground_name: str,
    ) -> dict:
        """Click nine landmarks on a frame, solve camera pose, and save on ENTER."""
        if camera_frame is None or camera_frame.size == 0:
            raise ValueError("camera_frame is empty")

        self.image_points = []
        display = camera_frame.copy()
        window_name = "DRS pitch calibration"

        def redraw() -> None:
            nonlocal display
            display = camera_frame.copy()
            instruction_index = min(len(self.image_points), len(PITCH_POINT_LABELS) - 1)
            cv2.putText(
                display,
                f"Click {len(self.image_points) + 1}/9: {PITCH_POINT_LABELS[instruction_index]}",
                (20, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            for idx, (x, y) in enumerate(self.image_points, start=1):
                cv2.circle(display, (int(x), int(y)), 7, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.putText(display, str(idx), (int(x) + 9, int(y) - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.imshow(window_name, display)

        def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
            if event == cv2.EVENT_LBUTTONDOWN and len(self.image_points) < 9:
                self.image_points.append([float(x), float(y)])
                redraw()

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, on_mouse)
        redraw()

        while True:
            key = cv2.waitKey(30) & 0xFF
            if key in {13, 10} and len(self.image_points) == 9:
                profile = self._solve_profile(camera_frame, camera_id, profile_name, ground_name, self.image_points)
                overlay = self.verify_calibration(camera_frame, profile)["overlay_frame"]
                cv2.imshow(window_name, overlay)
                print(f"Calibration RMS error: {profile['rms_error_px']:.2f}px - {self._quality_label(profile['rms_error_px'])}")
                confirm_key = cv2.waitKey(0) & 0xFF
                if confirm_key in {13, 10}:
                    self.profile = profile
                    self.save_profile(profile_name, ground_name, camera_id)
                    cv2.destroyWindow(window_name)
                    return profile
                if confirm_key in {ord("r"), ord("R")}:
                    self.image_points = []
                    redraw()
            elif key in {ord("r"), ord("R")}:
                self.image_points = []
                redraw()
            elif key == 27:
                cv2.destroyWindow(window_name)
                raise KeyboardInterrupt("Calibration cancelled")

    def save_profile(self, profile_name: str, ground_name: str, camera_id: int | str) -> Path:
        """Save the most recently solved calibration profile."""
        if self.profile is None:
            raise ValueError("No calibration profile has been solved")
        safe_ground = self._safe_name(ground_name)
        path = self.profile_dir / f"{camera_id}_{safe_ground}.json"
        return save_json(self.profile, path)

    def load_profile(self, camera_id: int | str, ground_name: str | None = None) -> dict | None:
        """Load a calibration profile for a camera, newest first if ground is omitted."""
        if ground_name:
            path = self.profile_dir / f"{camera_id}_{self._safe_name(ground_name)}.json"
            if not path.exists():
                return None
            self.profile = load_json(path)
            return self.profile
        matches = sorted(self.profile_dir.glob(f"{camera_id}_*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
        if not matches:
            return None
        self.profile = load_json(matches[0])
        return self.profile

    def pixel_to_world(self, pixel_x: float, pixel_y: float, ground_z: float = 0.0) -> tuple[float, float, float]:
        """Project a pixel ray to the configured world Z plane."""
        profile = self._require_profile()
        camera_matrix = np.asarray(profile["camera_matrix"], dtype=np.float64)
        rvec = np.asarray(profile["rvec"], dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(profile["tvec"], dtype=np.float64).reshape(3, 1)
        rotation, _ = cv2.Rodrigues(rvec)
        camera_center = -rotation.T @ tvec
        pixel = np.array([pixel_x, pixel_y, 1.0], dtype=np.float64).reshape(3, 1)
        ray_camera = np.linalg.inv(camera_matrix) @ pixel
        ray_world = rotation.T @ ray_camera
        denom = float(ray_world[2, 0])
        if abs(denom) < 1e-9:
            raise ValueError("Pixel ray is parallel to requested world plane")
        scale = (ground_z - float(camera_center[2, 0])) / denom
        world = camera_center + scale * ray_world
        return float(world[0, 0]), float(world[1, 0]), float(ground_z)

    def world_to_pixel(self, world_x: float, world_y: float, world_z: float) -> tuple[int, int]:
        """Project a 3D world coordinate back to image pixels."""
        profile = self._require_profile()
        projected, _ = cv2.projectPoints(
            np.array([[world_x, world_y, world_z]], dtype=np.float32),
            np.asarray(profile["rvec"], dtype=np.float64).reshape(3, 1),
            np.asarray(profile["tvec"], dtype=np.float64).reshape(3, 1),
            np.asarray(profile["camera_matrix"], dtype=np.float64),
            np.asarray(profile["dist_coeffs"], dtype=np.float64),
        )
        x, y = projected.reshape(-1, 2)[0]
        return int(round(float(x))), int(round(float(y)))

    def verify_calibration(self, frame: np.ndarray, profile: dict | None = None) -> dict:
        """Draw reprojected landmarks and return RMS validity metadata."""
        profile = profile or self._require_profile()
        overlay = frame.copy()
        image_points = np.asarray(profile["image_points"], dtype=np.float32)
        projected, _ = cv2.projectPoints(
            np.asarray(profile["world_points"], dtype=np.float32),
            np.asarray(profile["rvec"], dtype=np.float64).reshape(3, 1),
            np.asarray(profile["tvec"], dtype=np.float64).reshape(3, 1),
            np.asarray(profile["camera_matrix"], dtype=np.float64),
            np.asarray(profile["dist_coeffs"], dtype=np.float64),
        )
        projected_points = projected.reshape(-1, 2)
        for idx, (clicked, reproj) in enumerate(zip(image_points, projected_points), start=1):
            cv2.circle(overlay, tuple(clicked.astype(int)), 6, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(overlay, tuple(reproj.astype(int)), 8, (0, 180, 0), 2, cv2.LINE_AA)
            cv2.putText(overlay, str(idx), tuple(reproj.astype(int) + np.array([9, -9])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 2)
        rms = float(profile.get("rms_error_px", 999.0))
        return {"rms_error": rms, "is_valid": rms < 6.0, "overlay_frame": overlay}

    def _solve_profile(
        self,
        frame: np.ndarray,
        camera_id: int | str,
        profile_name: str,
        ground_name: str,
        image_points: list[list[float]],
    ) -> dict:
        image_size = (int(frame.shape[1]), int(frame.shape[0]))
        camera_matrix = self._initial_camera_matrix(image_size)
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            self.world_points.astype(np.float32),
            np.asarray(image_points, dtype=np.float32),
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            raise ValueError("cv2.solvePnP failed for clicked pitch points")
        projected, _ = cv2.projectPoints(self.world_points, rvec, tvec, camera_matrix, dist_coeffs)
        errors = np.linalg.norm(projected.reshape(-1, 2) - np.asarray(image_points, dtype=np.float32), axis=1)
        rms = float(np.sqrt(np.mean(errors**2)))
        return {
            "profile_name": profile_name,
            "ground": ground_name,
            "camera_id": str(camera_id),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "rms_error_px": rms,
            "image_size": list(image_size),
            "image_points": [[float(x), float(y)] for x, y in image_points],
            "world_points": self.world_points.astype(float).tolist(),
            "rvec": rvec.reshape(-1).astype(float).tolist(),
            "tvec": tvec.reshape(-1).astype(float).tolist(),
            "camera_matrix": camera_matrix.astype(float).tolist(),
            "dist_coeffs": dist_coeffs.reshape(-1).astype(float).tolist(),
        }

    def _require_profile(self) -> dict:
        if self.profile is None:
            raise ValueError("No calibration profile loaded")
        return self.profile

    def _initial_camera_matrix(self, image_size: tuple[int, int]) -> np.ndarray:
        width, height = image_size
        focal = float(max(width, height))
        return np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    def _safe_name(self, value: str) -> str:
        return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip()) or "default"

    def _quality_label(self, rms_error_px: float) -> str:
        if rms_error_px < 3.0:
            return "GOOD"
        if rms_error_px < 6.0:
            return "ACCEPTABLE"
        return "POOR"
