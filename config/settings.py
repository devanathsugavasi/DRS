"""Global settings for the cricket DRS prototype."""

from pathlib import Path

try:
    from pydantic_settings import BaseSettings
except Exception:  # pragma: no cover - allows old environments to import before deps are installed
    BaseSettings = object  # type: ignore[misc,assignment]


class DRSSettings(BaseSettings):
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    LOG_LEVEL: str = "INFO"
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8765
    TESTING_API_PORT: int = 8766
    CAMERA_SYNC_TOLERANCE_MS: float = 2.0
    REPLAY_BUFFER_SECONDS: float = 30.0
    BALL_CONFIDENCE_THRESHOLD: float = 0.45
    LBW_PITCH_ZONE_MARGIN_PX: int = 10
    STUMP_WIDTH_MM: float = 228.6
    STUMP_HEIGHT_MM: float = 711.2
    FRAME_HISTORY_SIZE: int = 300

    if BaseSettings is not object:
        model_config = {
            "env_file": ".env",
            "env_file_encoding": "utf-8",
            "extra": "ignore",
        }


settings = DRSSettings()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EXPORTS_DIR = DATA_DIR / "exports"
DECISIONS_DIR = DATA_DIR / "decisions"
CALIBRATION_DIR = DATA_DIR / "calibration"
RECORDINGS_DIR = DATA_DIR / "recordings"
DETECTIONS_DIR = DATA_DIR / "detections"
TRACKING_DIR = DATA_DIR / "tracking"
SYNC_DIR = DATA_DIR / "sync"
AUDIO_DIR = DATA_DIR / "audio"
LOGS_DIR = DATA_DIR / "logs"

for directory in (
    CALIBRATION_DIR,
    RECORDINGS_DIR,
    DETECTIONS_DIR,
    TRACKING_DIR,
    SYNC_DIR,
    AUDIO_DIR,
    LOGS_DIR,
    EXPORTS_DIR,
    DECISIONS_DIR,
    BASE_DIR / "models",
):
    directory.mkdir(parents=True, exist_ok=True)

CAMERA_IDS = [0, 1]
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 60
BUFFER_SECONDS = 30
SYNC_TOLERANCE_MS = 8.0
CAPTURE_QUEUE_SIZE = 4

VIDEO_CODEC = "mp4v"
VIDEO_EXT = ".mp4"

YOLO_MODEL_PATH = BASE_DIR / "models" / "cricket_ball_yolov8.pt"
YOLO_CONF_THRESH = 0.35
YOLO_IOU_THRESH = 0.45
YOLO_IMG_SIZE = 640
INFERENCE_DEVICE = "cuda"
USE_TENSORRT = False

KALMAN_PROCESS_NOISE = 1e-2
KALMAN_MEASUREMENT_NOISE = 1e-1
MAX_MISSING_FRAMES = 10
TRAJECTORY_HISTORY = 90

CHECKERBOARD_SIZE = (9, 6)
SQUARE_SIZE_MM = 25.0
CALIBRATION_MIN_IMAGES = 15

PITCH_LENGTH_M = 20.12
PITCH_WIDTH_M = 3.05
CREASE_TO_STUMPS_M = 1.22
STUMP_WIDTH_M = 0.2286
STUMP_HEIGHT_M = 0.711
BALL_RADIUS_M = 0.0363
GRAVITY_MPS2 = 9.81
BOUNCE_RESTITUTION = 0.58
DRAG_COEFFICIENT = 0.47
AIR_DENSITY = 1.225
BALL_MASS_KG = 0.156

AUDIO_SAMPLE_RATE = 44100
AUDIO_CHANNELS = 1
AUDIO_CHUNK_SIZE = 1024
EDGE_FREQ_LOW_HZ = 1500
EDGE_FREQ_HIGH_HZ = 8000
EDGE_SPIKE_THRESHOLD = 3.5

WORKER_THREADS = 4
