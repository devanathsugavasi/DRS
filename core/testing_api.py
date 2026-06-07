"""FastAPI API for the offline Cricket DRS Testing Platform."""

from __future__ import annotations

import json
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import BackgroundTasks, Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from core.testing_database import TestingDatabase
from core.testing_pipeline import AnalysisOptions, DeliveryTestingPipeline, OUTPUT_DIR, UPLOAD_DIR
from utils.logger import get_logger

log = get_logger("testing_api")


DB_PATH = Path("data/testing/drs_testing.sqlite3")
CALIBRATION_DIR = Path("data/calibration")
db = TestingDatabase(DB_PATH)
pipeline = DeliveryTestingPipeline()
START_TIME = time.time()

current_decision: dict[str, Any] = {
    "status": "WAITING",
    "outcome": None,
    "ball_confidence": None,
    "impact_point": None,
    "wicket_zone_status": "--",
    "ball_speed_kmh": None,
    "trajectory": [],
    "bounce_point": None,
    "predicted_extension": [],
    "wicket_zone": {"x": 412, "y": 64, "w": 18, "h": 42},
}


def create_testing_app() -> FastAPI:
    app = FastAPI(title="Cricket DRS Testing Platform", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def recover_stale_jobs_on_startup() -> None:
        cleaned, _job_ids = cleanup_stale_jobs(15)
        log.info("[API] Stale job cleanup: {} jobs recovered.", cleaned)

    def health_payload() -> dict[str, Any]:
        detector = pipeline.detector
        return {
            "status": "ok",
            "offline": True,
            "uptime_seconds": int(time.time() - START_TIME),
            "database": str(DB_PATH),
            "upload_dir": str(UPLOAD_DIR),
            "output_dir": str(OUTPUT_DIR),
            "active_model_name": detector.active_model_name,
            "ball_class_ids": sorted(detector.ball_class_ids),
            "model_loaded": detector.model is not None,
            "features": [
                "single_camera_upload",
                "dual_camera_upload",
                "ball_detection",
                "ball_tracking",
                "trajectory_prediction",
                "lbw_analysis",
                "edge_detection_option",
                "replay_generation",
                "json_csv_pdf_exports",
            ],
        }

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return health_payload()

    @app.get("/api/testing/health")
    def testing_health() -> dict[str, Any]:
        return health_payload()

    @app.get("/api/cameras/fps")
    def camera_fps() -> dict[str, Any]:
        now = time.time()
        return {
            "cameras": [
                {
                    "id": 1,
                    "fps": 30.0,
                    "sync_delta_ms": 0.0,
                    "status": "synthetic",
                    "updated_at": now,
                },
                {
                    "id": 2,
                    "fps": 30.0,
                    "sync_delta_ms": 0.0,
                    "status": "synthetic",
                    "updated_at": now,
                },
            ]
        }

    @app.get("/api/live/{camera_id}.jpg")
    def live_camera_frame(camera_id: int) -> Response:
        if camera_id not in {1, 2}:
            raise HTTPException(status_code=404, detail="Camera not available")
        frame = _synthetic_live_frame(camera_id)
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
        if not ok:
            raise HTTPException(status_code=500, detail="Could not encode live frame")
        return Response(
            content=encoded.tobytes(),
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store", "X-Camera-Status": "synthetic"},
        )

    @app.get("/api/decision/current")
    def decision_current() -> dict[str, Any]:
        return current_decision

    @app.post("/api/appeal/request")
    def request_appeal(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
        camera_ids = payload.get("camera_ids", [1, 2])
        current_decision.update(_sample_processing_decision())
        log.info("[API] Appeal requested for cameras {}", camera_ids)
        return {"ok": True, "camera_ids": camera_ids, "decision": current_decision}

    @app.post("/api/decision/confirm")
    def confirm_decision(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
        outcome = str(payload.get("outcome", "")).upper()
        if outcome not in {"OUT", "NOT_OUT"}:
            raise HTTPException(status_code=400, detail="outcome must be OUT or NOT_OUT")
        current_decision.update(_sample_processing_decision())
        current_decision["status"] = outcome
        current_decision["outcome"] = "OUT" if outcome == "OUT" else "NOT OUT"
        log.info("[API] Decision confirmed: {}", outcome)
        return current_decision

    @app.get("/api/calibration/status")
    def calibration_status() -> dict[str, Any]:
        files = sorted(CALIBRATION_DIR.glob("*.json"))
        last_calibrated = None
        if files:
            latest = max(files, key=lambda item: item.stat().st_mtime)
            last_calibrated = datetime.fromtimestamp(latest.stat().st_mtime).isoformat(timespec="seconds")
        return {
            "calibrated": len(files) > 0,
            "camera_count": len(files),
            "last_calibrated": last_calibrated,
            "data_dir": "data/calibration/",
        }

    @app.post("/api/calibration/import")
    async def import_calibration(file: UploadFile = File(...)) -> dict[str, Any]:
        if not file.filename or not file.filename.lower().endswith(".json"):
            raise HTTPException(status_code=400, detail="Calibration upload must be a JSON file")
        content = await file.read()
        try:
            json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail="Calibration upload must contain valid JSON") from exc
        CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
        dest = CALIBRATION_DIR / _clean_name(file.filename, "calibration.json")
        dest.write_bytes(content)
        await file.close()
        return {"saved": True, "path": str(dest), "status": calibration_status()}

    @app.post("/api/testing/jobs")
    async def create_job(
        background_tasks: BackgroundTasks,
        video_a: UploadFile = File(...),
        video_b: UploadFile | None = File(default=None),
        options_json: str = Form(default="{}"),
    ) -> dict[str, Any]:
        job_id = uuid.uuid4().hex[:12]
        try:
            options_data = json.loads(options_json or "{}")
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="options_json must be valid JSON") from exc
        options = AnalysisOptions(**{key: value for key, value in options_data.items() if key in AnalysisOptions.__dataclass_fields__})
        job_upload_dir = UPLOAD_DIR / job_id
        job_upload_dir.mkdir(parents=True, exist_ok=True)
        video_a_path = await _save_upload(video_a, job_upload_dir / _clean_name(video_a.filename, "camera_0.mp4"))
        video_b_path = None
        videos = [video_a_path]
        mode = "single_camera"
        if video_b is not None and video_b.filename:
            video_b_path = await _save_upload(video_b, job_upload_dir / _clean_name(video_b.filename, "camera_1.mp4"))
            videos.append(video_b_path)
            mode = "dual_camera"
        db.create_job(job_id, mode, options_data, video_a_path, video_b_path)
        background_tasks.add_task(_run_job, job_id, videos, options)
        return {"job_id": job_id, "mode": mode, "status": "queued"}

    @app.post("/api/test/upload")
    async def upload_test_job(
        background_tasks: BackgroundTasks,
        video_a: UploadFile = File(...),
        video_b: UploadFile | None = File(default=None),
        options_json: str = Form(default="{}"),
    ) -> dict[str, Any]:
        return await create_job(background_tasks, video_a, video_b, options_json)

    @app.get("/api/testing/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        job = db.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        job["output_dir"] = str(OUTPUT_DIR / job_id)
        return job

    @app.get("/api/test/jobs/{job_id}")
    def get_test_job(job_id: str) -> dict[str, Any]:
        return get_job(job_id)

    @app.get("/api/testing/jobs/{job_id}/exports/{export_name}")
    def download_export(job_id: str, export_name: str) -> FileResponse:
        job = db.get_job(job_id)
        if job is None or not job.get("result"):
            raise HTTPException(status_code=404, detail="Completed job not found")
        exports = job["result"].get("exports", {})
        key_map = {
            "json": "json",
            "csv": "csv",
            "pdf": "pdf",
            "video": "analyzed_video",
            "animation": "animation_video",
        }
        key = key_map.get(export_name)
        if key is None or key not in exports:
            raise HTTPException(status_code=404, detail="Export not available")
        path = Path(exports[key])
        if not path.exists():
            raise HTTPException(status_code=404, detail="Export file missing")
        return FileResponse(path)

    @app.post("/api/testing/jobs/{job_id}/reprocess")
    def reprocess_job(job_id: str, background_tasks: BackgroundTasks, options: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
        job = db.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        videos = [Path(job["video_a_path"])]
        if job.get("video_b_path"):
            videos.append(Path(job["video_b_path"]))
        analysis_options = AnalysisOptions(**{key: value for key, value in options.items() if key in AnalysisOptions.__dataclass_fields__})
        db.update_job(job_id, "queued", error=None)
        background_tasks.add_task(_run_job, job_id, videos, analysis_options)
        return {"job_id": job_id, "status": "queued"}

    @app.post("/api/jobs/cleanup-stale")
    def cleanup_stale_endpoint(older_than_minutes: int = Query(default=15, ge=1)) -> dict[str, Any]:
        cleaned, job_ids = cleanup_stale_jobs(older_than_minutes)
        return {"cleaned": cleaned, "job_ids": job_ids}

    return app


def _sample_processing_decision() -> dict[str, Any]:
    return {
        "status": "PROCESSING",
        "outcome": "Processing review",
        "ball_confidence": 0.82,
        "impact_point": {"x": 382, "y": 86},
        "wicket_zone_status": "Clipping leg stump",
        "ball_speed_kmh": 128.4,
        "trajectory": [
            {"x": 42, "y": 148},
            {"x": 112, "y": 122},
            {"x": 184, "y": 100},
            {"x": 252, "y": 84},
            {"x": 318, "y": 78},
            {"x": 382, "y": 86},
        ],
        "bounce_point": {"x": 252, "y": 84},
        "predicted_extension": [
            {"x": 382, "y": 86},
            {"x": 418, "y": 84},
            {"x": 450, "y": 82},
        ],
        "wicket_zone": {"x": 412, "y": 64, "w": 18, "h": 42},
    }


def _synthetic_live_frame(camera_id: int) -> np.ndarray:
    width, height = 960, 540
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (20, 60, 34) if camera_id == 1 else (45, 48, 58)
    cv2.rectangle(frame, (0, height // 2), (width, height), (35, 92, 48), -1)
    cv2.rectangle(frame, (width // 2 - 70, 80), (width // 2 + 70, height - 40), (178, 162, 114), -1)
    cv2.line(frame, (width // 2 + 140, 80), (width // 2 + 140, height - 40), (235, 235, 220), 3)
    for offset in (-18, 0, 18):
        cv2.line(frame, (width // 2 + 190 + offset, 170), (width // 2 + 190 + offset, 310), (238, 238, 220), 5)
    t = time.time()
    x = int(120 + ((t * 220 + camera_id * 80) % 650))
    y = int(140 + 90 * np.sin(t * 2.4 + camera_id))
    cv2.circle(frame, (x, y), 10, (245, 245, 245), -1, cv2.LINE_AA)
    cv2.putText(frame, f"Synthetic camera {camera_id}", (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 245, 245), 2)
    cv2.putText(frame, datetime.now().strftime("%H:%M:%S"), (24, height - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
    return frame


def cleanup_stale_jobs(older_than_minutes: int = 15) -> tuple[int, list[str]]:
    return db.cleanup_stale_processing_jobs(older_than_minutes)


async def _save_upload(upload: UploadFile, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)
    await upload.close()
    return dest


def _clean_name(filename: str | None, fallback: str) -> str:
    if not filename:
        return fallback
    safe = "".join(char for char in Path(filename).name if char.isalnum() or char in {".", "_", "-"})
    return safe or fallback


def _run_job(job_id: str, videos: list[Path], options: AnalysisOptions) -> None:
    db.update_job(job_id, "processing")
    try:
        log.info("[API] Starting analysis for job {} with {} video(s)", job_id, len(videos))
        result = pipeline.process(job_id, videos, options)
        db.insert_tracking(job_id, [point for cam in result["cameras"] for point in cam["tracking_points"]])
        db.update_job(job_id, "completed", result=result)
        log.info("[API] Job {} completed successfully", job_id)
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {str(exc)}"
        log.error("[API] Job {} failed with error: {}", job_id, error_msg, exc_info=True)
        db.update_job(job_id, "failed", error=error_msg)


def run_testing_api(host: str, port: int) -> None:
    import uvicorn

    log.info("[API] Starting Cricket DRS Testing API on http://{}:{}", host, port)
    log.info("[API] Loading ball detection model...")

    uvicorn.run(create_testing_app(), host=host, port=port)


app = create_testing_app()
