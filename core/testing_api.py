"""FastAPI API for the offline Cricket DRS Testing Platform."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from core.testing_database import TestingDatabase
from core.testing_pipeline import AnalysisOptions, DeliveryTestingPipeline, OUTPUT_DIR, UPLOAD_DIR
from utils.logger import get_logger

log = get_logger("testing_api")


DB_PATH = Path("data/testing/drs_testing.sqlite3")
db = TestingDatabase(DB_PATH)
pipeline = DeliveryTestingPipeline()


def create_testing_app() -> FastAPI:
    app = FastAPI(title="Cricket DRS Testing Platform", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/testing/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "offline": True,
            "database": str(DB_PATH),
            "upload_dir": str(UPLOAD_DIR),
            "output_dir": str(OUTPUT_DIR),
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

    @app.get("/api/testing/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        job = db.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

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

    return app


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
        log.info(f"Starting analysis for job {job_id} with {len(videos)} video(s)")
        result = pipeline.process(job_id, videos, options)
        db.insert_tracking(job_id, [point for cam in result["cameras"] for point in cam["tracking_points"]])
        db.update_job(job_id, "complete", result=result)
        log.info(f"Job {job_id} completed successfully")
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {str(exc)}"
        log.error(f"Job {job_id} failed with error: {error_msg}", exc_info=True)
        db.update_job(job_id, "failed", error=error_msg)


def run_testing_api(host: str, port: int) -> None:
    import uvicorn
    
    log.info(f"Starting Cricket DRS Testing API on http://{host}:{port}")
    log.info("Loading ball detection model...")
    
    uvicorn.run(create_testing_app(), host=host, port=port)
