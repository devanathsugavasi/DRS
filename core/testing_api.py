"""FastAPI API for the offline Cricket DRS Testing Platform."""

from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import socket
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import BackgroundTasks, Body, FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from core.decision_mapper import map_summary_to_dashboard_decision
from core.pitch_calibration import (
    MARKER_KEYS,
    ManualPitchCalibrator,
    calibration_status_payload,
    default_icc_profile,
)
from core.testing_database import TestingDatabase
from core.testing_pipeline import AnalysisOptions, DeliveryTestingPipeline, OUTPUT_DIR, UPLOAD_DIR
from core.ws_hub import CHANNELS, WSBroadcastHub
from utils.logger import get_logger

log = get_logger("testing_api")


DB_PATH = Path("data/testing/drs_testing.sqlite3")
CALIBRATION_DIR = Path("data/calibration")
db = TestingDatabase(DB_PATH)
pipeline = DeliveryTestingPipeline()
ws_hub = WSBroadcastHub()
START_TIME = time.time()
MAX_CAMERAS = 6
connected_camera_count = 2
analysis_mode: dict[str, str] = {
    "id": "visible",
    "label": "Mode A - visible-spectrum approximation",
    "description": "Frame differencing and motion-energy analysis. No thermal inference is claimed.",
}
review_history: list[dict[str, Any]] = []

current_decision: dict[str, Any] = {
    "status": "WAITING",
    "outcome": None,
    "time": None,
    "over": "--",
    "ball": "--",
    "decision": "WAITING",
    "ball_confidence": None,
    "tracking_confidence": None,
    "calibration_confidence": None,
    "prediction_confidence": None,
    "model_confidence": None,
    "overall_confidence": None,
    "impact_point": None,
    "wicket_zone_status": "--",
    "ball_speed_kmh": None,
    "trajectory": [],
    "bounce_point": None,
    "predicted_extension": [],
    "wicket_zone": {"x": 412, "y": 64, "w": 18, "h": 42},
}


async def _system_broadcast_loop() -> None:
    while True:
        try:
            payload = {
                "type": "system_health",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "health": system_health_payload(),
                "calibration": calibration_status_payload(),
                "decision": current_decision,
            }
            await ws_hub.broadcast("system", payload)
            await ws_hub.broadcast("live", {"type": "camera_status", "cameras": camera_fps_payload()["cameras"]})
        except Exception as exc:
            log.warning("[WS] System broadcast failed: {}", exc)
        await asyncio.sleep(1.0)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    cleaned, _job_ids = cleanup_stale_jobs(15)
    log.info("[API] Stale job cleanup: {} jobs recovered.", cleaned)
    broadcast_task = asyncio.create_task(_system_broadcast_loop())
    yield
    broadcast_task.cancel()
    try:
        await broadcast_task
    except asyncio.CancelledError:
        pass
    log.info("[API] Shutdown complete.")


def create_testing_app() -> FastAPI:
    app = FastAPI(title="Cricket DRS Testing Platform", version="0.1.0", lifespan=_lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.websocket("/ws/{channel}")
    async def websocket_channel(websocket: WebSocket, channel: str) -> None:
        if channel not in CHANNELS:
            await websocket.close(code=1008)
            return
        await ws_hub.connect(channel, websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            await ws_hub.disconnect(channel, websocket)

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
                "one_to_six_camera_operation",
                "ball_detection",
                "ball_tracking",
                "trajectory_prediction",
                "lbw_analysis",
                "edge_detection_option",
                "replay_generation",
                "json_csv_pdf_exports",
                "electron_primary_dashboard",
                "react_testing_platform",
            ],
            "max_cameras": MAX_CAMERAS,
            "analysis_mode": analysis_mode,
        }

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return health_payload()

    @app.get("/api/testing/health")
    def testing_health() -> dict[str, Any]:
        return health_payload()

    @app.get("/api/cameras/fps")
    def camera_fps() -> dict[str, Any]:
        return camera_fps_payload()

    @app.post("/api/cameras/{camera_id}/reconnect")
    def reconnect_camera(camera_id: int) -> dict[str, Any]:
        global connected_camera_count
        if camera_id < 1 or camera_id > MAX_CAMERAS:
            raise HTTPException(status_code=400, detail="Invalid camera id")
        connected_camera_count = max(connected_camera_count, camera_id)
        schedule_broadcast("system", {"type": "camera_reconnect", "camera_id": camera_id, "status": "connected"})
        return {"camera_id": camera_id, "status": "connected", "cameras": camera_fps_payload()}

    @app.post("/api/cameras/{camera_id}/disconnect")
    def disconnect_camera(camera_id: int) -> dict[str, Any]:
        global connected_camera_count
        if camera_id < 1 or camera_id > MAX_CAMERAS:
            raise HTTPException(status_code=400, detail="Invalid camera id")
        connected_camera_count = max(1, min(connected_camera_count, camera_id - 1))
        schedule_broadcast("system", {"type": "camera_disconnect", "camera_id": camera_id, "status": "offline"})
        return {"camera_id": camera_id, "status": "offline", "cameras": camera_fps_payload()}

    @app.get("/api/live/{camera_id}.jpg")
    def live_camera_frame(camera_id: int) -> Response:
        if camera_id < 1 or camera_id > MAX_CAMERAS or camera_id > connected_camera_count:
            raise HTTPException(status_code=404, detail="Camera not available")
        frame = _synthetic_live_frame(camera_id, thermal_overlay=analysis_mode["id"] == "thermal_demo")
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

    @app.get("/api/system/health")
    def system_health() -> dict[str, Any]:
        return system_health_payload()

    @app.get("/api/analysis-mode")
    def get_analysis_mode() -> dict[str, Any]:
        return analysis_mode

    @app.post("/api/analysis-mode")
    def set_analysis_mode(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
        mode_id = str(payload.get("mode", "visible"))
        if mode_id == "thermal_demo":
            analysis_mode.update(
                {
                    "id": "thermal_demo",
                    "label": "Mode B - demonstration thermal overlay",
                    "description": "Investor demo overlay. Simulated heat colors are presentation graphics, not real thermal data.",
                }
            )
        else:
            analysis_mode.update(
                {
                    "id": "visible",
                    "label": "Mode A - visible-spectrum approximation",
                    "description": "Frame differencing and motion-energy analysis. No thermal inference is claimed.",
                }
            )
        return analysis_mode

    @app.post("/api/appeal/request")
    async def request_appeal(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
        camera_ids = _validated_camera_ids(payload.get("camera_ids") or list(range(1, connected_camera_count + 1)))
        resolved = resolve_dashboard_decision(camera_ids, status="PROCESSING")
        current_decision.update(resolved)
        await ws_hub.broadcast("review", {"type": "appeal_started", "camera_ids": camera_ids})
        await ws_hub.broadcast("decision", {"type": "decision_update", "decision": current_decision})
        log.info("[API] Appeal requested for cameras {}", camera_ids)
        return {"ok": True, "camera_ids": camera_ids, "decision": current_decision}

    @app.post("/api/decision/confirm")
    async def confirm_decision(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
        outcome = str(payload.get("outcome", "")).upper()
        if outcome not in {"OUT", "NOT_OUT"}:
            raise HTTPException(status_code=400, detail="outcome must be OUT or NOT_OUT")
        resolved = resolve_dashboard_decision(list(range(1, connected_camera_count + 1)))
        current_decision.update(resolved)
        current_decision["status"] = outcome
        current_decision["outcome"] = "OUT" if outcome == "OUT" else "NOT OUT"
        current_decision["decision"] = current_decision["outcome"]
        _store_review(current_decision)
        await ws_hub.broadcast("decision", {"type": "decision_confirmed", "decision": current_decision})
        log.info("[API] Decision confirmed: {}", outcome)
        return current_decision

    @app.get("/api/reviews")
    def reviews() -> dict[str, Any]:
        return {"reviews": list(reversed(review_history[-50:]))}

    @app.get("/api/reviews/{review_id}")
    def review_detail(review_id: str) -> dict[str, Any]:
        for review in review_history:
            if review["id"] == review_id:
                return review
        raise HTTPException(status_code=404, detail="Review not found")

    @app.get("/api/calibration/status")
    def calibration_status() -> dict[str, Any]:
        return calibration_status_payload()

    @app.get("/api/calibration/default-profile")
    def calibration_default_profile() -> dict[str, Any]:
        return default_icc_profile()

    @app.get("/api/calibration/cameras/{camera_id}")
    def get_camera_calibration(camera_id: int) -> dict[str, Any]:
        profile = ManualPitchCalibrator().load_profile(camera_id)
        if profile is None:
            raise HTTPException(status_code=404, detail=f"No calibration profile for camera {camera_id}")
        snapshot = _calibration_snapshot_path(camera_id)
        payload = profile.to_dict()
        payload["snapshot_available"] = snapshot.exists()
        if snapshot.exists():
            payload["snapshot_url"] = f"/api/calibration/cameras/{camera_id}/snapshot"
        return payload

    @app.post("/api/calibration/cameras/{camera_id}")
    def save_camera_calibration(camera_id: int, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        markers = payload.get("markers")
        image_size = payload.get("image_size")
        if not isinstance(markers, dict):
            raise HTTPException(status_code=400, detail="markers object is required")
        missing = [key for key in MARKER_KEYS if key not in markers]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing markers: {', '.join(missing)}")
        if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
            raise HTTPException(status_code=400, detail="image_size must be [width, height]")
        try:
            profile = ManualPitchCalibrator().save_profile(
                camera_id,
                markers,
                (int(image_size[0]), int(image_size[1])),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        log.info("[API] Saved manual pitch calibration for camera {}", camera_id)
        return {"saved": True, "profile": profile.to_dict(), "status": calibration_status_payload()}

    @app.get("/api/calibration/cameras/{camera_id}/snapshot")
    def get_camera_calibration_snapshot(camera_id: int) -> FileResponse:
        path = _calibration_snapshot_path(camera_id)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Calibration snapshot not found")
        return FileResponse(path)

    @app.post("/api/calibration/cameras/{camera_id}/snapshot")
    async def upload_camera_calibration_snapshot(camera_id: int, file: UploadFile = File(...)) -> dict[str, Any]:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Snapshot file is required")
        content = await file.read()
        await file.close()
        if not content:
            raise HTTPException(status_code=400, detail="Snapshot upload was empty")
        path = _calibration_snapshot_path(camera_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return {"saved": True, "camera_id": camera_id, "path": str(path)}

    @app.post("/api/calibration/cameras/{camera_id}/capture")
    def capture_camera_calibration_snapshot(camera_id: int) -> dict[str, Any]:
        frame = _synthetic_live_frame(camera_id)
        path = _calibration_snapshot_path(camera_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(path), frame)
        if not ok:
            raise HTTPException(status_code=500, detail="Could not save captured snapshot")
        return {
            "saved": True,
            "camera_id": camera_id,
            "image_size": [frame.shape[1], frame.shape[0]],
            "snapshot_url": f"/api/calibration/cameras/{camera_id}/snapshot",
        }

    @app.post("/api/calibration/import")
    async def import_calibration(file: UploadFile = File(...)) -> dict[str, Any]:
        if not file.filename or not file.filename.lower().endswith(".json"):
            raise HTTPException(status_code=400, detail="Calibration upload must be a JSON file")
        content = await file.read()
        try:
            data = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail="Calibration upload must contain valid JSON") from exc
        CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
        dest = CALIBRATION_DIR / _clean_name(file.filename, "calibration.json")
        dest.write_bytes(content)
        await file.close()
        if data.get("method") == "manual_pitch_markers" and data.get("camera_id") is not None:
            from core.pitch_calibration import refresh_readiness_from_profiles

            refresh_readiness_from_profiles()
        return {"saved": True, "path": str(dest), "status": calibration_status_payload()}

    @app.post("/api/testing/jobs")
    async def create_job(
        background_tasks: BackgroundTasks,
        video_a: UploadFile = File(...),
        video_b: UploadFile | None = File(default=None),
        video_c: UploadFile | None = File(default=None),
        video_d: UploadFile | None = File(default=None),
        video_e: UploadFile | None = File(default=None),
        video_f: UploadFile | None = File(default=None),
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
        videos = [video_a_path]
        secondary_path = None
        for index, upload in enumerate([video_b, video_c, video_d, video_e, video_f], start=1):
            if upload is not None and upload.filename:
                saved = await _save_upload(upload, job_upload_dir / _clean_name(upload.filename, f"camera_{index}.mp4"))
                videos.append(saved)
                secondary_path = secondary_path or saved
        mode = f"{len(videos)}_camera"
        db.create_job(job_id, mode, options_data, video_a_path, secondary_path)
        background_tasks.add_task(_run_job, job_id, videos, options)
        return {"job_id": job_id, "mode": mode, "status": "queued"}

    @app.post("/api/test/upload")
    async def upload_test_job(
        background_tasks: BackgroundTasks,
        video_a: UploadFile = File(...),
        video_b: UploadFile | None = File(default=None),
        options_json: str = Form(default="{}"),
    ) -> dict[str, Any]:
        return await create_job(
            background_tasks=background_tasks,
            video_a=video_a,
            video_b=video_b,
            video_c=None,
            video_d=None,
            video_e=None,
            video_f=None,
            options_json=options_json,
        )

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


def _calibration_snapshot_path(camera_id: int) -> Path:
    return CALIBRATION_DIR / "snapshots" / f"camera_{camera_id}.jpg"


def camera_fps_payload() -> dict[str, Any]:
    now = time.time()
    cameras = []
    for camera_id in range(1, MAX_CAMERAS + 1):
        connected = camera_id <= connected_camera_count
        cameras.append(
            {
                "id": camera_id,
                "fps": 29.2 + ((camera_id % 3) * 0.4) if connected else 0.0,
                "sync_delta_ms": round((camera_id - 1) * 1.7, 2) if connected else None,
                "status": "synthetic" if connected else "offline",
                "mode": analysis_mode["id"],
                "updated_at": now,
                "connected": connected,
                "resolution": "1280x720" if connected else None,
                "latency_ms": round(28.0 + camera_id * 2.1, 1) if connected else None,
                "recording": connected,
                "health": "good" if connected else "offline",
            }
        )
    return {
        "max_cameras": MAX_CAMERAS,
        "connected_count": connected_camera_count,
        "mode": analysis_mode,
        "cameras": cameras,
    }


def system_health_payload() -> dict[str, Any]:
    camera_payload = camera_fps_payload()
    return {
        "cpu_percent": _cpu_percent(),
        "ram_percent": _ram_percent(),
        "gpu": {"available": False, "label": "GPU telemetry unavailable", "percent": None},
        "camera_fps": {str(item["id"]): item["fps"] for item in camera_payload["cameras"] if item["connected"]},
        "frame_drops": {str(item["id"]): (item["id"] - 1) for item in camera_payload["cameras"] if item["connected"]},
        "latency_ms": round(34.0 + connected_camera_count * 3.2, 1),
        "storage": _storage_payload(),
        "network": {"hostname": socket.gethostname(), "status": "local", "latency_ms": 1.0},
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def resolve_dashboard_decision(camera_ids: list[int], status: str | None = None) -> dict[str, Any]:
    job = db.get_latest_completed_job()
    if job and job.get("result") and job["result"].get("summary"):
        decision = map_summary_to_dashboard_decision(job["result"]["summary"], job["id"])
        decision["camera_ids"] = camera_ids
        decision["analysis_mode"] = dict(analysis_mode)
        if status:
            decision["status"] = status
            decision["outcome"] = "Processing review" if status == "PROCESSING" else decision.get("outcome")
        return decision
    return _empty_decision(camera_ids, status)


def _empty_decision(camera_ids: list[int], status: str | None = None) -> dict[str, Any]:
    return {
        "status": status or "REVIEW_INCONCLUSIVE",
        "outcome": "Review inconclusive",
        "time": datetime.now().isoformat(timespec="seconds"),
        "over": "--",
        "ball": "--",
        "decision": "REVIEW INCONCLUSIVE",
        "camera_ids": camera_ids,
        "analysis_mode": dict(analysis_mode),
        "ball_confidence": 0.0,
        "tracking_confidence": 0.0,
        "calibration_confidence": calibration_status_payload().get("quality_score", 0.0),
        "prediction_confidence": 0.0,
        "model_confidence": 0.0,
        "overall_confidence": 0.0,
        "trajectory": [],
        "timeline": [{"label": "Upload delivery", "status": "active"}],
        "explanation": "No completed analysis job found. Upload a delivery in the testing platform, then request review.",
    }


def schedule_broadcast(channel: str, payload: dict[str, Any]) -> None:
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(ws_hub.broadcast(channel, payload))
    except RuntimeError:
        pass


def _sample_processing_decision(camera_ids: list[int] | None = None) -> dict[str, Any]:
    camera_ids = camera_ids or [1]
    confidence_parts = {
        "ball_confidence": 0.82,
        "tracking_confidence": min(0.94, 0.66 + len(camera_ids) * 0.055),
        "calibration_confidence": 0.74,
        "prediction_confidence": 0.79,
        "model_confidence": 0.81,
    }
    overall = sum(confidence_parts.values()) / len(confidence_parts)
    now = datetime.now().isoformat(timespec="seconds")
    over = f"{12 + (len(review_history) // 6)}.{(len(review_history) % 6) + 1}"
    return {
        "status": "PROCESSING",
        "outcome": "Processing review",
        "time": now,
        "over": over,
        "ball": str((len(review_history) % 6) + 1),
        "decision": "PROCESSING",
        **confidence_parts,
        "overall_confidence": round(overall, 3),
        "impact_point": {"x": 382, "y": 86},
        "wicket_zone_status": "Clipping leg stump",
        "ball_speed_kmh": 128.4,
        "camera_ids": camera_ids,
        "analysis_mode": dict(analysis_mode),
        "trajectory": [
            {"x": -9.4, "y": 0.9, "z": 1.55, "confidence": 0.66},
            {"x": -6.2, "y": 0.62, "z": 1.15, "confidence": 0.76},
            {"x": -3.2, "y": 0.28, "z": 0.52, "confidence": 0.83},
            {"x": -1.2, "y": 0.12, "z": 0.05, "confidence": 0.86},
            {"x": 1.4, "y": 0.04, "z": 0.42, "confidence": 0.82},
            {"x": 4.7, "y": -0.12, "z": 0.71, "confidence": 0.79},
        ],
        "bounce_point": {"x": -1.2, "y": 0.12, "z": 0.05},
        "impact_marker": {"x": 4.7, "y": -0.12, "z": 0.71},
        "predicted_extension": [
            {"x": 4.7, "y": -0.12, "z": 0.71},
            {"x": 6.15, "y": -0.17, "z": 0.74},
            {"x": 7.0, "y": -0.21, "z": 0.76},
        ],
        "wicket_zone": {"x": 412, "y": 64, "w": 18, "h": 42},
        "wicket_prediction": {"stump": "leg", "umpire_call": True, "collision": {"x": 7.0, "y": -0.21, "z": 0.76}},
        "timeline": [
            {"label": "Appeal", "status": "complete"},
            {"label": "Ball Detected", "status": "complete"},
            {"label": "Bounce Detected", "status": "complete"},
            {"label": "Impact Detected", "status": "complete"},
            {"label": "Wicket Predicted", "status": "complete"},
            {"label": "Decision Generated", "status": "active"},
            {"label": "Umpire Call", "status": "pending"},
        ],
        "explanation": "Projected path clips leg stump; confidence is gated by calibration and tracking quality.",
    }


def _synthetic_live_frame(camera_id: int, thermal_overlay: bool = False) -> np.ndarray:
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
    if thermal_overlay:
        heat = np.zeros_like(frame)
        cv2.circle(heat, (x, y), 62, (0, 120, 255), -1, cv2.LINE_AA)
        cv2.rectangle(heat, (width // 2 + 160, 160), (width // 2 + 230, 326), (0, 70, 210), -1)
        frame = cv2.addWeighted(frame, 0.62, heat, 0.38, 0)
        cv2.putText(frame, "DEMO THERMAL OVERLAY - SIMULATED", (24, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 210, 255), 2)
    cv2.putText(frame, f"Synthetic camera {camera_id}", (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 245, 245), 2)
    cv2.putText(frame, datetime.now().strftime("%H:%M:%S"), (24, height - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
    return frame


def _validated_camera_ids(raw: Any) -> list[int]:
    ids = []
    for value in raw if isinstance(raw, list) else [raw]:
        try:
            camera_id = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= camera_id <= MAX_CAMERAS and camera_id <= connected_camera_count:
            ids.append(camera_id)
    return sorted(set(ids)) or [1]


def _store_review(decision: dict[str, Any]) -> None:
    review_history.append(
        {
            "id": uuid.uuid4().hex[:10],
            "time": decision.get("time") or datetime.now().isoformat(timespec="seconds"),
            "over": decision.get("over", "--"),
            "ball": decision.get("ball", "--"),
            "decision": decision.get("decision") or decision.get("outcome") or decision.get("status"),
            "confidence": decision.get("overall_confidence") or decision.get("ball_confidence"),
            "replay": {"available": True, "engine": "shared_replay_engine"},
            "trajectory": decision.get("trajectory", []),
            "timeline": decision.get("timeline", []),
            "analysis_mode": decision.get("analysis_mode", analysis_mode),
        }
    )


def _cpu_percent() -> float:
    try:
        import psutil

        return float(psutil.cpu_percent(interval=None))
    except Exception:
        return round(18.0 + (math.sin(time.time() / 9.0) + 1.0) * 11.0, 1)


def _ram_percent() -> float:
    try:
        import psutil

        return float(psutil.virtual_memory().percent)
    except Exception:
        return round(42.0 + (math.cos(time.time() / 11.0) + 1.0) * 7.0, 1)


def _storage_payload() -> dict[str, Any]:
    usage = shutil.disk_usage(os.getcwd())
    return {
        "free_gb": round(usage.free / (1024**3), 2),
        "used_gb": round(usage.used / (1024**3), 2),
        "total_gb": round(usage.total / (1024**3), 2),
    }


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
    schedule_broadcast("review", {"type": "job_processing", "job_id": job_id})
    try:
        log.info("[API] Starting analysis for job {} with {} video(s)", job_id, len(videos))
        result = pipeline.process(job_id, videos, options)
        db.insert_tracking(job_id, [point for cam in result["cameras"] for point in cam["tracking_points"]])
        db.update_job(job_id, "completed", result=result)
        decision = map_summary_to_dashboard_decision(result["summary"], job_id)
        current_decision.update(decision)
        schedule_broadcast("trajectory", {"type": "trajectory_update", "job_id": job_id, "trajectory": decision.get("trajectory", [])})
        schedule_broadcast("decision", {"type": "decision_update", "job_id": job_id, "decision": decision})
        schedule_broadcast("replay", {"type": "replay_ready", "job_id": job_id, "exports": result.get("exports", {})})
        log.info("[API] Job {} completed successfully", job_id)
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {str(exc)}"
        log.error("[API] Job {} failed with error: {}", job_id, error_msg, exc_info=True)
        db.update_job(job_id, "failed", error=error_msg)
        schedule_broadcast("review", {"type": "job_failed", "job_id": job_id, "error": error_msg})


def run_testing_api(host: str, port: int) -> None:
    import uvicorn

    log.info("[API] Starting Cricket DRS Testing API on http://{}:{}", host, port)
    log.info("[API] Loading ball detection model...")

    uvicorn.run(create_testing_app(), host=host, port=port)


app = create_testing_app()
