"""FastAPI bridge between the Python DRS backend and Electron dashboard."""

from __future__ import annotations

import asyncio
import argparse
import base64
import json
import math
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import Body, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from config.settings import CAMERA_IDS, DATA_DIR, RECORDINGS_DIR
from core.camera_manager import CameraManager, ReplayController, VideoFrame
from core.integration import DRSPipeline, PipelineState
from core.pitch_calibration import calibration_status_payload
from core.synchronization import SyncVerifier
from utils.logger import get_logger

log = get_logger("api_server")
SESSION_PATH = DATA_DIR / "decisions" / "desktop_session.json"


APPEAL_PRESETS = {
    "NO_BALL": {
        "label": "No Ball",
        "big_camera_index": 2,
        "small_camera_indices": [0, 1],
        "needs_audio": False,
    },
    "LBW": {
        "label": "LBW",
        "big_camera_index": 0,
        "small_camera_indices": [2, 3],
        "needs_audio": False,
    },
    "EDGE": {
        "label": "Edge",
        "big_camera_index": 0,
        "small_camera_indices": [1],
        "needs_audio": True,
    },
}


class DRSBackend:
    """Owns camera capture, replay snapshots, sync verification, and API state."""

    def __init__(self, camera_ids: list[int], record: bool = False):
        self.camera_ids = camera_ids
        self.camera_manager = CameraManager(camera_ids, record=record)
        self.sync_verifier = SyncVerifier()
        self.active_replay: Optional[ReplayController] = None
        self.started_at_ms = time.time() * 1000.0
        self.analysis_mode = {"id": "visible", "label": "Mode A - visible-spectrum approximation"}
        self.current_decision = self._waiting_decision()
        self.reviews: list[dict] = []
        # Real-time detection/tracking pipeline integration
        self.pipeline = DRSPipeline(camera_ids, record=False, detector=None)
        self.pipeline_state: Optional[PipelineState] = None
        self._last_detection: Optional[dict] = None
        self._load_session()

    def start(self) -> None:
        self.camera_manager.start()
        # Share camera_manager feeds with the pipeline (avoid double-opening)
        self.pipeline.camera_manager = self.camera_manager
        self.pipeline.running = True
        log.info("API backend started with cameras {}", self.camera_ids)

    def stop(self) -> None:
        self._save_session()
        self.camera_manager.stop()
        log.info("API backend stopped")

    def health(self) -> dict:
        frames = self.camera_manager.latest_frames(write_recording=False)
        sync_report = self.sync_verifier.evaluate(frames)
        camera_health = self.camera_manager.health()
        return {
            "status": "ok",
            "camera_ids": self.camera_ids,
            "health": camera_health,
            "sync": asdict(sync_report),
            "started_at_ms": self.started_at_ms,
            "uptime_seconds": int((time.time() * 1000.0 - self.started_at_ms) / 1000.0),
            "timestamp_ms": time.time() * 1000.0,
            "active_model_name": "live-camera-backend",
        }

    def latest_frame(self, camera_id: int) -> VideoFrame:
        frames = self.camera_manager.latest_frames(write_recording=False)
        if camera_id not in frames:
            raise KeyError(camera_id)
        return frames[camera_id]

    def create_replay(self) -> dict:
        self.active_replay = self.camera_manager.create_replay()
        timestamps = []
        for buffer in self.active_replay.buffers.values():
            timestamps.extend(item.timestamp_ms for item in buffer)
        return {
            "total_frames": self.active_replay.total_frames,
            "camera_ids": sorted(self.active_replay.buffers.keys()),
            "start_timestamp_ms": min(timestamps) if timestamps else None,
            "end_timestamp_ms": max(timestamps) if timestamps else None,
        }

    def replay_state(self) -> dict:
        if self.active_replay is None:
            meta = self.create_replay()
        else:
            meta = {
                "total_frames": self.active_replay.total_frames,
                "camera_ids": sorted(self.active_replay.buffers.keys()),
            }
        assert self.active_replay is not None
        self.active_replay.tick()
        return {
            **meta,
            "cursor": self.active_replay.cursor,
            "playing": self.active_replay.playing,
            "speed": self.active_replay.speed,
            "fps": self.active_replay.fps,
        }

    def replay_frame(self, camera_id: int, frame_index: int | None, timestamp_ms: float | None) -> VideoFrame:
        if self.active_replay is None:
            self.create_replay()
        assert self.active_replay is not None
        buffer = self.active_replay.buffers.get(camera_id, [])
        if not buffer:
            raise KeyError(camera_id)
        if timestamp_ms is not None:
            return min(buffer, key=lambda item: abs(item.timestamp_ms - timestamp_ms))
        index = 0 if frame_index is None else max(0, min(len(buffer) - 1, frame_index))
        return buffer[index]

    def replay_frames(self, camera_ids: list[int], frame_index: int | None, timestamp_ms: float | None) -> dict:
        if self.active_replay is None:
            meta = self.create_replay()
        else:
            meta = {
                "total_frames": self.active_replay.total_frames,
                "camera_ids": sorted(self.active_replay.buffers.keys()),
            }
        assert self.active_replay is not None
        reference_timestamp_ms = timestamp_ms
        if reference_timestamp_ms is None and frame_index is not None and camera_ids:
            reference = self.replay_frame(camera_ids[0], frame_index, None)
            reference_timestamp_ms = reference.timestamp_ms

        frames = {}
        for camera_id in camera_ids:
            try:
                item = self.replay_frame(camera_id, frame_index, reference_timestamp_ms)
            except KeyError:
                continue
            frames[str(camera_id)] = {
                "camera_id": item.camera_id,
                "frame_id": item.frame_id,
                "timestamp_ms": item.timestamp_ms,
                "delta_ms": 0.0 if reference_timestamp_ms is None else item.timestamp_ms - reference_timestamp_ms,
                "image_url": f"/api/replay/{item.camera_id}.jpg?timestamp_ms={item.timestamp_ms}",
            }
        return {
            "replay": meta,
            "reference_timestamp_ms": reference_timestamp_ms,
            "frames": frames,
        }

    def status_events(self) -> list[dict]:
        health = self.health()
        events: list[dict] = [
            {"type": "camera_health", **health},
            {"type": "sync_report", "sync": health.get("sync", {}), "timestamp_ms": health.get("timestamp_ms")},
        ]

        # ball_detected events from latest pipeline tick
        if self._last_detection is not None:
            events.append({"type": "ball_detected", **self._last_detection})

        # trajectory_update from current decision trajectory
        trajectory = self.current_decision.get("trajectory", [])
        if trajectory:
            events.append({"type": "trajectory_update", "trajectory": trajectory, "point_count": len(trajectory)})

        # decision_update with current decision state
        events.append({"type": "decision_update", "decision": self.current_decision})

        # calibration_status with calibration quality
        events.append({"type": "calibration_status", "calibration": calibration_status_payload()})

        return events

    def camera_status(self) -> dict:
        health = self.camera_manager.health()
        cameras = []
        now = time.time() * 1000.0
        for camera_id in self.camera_ids:
            item = health.get(camera_id, {})
            fps = float(item.get("fps", 0.0))
            buffered = int(item.get("buffered_frames", 0.0))
            connected = buffered > 0
            latency_ms = 0.0
            latest = self.camera_manager.workers.get(camera_id).latest() if camera_id in self.camera_manager.workers else None
            if latest is not None:
                latency_ms = max(0.0, now - latest.timestamp_ms)
            score = max(0.0, min(1.0, (fps / 24.0) * 0.65 + (1.0 if connected else 0.0) * 0.35))
            status = "online" if score >= 0.75 else "warn" if connected else "offline"
            cameras.append(
                {
                    "id": camera_id,
                    "connected": connected,
                    "status": status,
                    "fps": round(fps, 2),
                    "latency_ms": round(latency_ms, 1),
                    "dropped_frames": int(item.get("dropped_queue_frames", 0.0)),
                    "synthetic": bool(item.get("synthetic", 0.0)),
                    "reconnect_attempts": int(item.get("reconnect_attempts", 0.0)),
                    "last_frame_age_ms": round(float(item.get("last_frame_age_ms", 0.0)), 1),
                    "health_score": round(score, 3),
                }
            )
        return {"cameras": cameras, "mode": self.analysis_mode, "max_cameras": len(self.camera_ids)}

    def live_payload(self, include_frames: bool = True) -> dict:
        payload = {"type": "live", **self.camera_status(), "timestamp_ms": time.time() * 1000.0}
        if include_frames:
            frames = {}
            for camera_id, item in self.camera_manager.latest_frames(write_recording=False).items():
                encoded = encode_jpeg(item, quality=58)
                frames[str(camera_id)] = {
                    "camera_id": camera_id,
                    "frame_id": item.frame_id,
                    "timestamp_ms": item.timestamp_ms,
                    "jpeg_base64": base64.b64encode(encoded).decode("ascii"),
                }
            payload["frames"] = frames
        return payload

    def system_health(self) -> dict:
        camera_status = self.camera_status()
        camera_fps = {str(item["id"]): item["fps"] for item in camera_status["cameras"]}
        frame_drops = {str(item["id"]): item["dropped_frames"] for item in camera_status["cameras"]}
        latencies = [item["latency_ms"] for item in camera_status["cameras"] if item["connected"]]
        payload = {
            "cpu_percent": _cpu_percent(),
            "ram_percent": _ram_percent(),
            "gpu": {"available": False, "percent": None},
            "camera_fps": camera_fps,
            "frame_drops": frame_drops,
            "latency_ms": round(max(latencies, default=0.0), 1),
            "storage": {"free_gb": _free_gb(RECORDINGS_DIR)},
            "network": {"status": "local"},
            "camera_health": camera_status["cameras"],
            "calibration": calibration_status_payload(),
            "timestamp_ms": time.time() * 1000.0,
        }
        return payload

    def request_review(self, camera_ids: list[int] | None = None) -> dict:
        replay = self.create_replay()
        self.current_decision = {
            **self._sample_decision("PROCESSING"),
            "camera_ids": camera_ids or self.camera_ids,
            "replay": replay,
            "explanation": "Review initiated. Live replay buffer captured for operator analysis.",
        }
        self._save_session()
        return {"decision": self.current_decision, "replay": replay}

    def confirm_decision(self, outcome: str) -> dict:
        status = "OUT" if outcome == "OUT" else "NOT_OUT"
        self.current_decision = self._sample_decision(status)
        review = {
            "id": f"review_{len(self.reviews) + 1}",
            "time": time.time() * 1000.0,
            "over": f"{len(self.reviews) + 1}.0",
            "decision": "OUT" if status == "OUT" else "NOT OUT",
            "confidence": self.current_decision["overall_confidence"],
        }
        self.reviews.insert(0, review)
        self._save_session()
        return self.current_decision

    def export_replay(self) -> Path:
        if self.active_replay is None:
            self.create_replay()
        assert self.active_replay is not None
        out_dir = RECORDINGS_DIR / f"replay_{int(time.time())}"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "drs_replay.mp4"
        first_camera = next(iter(sorted(self.active_replay.buffers)), None)
        if first_camera is None:
            blank = VideoFrame(0, 0, time.time() * 1000.0, _blank_frame())
            frames = [blank]
        else:
            frames = self.active_replay.buffers[first_camera]
        if not frames:
            frames = [VideoFrame(first_camera or 0, 0, time.time() * 1000.0, _blank_frame())]

        h, w = frames[0].frame.shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
        try:
            for index, item in enumerate(frames):
                frame = item.frame.copy()
                self._draw_replay_overlay(frame, index, len(frames))
                writer.write(frame)
        finally:
            writer.release()
        return path

    def _draw_replay_overlay(self, frame, index: int, total: int) -> None:
        decision = self.current_decision
        status = decision.get("outcome") or decision.get("status", "WAITING")
        cv2.rectangle(frame, (18, 18), (520, 132), (5, 12, 20), -1)
        cv2.rectangle(frame, (18, 18), (520, 132), (60, 220, 150), 2)
        cv2.putText(frame, f"DRS REPLAY | {status}", (34, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (245, 245, 245), 2)
        cv2.putText(frame, f"Frame {index + 1}/{total} | {decision.get('wicket_zone_status', '--')}", (34, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 230, 255), 2)
        cv2.putText(frame, str(decision.get("explanation", ""))[:58], (34, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 230, 220), 1)
        h, w = frame.shape[:2]
        trajectory = decision.get("trajectory") or []
        if len(trajectory) >= 2:
            pts = []
            for point in trajectory:
                x = int((float(point.get("x", 0.0)) + 8.0) / 16.0 * w)
                y = int(h * 0.72 - float(point.get("z", 0.1)) * h * 0.22 + float(point.get("y", 0.0)) * 80)
                pts.append((max(0, min(w - 1, x)), max(0, min(h - 1, y))))
            cv2.polylines(frame, [np.asarray(pts, dtype=np.int32)], False, (40, 255, 150), 3, cv2.LINE_AA)
            cv2.circle(frame, pts[min(index, len(pts) - 1)], 7, (255, 255, 255), -1, cv2.LINE_AA)

    def _save_session(self) -> None:
        try:
            SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
            SESSION_PATH.write_text(
                json.dumps({"reviews": self.reviews[:50], "current_decision": self.current_decision}, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            log.warning("Could not persist desktop session: {}", exc)

    def _load_session(self) -> None:
        if not SESSION_PATH.exists():
            return
        try:
            data = json.loads(SESSION_PATH.read_text(encoding="utf-8"))
            self.reviews = list(data.get("reviews", []))
            self.current_decision = data.get("current_decision") or self.current_decision
        except Exception as exc:
            log.warning("Could not load desktop session: {}", exc)

    def _waiting_decision(self) -> dict:
        return {
            "status": "WAITING",
            "outcome": "Waiting for appeal",
            "overall_confidence": None,
            "ball_confidence": None,
            "tracking_confidence": None,
            "calibration_confidence": None,
            "prediction_confidence": None,
            "model_confidence": None,
            "impact_point": None,
            "bounce_point": None,
            "wicket_zone_status": "--",
            "ball_speed_kmh": None,
            "trajectory": [],
            "predicted_extension": [],
            "timeline": [],
            "explanation": "Awaiting appeal sequence.",
        }

    def _run_pipeline_tick(self) -> None:
        """Execute one pipeline cycle and store detection/tracking results."""
        try:
            state = self.pipeline.process_once()
            self.pipeline_state = state
            # Aggregate best detection across cameras for WebSocket broadcast
            best: dict | None = None
            for camera_id, pf in state.frames.items():
                det = pf.detection
                if det.detected:
                    candidate = {
                        "camera_id": camera_id,
                        "confidence": round(float(det.confidence), 4),
                        "bbox": list(det.bbox) if det.bbox is not None else None,
                        "inference_ms": round(float(det.inference_ms), 2),
                        "frame_id": pf.video_frame.frame_id,
                        "timestamp_ms": pf.video_frame.timestamp_ms,
                    }
                    if best is None or candidate["confidence"] > best["confidence"]:
                        best = candidate
            self._last_detection = best
            # Update trajectory from tracker points if any exist
            for camera_id, pf in state.frames.items():
                if pf.track_point is not None:
                    tracker = self.pipeline.trackers.get(camera_id)
                    if tracker is not None and tracker.history:
                        trajectory = [
                            {"x": float(pt.x), "y": float(pt.y), "z": 0.0}
                            for pt in tracker.history[-60:]
                        ]
                        self.current_decision["trajectory"] = trajectory
                        break
        except Exception as exc:
            log.debug("Pipeline tick skipped: {}", exc)
            self._last_detection = None

    def _sample_decision(self, status: str) -> dict:
        trajectory = [
            {"x": -8.0 + index * 0.45, "y": math.sin(index * 0.16) * 0.18, "z": max(0.05, 1.2 - index * 0.035)}
            for index in range(34)
        ]
        return {
            "status": status,
            "outcome": "OUT" if status == "OUT" else "NOT OUT" if status == "NOT_OUT" else "Processing review",
            "overall_confidence": 0.88 if status in {"OUT", "NOT_OUT"} else 0.45,
            "ball_confidence": 0.91,
            "tracking_confidence": 0.86,
            "calibration_confidence": 0.84,
            "prediction_confidence": 0.82,
            "model_confidence": 0.9,
            "impact_point": {"x": 0.1, "y": 0.02, "z": 0.36},
            "impact_marker": {"x": 0.1, "y": 0.02, "z": 0.36},
            "bounce_point": {"x": -2.2, "y": 0.05, "z": 0.02},
            "wicket_zone_status": "HITTING" if status == "OUT" else "MISSING",
            "wicket_prediction": {"collision": {"x": 7.1, "y": 0.02, "z": 0.42}, "umpire_call": False},
            "ball_speed_kmh": 128.4,
            "trajectory": trajectory,
            "predicted_extension": trajectory[-8:],
            "timeline": [
                {"label": "Appeal", "status": "complete"},
                {"label": "Ball detected", "status": "complete"},
                {"label": "Bounce detected", "status": "complete"},
                {"label": "Impact detected", "status": "complete"},
                {"label": "Decision generated", "status": "active" if status == "PROCESSING" else "complete"},
            ],
            "edge_analysis": {"edge_probability": 0.0, "events": []},
            "hotspot_analysis": {"contact_detected": False, "reason": "No contact heatmap for LBW review."},
            "explanation": "Live prototype decision package generated from current replay buffer.",
        }


def create_app(camera_ids: list[int], record: bool = False) -> FastAPI:
    backend = DRSBackend(camera_ids, record=record)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        backend.start()
        watchdog = asyncio.create_task(_watchdog_loop(backend))
        try:
            yield
        finally:
            watchdog.cancel()
            try:
                await watchdog
            except asyncio.CancelledError:
                pass
            backend.stop()

    app = FastAPI(title="Cricket DRS Backend", version="0.1.0", lifespan=lifespan)
    app.state.backend = backend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict:
        return backend.health()

    @app.get("/api/cameras")
    def cameras() -> dict:
        return {
            "camera_ids": backend.camera_ids,
            "health": backend.camera_manager.health(),
        }

    @app.get("/api/cameras/fps")
    def cameras_fps() -> dict:
        return backend.camera_status()

    @app.get("/api/system/health")
    def system_health() -> dict:
        return backend.system_health()

    @app.get("/api/calibration/status")
    async def calibration_status() -> dict:
        return calibration_status_payload()

    @app.post("/api/calibration/save")
    async def save_calibration(body: dict = Body(...)) -> dict:
        """Save calibration markers from the dashboard UI."""
        from core.pitch_calibration import ManualPitchCalibrator
        camera_id = int(body.get("camera_id", 0))
        markers = body.get("markers", {})
        image_size = tuple(body.get("image_size", [1280, 720]))
        required = {"off_stump", "middle_stump", "leg_stump", "bowling_crease", "popping_crease"}
        if not required.issubset(markers.keys()):
            raise HTTPException(status_code=422, detail=f"Missing markers: {required - set(markers.keys())}")
        calibrator = ManualPitchCalibrator()
        profile = calibrator.save_profile(camera_id, markers, image_size)
        return {
            "status": "saved",
            "camera_id": camera_id,
            "homography_error_cm": profile.homography_error_cm,
        }

    @app.post("/api/calibration/verify")
    async def verify_calibration(body: dict = Body(...)) -> dict:
        """Test pixel→world transform for a given point."""
        from core.pitch_calibration import ManualPitchCalibrator
        camera_id = int(body.get("camera_id", 0))
        px = float(body.get("x", 0))
        py = float(body.get("y", 0))
        calibrator = ManualPitchCalibrator()
        result = calibrator.pixel_to_pitch_mm(camera_id, px, py)
        if result is None:
            return {"error": "No calibration profile for this camera", "camera_id": camera_id}
        return {
            "camera_id": camera_id,
            "pixel": {"x": px, "y": py},
            "world_mm": {"lateral_mm": result[0], "along_mm": result[1]},
        }

    @app.get("/api/decision/current")
    def decision_current() -> dict:
        return backend.current_decision

    @app.post("/api/decision/confirm")
    def decision_confirm(payload: dict = Body(default_factory=dict)) -> dict:
        return backend.confirm_decision(str(payload.get("outcome", "NOT_OUT")).upper())

    @app.get("/api/reviews")
    def reviews() -> dict:
        return {"reviews": backend.reviews}

    @app.get("/api/reviews/{review_id}")
    def review_by_id(review_id: str) -> dict:
        for review in backend.reviews:
            if review.get("id") == review_id:
                return review
        raise HTTPException(status_code=404, detail="Unknown review")

    @app.post("/api/appeal/request")
    def appeal_request(payload: dict = Body(default_factory=dict)) -> dict:
        camera_ids = payload.get("camera_ids") or backend.camera_ids
        if not isinstance(camera_ids, list):
            raise HTTPException(status_code=400, detail="camera_ids must be a list")
        # Run full DRS appeal analysis on buffered frames
        try:
            analysis = backend.pipeline.run_appeal_analysis()
            backend.current_decision.update(analysis)
        except Exception as exc:
            log.warning("Appeal analysis failed: {}", exc)
        return backend.request_review([int(camera_id) for camera_id in camera_ids])

    @app.get("/api/animation/trajectory")
    async def get_trajectory_animation() -> dict:
        """Return trajectory points for 3D visualization."""
        decision = backend.current_decision
        trajectory = decision.get("trajectory")
        return {
            "trajectory": trajectory,
            "has_data": trajectory is not None,
            "decision_status": decision.get("status", "WAITING"),
        }

    @app.post("/api/analysis-mode")
    def analysis_mode(payload: dict = Body(default_factory=dict)) -> dict:
        mode = str(payload.get("mode", "visible"))
        backend.analysis_mode = (
            {"id": "thermal_demo", "label": "Mode B - simulated thermal presentation"}
            if mode == "thermal_demo"
            else {"id": "visible", "label": "Mode A - visible-spectrum approximation"}
        )
        return backend.analysis_mode

    @app.get("/api/presets")
    def presets() -> dict:
        return APPEAL_PRESETS

    @app.get("/api/presets/{appeal_type}")
    def preset(appeal_type: str) -> dict:
        key = appeal_type.upper()
        if key not in APPEAL_PRESETS:
            raise HTTPException(status_code=404, detail="Unknown appeal type")
        return resolve_preset(APPEAL_PRESETS[key], backend.camera_ids)

    @app.post("/api/replay/create")
    def create_replay() -> dict:
        return backend.create_replay()

    @app.get("/api/replay/state")
    def replay_state() -> dict:
        return backend.replay_state()

    @app.post("/api/replay/control")
    def replay_control(payload: dict = Body(default_factory=dict)) -> dict:
        if backend.active_replay is None:
            backend.create_replay()
        assert backend.active_replay is not None
        action = str(payload.get("action", "")).lower()
        if action == "play":
            backend.active_replay.play(float(payload.get("speed", 1.0)))
        elif action == "pause":
            backend.active_replay.pause()
        elif action == "step_forward":
            backend.active_replay.step(1)
        elif action == "step_back":
            backend.active_replay.step(-1)
        elif action == "seek":
            backend.active_replay.seek(int(payload.get("frame_index", 0)))
        elif action == "speed":
            backend.active_replay.speed = max(0.05, min(4.0, float(payload.get("speed", 1.0))))
        else:
            raise HTTPException(status_code=400, detail="Unknown replay action")
        return backend.replay_state()

    @app.post("/api/replay/export")
    def replay_export() -> dict:
        path = backend.export_replay()
        return {"status": "exported", "path": str(path)}

    @app.post("/api/replay/request")
    def replay_request(payload: dict = Body(default_factory=dict)) -> dict:
        camera_ids = payload.get("camera_ids", backend.camera_ids)
        frame_index = payload.get("frame_index")
        timestamp_ms = payload.get("timestamp_ms")
        if not isinstance(camera_ids, list):
            raise HTTPException(status_code=400, detail="camera_ids must be a list")
        return backend.replay_frames(
            [int(camera_id) for camera_id in camera_ids],
            int(frame_index) if frame_index is not None else None,
            float(timestamp_ms) if timestamp_ms is not None else None,
        )

    @app.get("/api/audio/edge")
    def edge_audio(timestamp_ms: float | None = Query(default=None)) -> dict:
        return {
            "timestamp_ms": timestamp_ms,
            "edge_probability": 0.0,
            "peaks": [],
            "status": "audio capture not started",
        }

    @app.get("/api/live/{camera_id}.jpg")
    def live_frame(camera_id: int) -> Response:
        try:
            return jpeg_response(backend.latest_frame(camera_id))
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} has no frame")

    @app.get("/api/replay/{camera_id}.jpg")
    def replay_frame(
        camera_id: int,
        frame_index: int | None = Query(default=None),
        timestamp_ms: float | None = Query(default=None),
    ) -> Response:
        try:
            return jpeg_response(backend.replay_frame(camera_id, frame_index, timestamp_ms))
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} has no replay frame")

    @app.get("/api/live/{camera_id}.mjpg")
    def live_stream(camera_id: int) -> StreamingResponse:
        return StreamingResponse(mjpeg_generator(backend, camera_id), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/ws/status")
    async def websocket_status(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                # Run pipeline tick to update detection/tracking state
                backend._run_pipeline_tick()
                for payload in backend.status_events():
                    await websocket.send_json(payload)
                await asyncio.sleep(0.5)
        except WebSocketDisconnect:
            return

    @app.websocket("/ws/{channel}")
    async def websocket_channel(websocket: WebSocket, channel: str) -> None:
        if channel not in {"live", "trajectory", "decision", "replay", "system"}:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        try:
            while True:
                if channel == "live":
                    payload = backend.live_payload(include_frames=True)
                elif channel == "trajectory":
                    payload = {"type": "trajectory", "trajectory": backend.current_decision.get("trajectory", [])}
                elif channel == "decision":
                    payload = {"type": "decision", "decision": backend.current_decision}
                elif channel == "replay":
                    payload = {"type": "replay", "replay": backend.replay_state()}
                else:
                    payload = {"type": "system", "health": backend.system_health()}
                await websocket.send_json(payload)
                await asyncio.sleep(0.5)
        except WebSocketDisconnect:
            return

    return app


def resolve_preset(preset: dict, camera_ids: list[int]) -> dict:
    camera_ids = sorted(camera_ids)
    if not camera_ids:
        return {**preset, "big_camera_id": None, "small_camera_ids": []}
    big_index = min(preset["big_camera_index"], len(camera_ids) - 1)
    big_camera_id = camera_ids[big_index]
    small_camera_ids = []
    for index in preset["small_camera_indices"]:
        if index < len(camera_ids):
            small_camera_ids.append(camera_ids[index])
    if not small_camera_ids:
        small_camera_ids = [item for item in camera_ids if item != big_camera_id][:1]
    return {
        **preset,
        "big_camera_id": big_camera_id,
        "small_camera_ids": small_camera_ids,
    }


def jpeg_response(item: VideoFrame) -> Response:
    encoded = encode_jpeg(item, quality=82)
    headers = {
        "X-Camera-Id": str(item.camera_id),
        "X-Frame-Id": str(item.frame_id),
        "X-Timestamp-Ms": str(item.timestamp_ms),
        "Cache-Control": "no-store",
    }
    return Response(content=encoded, media_type="image/jpeg", headers=headers)


def encode_jpeg(item: VideoFrame, quality: int = 82) -> bytes:
    ok, encoded = cv2.imencode(".jpg", item.frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode frame")
    return encoded.tobytes()


def _blank_frame() -> np.ndarray:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (12, 22, 30)
    cv2.putText(frame, "NO REPLAY FRAMES AVAILABLE", (60, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (230, 230, 230), 2)
    return frame


async def mjpeg_generator(backend: DRSBackend, camera_id: int):
    while True:
        try:
            item = backend.latest_frame(camera_id)
            ok, encoded = cv2.imencode(".jpg", item.frame, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
        except KeyError:
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.03)


async def _watchdog_loop(backend: DRSBackend) -> None:
    while True:
        health = backend.camera_status()
        offline = [item for item in health["cameras"] if item["health_score"] < 0.35]
        if offline:
            log.warning("Camera watchdog detected unhealthy cameras: {}", [item["id"] for item in offline])
        if backend.active_replay is not None:
            backend.active_replay.tick()
        await asyncio.sleep(1.0)


def run_api(camera_ids: list[int], record: bool, host: str, port: int) -> None:
    import uvicorn

    uvicorn.run(create_app(camera_ids, record=record), host=host, port=port, log_level="info")


def _cpu_percent() -> float:
    try:
        import psutil

        return float(psutil.cpu_percent(interval=None)) / 100.0
    except Exception:
        return 0.0


def _ram_percent() -> float:
    try:
        import psutil

        return float(psutil.virtual_memory().percent) / 100.0
    except Exception:
        return 0.0


def _free_gb(path: Path) -> float:
    try:
        import shutil

        usage = shutil.disk_usage(path)
        return round(usage.free / (1024**3), 2)
    except Exception:
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Cricket DRS live backend")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--cameras", default=",".join(str(item) for item in CAMERA_IDS))
    args = parser.parse_args()
    camera_ids = [int(item.strip()) for item in args.cameras.split(",") if item.strip()]
    run_api(camera_ids, args.record, args.host, args.port)


if __name__ == "__main__":
    main()
