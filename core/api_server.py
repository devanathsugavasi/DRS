"""FastAPI bridge between the Python DRS backend and Electron dashboard."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Optional

import cv2
from fastapi import Body, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from core.camera_manager import CameraManager, ReplayController, VideoFrame
from core.synchronization import SyncVerifier
from utils.logger import get_logger

log = get_logger("api_server")


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

    def start(self) -> None:
        self.camera_manager.start()
        log.info("API backend started with cameras {}", self.camera_ids)

    def stop(self) -> None:
        self.camera_manager.stop()
        log.info("API backend stopped")

    def health(self) -> dict:
        frames = self.camera_manager.latest_frames(write_recording=False)
        sync_report = self.sync_verifier.evaluate(frames)
        return {
            "camera_ids": self.camera_ids,
            "health": self.camera_manager.health(),
            "sync": asdict(sync_report),
            "started_at_ms": self.started_at_ms,
            "timestamp_ms": time.time() * 1000.0,
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
        return [
            {"type": "camera_health", **health},
            {"type": "sync_report", "sync": health.get("sync", {}), "timestamp_ms": health.get("timestamp_ms")},
        ]


def create_app(camera_ids: list[int], record: bool = False) -> FastAPI:
    backend = DRSBackend(camera_ids, record=record)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        backend.start()
        try:
            yield
        finally:
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
                for payload in backend.status_events():
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
    ok, encoded = cv2.imencode(".jpg", item.frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode frame")
    headers = {
        "X-Camera-Id": str(item.camera_id),
        "X-Frame-Id": str(item.frame_id),
        "X-Timestamp-Ms": str(item.timestamp_ms),
        "Cache-Control": "no-store",
    }
    return Response(content=encoded.tobytes(), media_type="image/jpeg", headers=headers)


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


def run_api(camera_ids: list[int], record: bool, host: str, port: int) -> None:
    import uvicorn

    uvicorn.run(create_app(camera_ids, record=record), host=host, port=port, log_level="info")
