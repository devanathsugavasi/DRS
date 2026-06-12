"""WebSocket broadcast hub for real-time DRS dashboard updates."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import WebSocket
from utils.logger import get_logger

log = get_logger("ws_hub")

CHANNELS = ("live", "ball", "trajectory", "review", "decision", "replay", "system")


class WSBroadcastHub:
    """Fan-out JSON events to subscribed dashboard clients."""

    def __init__(self) -> None:
        self._clients: dict[str, set[WebSocket]] = {channel: set() for channel in CHANNELS}
        self._lock = asyncio.Lock()

    async def connect(self, channel: str, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            if channel not in self._clients:
                self._clients[channel] = set()
            self._clients[channel].add(websocket)
        log.info("[WS] Client connected to /ws/{}", channel)

    async def disconnect(self, channel: str, websocket: WebSocket) -> None:
        async with self._lock:
            self._clients[channel].discard(websocket)

    async def broadcast(self, channel: str, payload: dict[str, Any]) -> None:
        if channel not in self._clients:
            return
        dead: list[WebSocket] = []
        async with self._lock:
            clients = list(self._clients[channel])
        for websocket in clients:
            try:
                await websocket.send_json(payload)
            except Exception:
                dead.append(websocket)
        for websocket in dead:
            await self.disconnect(channel, websocket)

    async def broadcast_all(self, payload: dict[str, Any], channels: tuple[str, ...] = CHANNELS) -> None:
        for channel in channels:
            await self.broadcast(channel, payload)

    def client_count(self, channel: str | None = None) -> int:
        if channel:
            return len(self._clients.get(channel, set()))
        return sum(len(items) for items in self._clients.values())

    @staticmethod
    def job_channel(job_id: str) -> str:
        return f"job/{job_id}"
