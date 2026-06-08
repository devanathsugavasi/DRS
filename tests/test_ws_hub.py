"""WebSocket hub tests."""

from __future__ import annotations

import pytest

from core.ws_hub import WSBroadcastHub


class FakeWebSocket:
    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.accepted = False

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, payload: dict) -> None:
        self.messages.append(payload)


@pytest.mark.asyncio
async def test_ws_hub_broadcast() -> None:
    hub = WSBroadcastHub()
    client = FakeWebSocket()
    await hub.connect("decision", client)
    await hub.broadcast("decision", {"type": "decision_update", "status": "OUT"})
    assert client.accepted is True
    assert client.messages[-1]["status"] == "OUT"
