import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from core.api_server import create_app


@pytest.mark.asyncio
async def test_dashboard_backend_routes_without_camera_startup():
    app = create_app([0], record=False)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        health = await client.get("/api/system/health")
        cameras = await client.get("/api/cameras/fps")
        decision = await client.get("/api/decision/current")
        review = await client.post("/api/appeal/request", json={"camera_ids": [0]})
        replay = await client.post("/api/replay/control", json={"action": "pause"})
        export = await client.post("/api/replay/export")
        mode = await client.post("/api/analysis-mode", json={"mode": "thermal_demo"})

    assert health.status_code == 200
    assert "camera_fps" in health.json()
    assert cameras.status_code == 200
    assert cameras.json()["cameras"][0]["id"] == 0
    assert "status" in decision.json()
    assert review.status_code == 200
    assert "decision" in review.json()
    assert replay.status_code == 200
    assert "total_frames" in replay.json()
    assert export.status_code == 200
    assert export.json()["path"].endswith(".mp4")
    assert mode.json()["id"] == "thermal_demo"


@pytest.mark.asyncio
async def test_confirm_decision_records_review_history():
    app = create_app([0], record=False)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        confirm = await client.post("/api/decision/confirm", json={"outcome": "OUT"})
        reviews = await client.get("/api/reviews")

    assert confirm.status_code == 200
    assert confirm.json()["status"] == "OUT"
    assert reviews.json()["reviews"][0]["decision"] == "OUT"


def test_live_websocket_streams_camera_frame_payloads():
    app = create_app([9999], record=False)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/live") as websocket:
            payload = websocket.receive_json()
            for _ in range(8):
                if payload.get("frames", {}).get("9999", {}).get("jpeg_base64"):
                    break
                payload = websocket.receive_json()

    assert payload["type"] == "live"
    assert payload["cameras"][0]["connected"] is True
    assert payload["frames"]["9999"]["jpeg_base64"]
