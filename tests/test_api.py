import pytest
from httpx import ASGITransport, AsyncClient

from core.testing_api import create_testing_app

SAMPLE_MARKERS = {
    "off_stump": {"x": 100.0, "y": 300.0},
    "middle_stump": {"x": 180.0, "y": 298.0},
    "leg_stump": {"x": 260.0, "y": 300.0},
    "bowling_crease": {"x": 180.0, "y": 360.0},
    "popping_crease": {"x": 180.0, "y": 420.0},
}


@pytest.mark.asyncio
async def test_testing_api_health():
    app = create_testing_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/testing/health")
    assert response.status_code == 200
    assert response.json()["offline"] is True


@pytest.mark.asyncio
async def test_calibration_default_profile_and_save(tmp_path, monkeypatch):
    monkeypatch.setattr("core.testing_api.CALIBRATION_DIR", tmp_path)
    monkeypatch.setattr("core.pitch_calibration.CALIBRATION_DIR", tmp_path)
    monkeypatch.setattr("core.pitch_calibration.READINESS_PATH", tmp_path / "readiness.json")
    app = create_testing_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        default_response = await client.get("/api/calibration/default-profile")
        assert default_response.status_code == 200
        assert "pitch_length_m" in default_response.json()["world_dimensions"]

        save_response = await client.post(
            "/api/calibration/cameras/1",
            json={"markers": SAMPLE_MARKERS, "image_size": [1280, 720]},
        )
        assert save_response.status_code == 200
        payload = save_response.json()
        assert payload["saved"] is True
        assert payload["profile"]["camera_id"] == 1

        get_response = await client.get("/api/calibration/cameras/1")
        assert get_response.status_code == 200
        assert get_response.json()["markers"] == SAMPLE_MARKERS
