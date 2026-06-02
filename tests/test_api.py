import pytest
from httpx import ASGITransport, AsyncClient

from core.testing_api import create_testing_app


@pytest.mark.asyncio
async def test_testing_api_health():
    app = create_testing_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/testing/health")
    assert response.status_code == 200
    assert response.json()["offline"] is True
