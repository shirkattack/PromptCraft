"""Tests for the provider catalogue endpoints."""

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.core.exceptions import OllamaConnectionError
from app.schemas.optimization import AIModelResponse

BASE = "/api/v1/providers"

FAKE_MODEL = AIModelResponse(
    id="llama3.2:latest",
    name="Llama3.2",
    context_window=8192,
    cost_per_1k_tokens=0.0,
    speed_rating=4,
    best_use_case="General purpose AI tasks",
    is_free=True,
)


def _by_id(providers):
    return {p["id"]: p for p in providers}


class TestProviderCatalogue:
    def test_ollama_is_available_when_reachable(self, client: TestClient):
        with patch(
            "app.api.v1.endpoints.providers.ollama_service.list_models",
            new=AsyncMock(return_value=[FAKE_MODEL]),
        ):
            response = client.get(f"{BASE}/")

        assert response.status_code == 200, response.text
        ollama = _by_id(response.json())["ollama"]
        assert ollama["available"] is True
        assert ollama["models"][0]["id"] == "llama3.2:latest"

    def test_catalogue_survives_ollama_being_down(self, client: TestClient):
        """A stopped Ollama must not take the whole provider list down."""
        with patch(
            "app.api.v1.endpoints.providers.ollama_service.list_models",
            new=AsyncMock(side_effect=OllamaConnectionError("down")),
        ):
            response = client.get(f"{BASE}/")

        assert response.status_code == 200, response.text
        providers = _by_id(response.json())
        assert providers["ollama"]["available"] is False
        assert providers["ollama"]["unavailable_reason"]

    def test_only_ollama_is_listed(self, client: TestClient):
        """Hosted providers were listed as placeholders; nothing can drive them."""
        with patch(
            "app.api.v1.endpoints.providers.ollama_service.list_models",
            new=AsyncMock(return_value=[FAKE_MODEL]),
        ):
            providers = _by_id(client.get(f"{BASE}/").json())

        assert set(providers) == {"ollama"}
        assert providers["ollama"]["available"] is True

    def test_health_reports_unavailable_without_ollama(self, client: TestClient):
        with patch(
            "app.api.v1.endpoints.providers.ollama_service.health_check",
            new=AsyncMock(return_value=False),
        ):
            response = client.get(f"{BASE}/ollama/health")

        assert response.status_code == 200
        assert response.json() == {"status": "unavailable", "healthy": False}
