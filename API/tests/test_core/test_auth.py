"""Tests for optional API key authentication."""

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings

PROTECTED = "/api/v1/sessions/"
API_KEY = "test-secret-key"


@pytest.fixture
def auth_required(monkeypatch):
    monkeypatch.setattr(settings, "require_api_key", True)
    monkeypatch.setattr(settings, "api_key", API_KEY)


class TestAuthDisabled:
    def test_requests_pass_without_a_key(self, client: TestClient):
        assert client.get(PROTECTED).status_code == 200


class TestAuthEnabled:
    def test_missing_key_is_rejected(self, client: TestClient, auth_required):
        response = client.get(PROTECTED)

        assert response.status_code == 401
        assert response.headers.get("www-authenticate") == "ApiKey"

    def test_wrong_key_is_rejected(self, client: TestClient, auth_required):
        response = client.get(PROTECTED, headers={"X-API-Key": "nope"})

        assert response.status_code == 403

    def test_correct_key_is_accepted(self, client: TestClient, auth_required):
        response = client.get(PROTECTED, headers={"X-API-Key": API_KEY})

        assert response.status_code == 200

    def test_health_stays_public(self, client: TestClient, auth_required):
        """Monitoring must not need a credential."""
        assert client.get("/health").status_code == 200
        assert client.get("/").status_code == 200

    def test_enabled_without_a_configured_key_fails_loudly(
        self, client: TestClient, monkeypatch
    ):
        monkeypatch.setattr(settings, "require_api_key", True)
        monkeypatch.setattr(settings, "api_key", None)

        response = client.get(PROTECTED, headers={"X-API-Key": "anything"})

        assert response.status_code == 500
