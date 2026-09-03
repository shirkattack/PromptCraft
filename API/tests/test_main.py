"""
Tests for the main FastAPI application.

This module tests the core application endpoints and functionality.
"""

from fastapi.testclient import TestClient

from app import __version__


class TestMainEndpoints:
    """Test main application endpoints."""

    def test_root_endpoint(self, client: TestClient):
        """Test the root endpoint returns correct information."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "PromptCraft API"
        assert data["version"] == __version__
        assert "description" in data

    def test_health_check(self, client: TestClient):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["version"] == __version__

    def test_cors_preflight(self, client: TestClient):
        """Preflight is answered by CORSMiddleware, not a catch-all route."""
        response = client.options(
            "/api/v1/sessions/",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert response.status_code == 200
        assert (
            response.headers["access-control-allow-origin"] == "http://localhost:3000"
        )

    def test_cors_headers(self, client: TestClient):
        """Test that CORS headers are properly set."""
        response = client.get("/", headers={"Origin": "http://localhost:3000"})
        assert response.status_code == 200
        assert (
            response.headers["access-control-allow-origin"] == "http://localhost:3000"
        )

    def test_unknown_origin_is_not_allowed(self, client: TestClient):
        """Requests from an unlisted origin get no allow-origin header."""
        response = client.get("/", headers={"Origin": "http://evil.example"})
        assert response.status_code == 200
        assert "access-control-allow-origin" not in response.headers


class TestErrorHandling:
    """Test error handling and exception responses."""

    def test_404_error(self, client: TestClient):
        """Test 404 error for non-existent endpoints."""
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_method_not_allowed(self, client: TestClient):
        """Test 405 error for wrong HTTP methods."""
        response = client.post("/")  # Root only accepts GET
        assert response.status_code == 405
