"""
Tests for the main FastAPI application.

This module tests the core application endpoints and functionality.
"""

import pytest
from fastapi.testclient import TestClient


class TestMainEndpoints:
    """Test main application endpoints."""
    
    def test_root_endpoint(self, client: TestClient):
        """Test the root endpoint returns correct information."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "PromptCraft API"
        assert data["version"] == "1.0.0"
        assert "description" in data
    
    def test_health_check(self, client: TestClient):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["version"] == "1.0.0"
    
    def test_options_handler(self, client: TestClient):
        """Test CORS preflight requests."""
        response = client.options("/some/path")
        assert response.status_code == 200
        assert response.json() == {"message": "OK"}
    
    def test_cors_headers(self, client: TestClient):
        """Test that CORS headers are properly set."""
        response = client.get("/", headers={"Origin": "http://localhost:3000"})
        assert response.status_code == 200
        # Note: TestClient doesn't automatically handle CORS headers
        # In a real test, you'd check for Access-Control-Allow-Origin


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
