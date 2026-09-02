"""Tests for the optimization session endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

BASE = "/api/v1/sessions"


@pytest.fixture
def session_id(client: TestClient, sample_optimization_session) -> str:
    response = client.post(f"{BASE}/", json=sample_optimization_session)
    assert response.status_code == 200, response.text
    return response.json()["id"]


class TestSessionCrud:
    def test_new_session_starts_running(self, client: TestClient, session_id: str):
        session = client.get(f"{BASE}/{session_id}").json()

        assert session["status"] == "running"
        assert session["optimized_prompt"] is None
        assert session["performance_score"] == 0.0

    def test_missing_session_returns_404(self, client: TestClient):
        assert client.get(f"{BASE}/nope").status_code == 404

    def test_delete(self, client: TestClient, session_id: str):
        assert client.delete(f"{BASE}/{session_id}").status_code == 200
        assert client.get(f"{BASE}/{session_id}").status_code == 404

    def test_limit_is_capped(self, client: TestClient):
        assert client.get(f"{BASE}/?limit=100000").status_code == 422

    def test_optimization_methods_route_is_not_shadowed(self, client: TestClient):
        """`/optimization-methods` must not be matched as a session id."""
        response = client.get(f"{BASE}/optimization-methods")

        assert response.status_code == 200
        assert {m["id"] for m in response.json()["methods"]} == {
            "meta_prompt",
            "dspy",
            "simple",
        }


class TestOptimize:
    def _result(self, **overrides):
        result = {
            "optimized_prompt": "A much better prompt",
            "method": "meta_prompt",
            "improvement_score": 80.0,
            "processing_time": 1.5,
            "metadata": {"method": "meta_prompt"},
            "success": True,
        }
        result.update(overrides)
        return result

    def test_success_marks_session_completed(self, client: TestClient, session_id: str):
        with patch(
            "app.api.v1.endpoints.sessions.optimization_service.optimize_prompt",
            new=AsyncMock(return_value=self._result()),
        ):
            response = client.post(f"{BASE}/{session_id}/optimize")

        assert response.status_code == 200, response.text

        session = client.get(f"{BASE}/{session_id}").json()
        assert session["status"] == "completed"
        assert session["optimized_prompt"] == "A much better prompt"
        assert session["performance_score"] == 80.0

    def test_failed_optimization_is_not_reported_as_success(
        self, client: TestClient, session_id: str
    ):
        """A failed run must surface an error, not a 200 with the original prompt."""
        failure = self._result(
            success=False, error="Ollama unreachable", improvement_score=0.0
        )

        with patch(
            "app.api.v1.endpoints.sessions.optimization_service.optimize_prompt",
            new=AsyncMock(return_value=failure),
        ):
            response = client.post(f"{BASE}/{session_id}/optimize")

        assert response.status_code == 502
        assert "Ollama unreachable" in response.json()["error"]

        session = client.get(f"{BASE}/{session_id}").json()
        assert session["status"] == "failed"
        assert session["optimized_prompt"] is None

    def test_body_options_are_forwarded(self, client: TestClient, session_id: str):
        """Advanced settings travel from the request body to the service."""
        mock = AsyncMock(return_value=self._result(method="dspy"))
        with patch(
            "app.api.v1.endpoints.sessions.optimization_service.optimize_prompt",
            new=mock,
        ):
            response = client.post(
                f"{BASE}/{session_id}/optimize",
                json={
                    "optimization_method": "dspy",
                    "temperature": 0.2,
                    "max_tokens": 512,
                    "output_format": "json",
                    "target_length": "concise",
                    "preserve_wording": True,
                },
            )

        assert response.status_code == 200, response.text
        kwargs = mock.call_args.kwargs
        assert kwargs["optimization_method"] == "dspy"
        assert kwargs["temperature"] == 0.2
        assert kwargs["max_tokens"] == 512
        assert kwargs["output_format"] == "json"
        assert kwargs["target_length"] == "concise"
        assert kwargs["preserve_wording"] is True

    def test_query_method_still_works_and_wins(
        self, client: TestClient, session_id: str
    ):
        mock = AsyncMock(return_value=self._result(method="simple"))
        with patch(
            "app.api.v1.endpoints.sessions.optimization_service.optimize_prompt",
            new=mock,
        ):
            response = client.post(
                f"{BASE}/{session_id}/optimize?optimization_method=simple",
                json={"optimization_method": "dspy"},
            )

        assert response.status_code == 200, response.text
        assert mock.call_args.kwargs["optimization_method"] == "simple"

    def test_out_of_range_temperature_rejected(
        self, client: TestClient, session_id: str
    ):
        response = client.post(f"{BASE}/{session_id}/optimize", json={"temperature": 5})
        assert response.status_code == 422

    def test_unknown_method_rejected(self, client: TestClient, session_id: str):
        response = client.post(
            f"{BASE}/{session_id}/optimize?optimization_method=magic"
        )

        assert response.status_code == 422

    def test_optimize_missing_session_returns_404(self, client: TestClient):
        assert client.post(f"{BASE}/nope/optimize").status_code == 404


class TestPerformanceMetrics:
    def test_empty_database(self, client: TestClient):
        metrics = client.get(f"{BASE}/analytics/performance").json()

        assert metrics["total_optimizations"] == 0
        assert metrics["success_rate"] == 0.0
        assert metrics["average_improvement"] == 0.0

    def test_counts_only_completed_sessions(
        self, client: TestClient, sample_optimization_session
    ):
        first = client.post(f"{BASE}/", json=sample_optimization_session).json()["id"]
        client.post(f"{BASE}/", json=sample_optimization_session)

        client.put(
            f"{BASE}/{first}", json={"status": "completed", "performance_score": 70.0}
        )

        metrics = client.get(f"{BASE}/analytics/performance").json()
        assert metrics["total_optimizations"] == 2
        assert metrics["success_rate"] == 50.0
        assert metrics["average_improvement"] == 70.0
