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


class TestOptimizeAgainstDataset:
    """POST /optimize with a dataset_id measures the prompt instead of guessing."""

    ORIGINAL = "Classify the ticket priority."

    @pytest.fixture
    def dataset_id(self, client: TestClient) -> str:
        response = client.post(
            "/api/v1/training/",
            json={
                "name": "Tickets",
                "task_type": "classification",
                "samples": [
                    {"input_text": "alpha ticket", "expected_output": "high"},
                    {"input_text": "beta ticket", "expected_output": "low"},
                    {"input_text": "gamma ticket", "expected_output": "high"},
                    {"input_text": "delta ticket", "expected_output": "medium"},
                ],
            },
        )
        assert response.status_code == 200, response.text
        return response.json()["id"]

    @pytest.fixture
    def session_id(self, client: TestClient) -> str:
        response = client.post(
            f"{BASE}/",
            json={
                "name": "Measured",
                "original_prompt": self.ORIGINAL,
                "provider": "ollama",
                "model": "llama3.2:latest",
                "task_type": "classification",
            },
        )
        return response.json()["id"]

    def _lm(self):
        from dspy.utils.dummies import DummyLM

        return DummyLM(
            {
                # The meta-prompt contains the original prompt; sample inputs
                # only appear in evaluation calls.
                self.ORIGINAL: {"optimized_prompt": "Classify as high, medium or low."},
                "alpha ticket": {"output": "high"},
                "beta ticket": {"output": "low"},
                "gamma ticket": {"output": "high"},
                "delta ticket": {"output": "nope"},
            }
        )

    def test_measured_scores_are_stored(
        self, client: TestClient, session_id: str, dataset_id: str
    ):
        with patch("app.services.lm_manager.LMManager.get_lm", return_value=self._lm()):
            response = client.post(
                f"{BASE}/{session_id}/optimize",
                json={
                    "dataset_id": dataset_id,
                    "eval_metric": "contains",
                    "max_demos": 2,
                },
            )

        assert response.status_code == 200, response.text
        payload = response.json()
        details = payload["optimization_details"]
        assert details["score_type"] == "measured"
        evaluation = details["metadata"]["eval"]
        assert evaluation["metric"] == "contains"
        assert evaluation["dev_size"] == 1 and evaluation["train_size"] == 3
        assert {c["name"] for c in evaluation["candidates"]} >= {
            "original",
            "rewritten",
        }
        assert details["metadata"]["rewrite"] == "Classify as high, medium or low."

        session = payload["session"]
        assert session["status"] == "completed"
        assert session["dataset_id"] == dataset_id
        assert session["eval_metric"] == "contains"
        assert session["eval_sample_count"] == 1
        assert session["eval_score"] == details["improvement_score"]
        assert session["performance_score"] == session["eval_score"]
        assert session["baseline_score"] is not None
        assert session["optimization_method"] == "meta_prompt"
        assert session["processing_time"] is not None
        assert session["optimized_prompt"].endswith("Input: {input}\nOutput:")

    def test_unknown_dataset_is_404(self, client: TestClient, session_id: str):
        response = client.post(
            f"{BASE}/{session_id}/optimize", json={"dataset_id": "nope"}
        )
        assert response.status_code == 404

    def test_tiny_dataset_is_422(self, client: TestClient, session_id: str):
        created = client.post(
            "/api/v1/training/",
            json={
                "name": "One",
                "task_type": "t",
                "samples": [{"input_text": "a", "expected_output": "b"}],
            },
        ).json()
        response = client.post(
            f"{BASE}/{session_id}/optimize", json={"dataset_id": created["id"]}
        )
        assert response.status_code == 422
        assert client.get(f"{BASE}/{session_id}").json()["status"] == "running"

    def test_runs_without_dataset_store_method_and_duration(
        self, client: TestClient, session_id: str
    ):
        with patch(
            "app.api.v1.endpoints.sessions.optimization_service.optimize_prompt",
            new=AsyncMock(
                return_value={
                    "optimized_prompt": "Better",
                    "method": "simple",
                    "improvement_score": 70.0,
                    "score_type": "heuristic",
                    "processing_time": 2.5,
                    "metadata": {},
                    "success": True,
                }
            ),
        ):
            response = client.post(
                f"{BASE}/{session_id}/optimize", json={"optimization_method": "simple"}
            )

        session = response.json()["session"]
        assert session["optimization_method"] == "simple"
        assert session["processing_time"] == 2.5
        assert session["eval_score"] is None and session["dataset_id"] is None

        metrics = client.get(f"{BASE}/analytics/performance").json()
        assert metrics["total_processing_time"] == 2.5
