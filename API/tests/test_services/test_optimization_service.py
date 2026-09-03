"""
Tests for the optimization service.

This module tests the core prompt optimization functionality.
"""

from unittest.mock import patch

import pytest

from app.services.optimization_service import PromptOptimizationService


class TestPromptOptimizationService:
    """Test the PromptOptimizationService class."""

    @pytest.fixture
    def service(self):
        """Create a service instance for testing."""
        return PromptOptimizationService()

    @pytest.mark.asyncio
    async def test_optimize_prompt_meta_method(self, service, mock_successful_llm):
        """Test prompt optimization using meta-prompt method."""
        with patch(
            "app.services.lm_manager.LMManager.get_lm", return_value=mock_successful_llm
        ):
            result = await service.optimize_prompt(
                original_prompt="Hello world",
                provider="ollama",
                model="llama3.2:latest",
                optimization_method="meta_prompt",
            )

        assert result["success"] is True
        assert result["original_prompt"] == "Hello world"
        assert result["method"] == "meta_prompt"
        assert result["provider"] == "ollama"
        assert result["model"] == "llama3.2:latest"
        assert "optimized_prompt" in result
        assert "improvement_score" in result
        assert "processing_time" in result

    @pytest.mark.asyncio
    async def test_optimize_prompt_dspy_method(self, service, mock_successful_llm):
        """Test prompt optimization using DSPy method."""
        with patch(
            "app.services.lm_manager.LMManager.get_lm", return_value=mock_successful_llm
        ):
            result = await service.optimize_prompt(
                original_prompt="Explain AI",
                provider="ollama",
                model="llama3.2:latest",
                optimization_method="dspy",
            )

        assert result["success"] is True
        assert result["method"] == "dspy"
        assert "Task:" in result["optimized_prompt"]

    @pytest.mark.asyncio
    async def test_optimize_prompt_simple_method(self, service, mock_successful_llm):
        """Test prompt optimization using simple method."""
        with patch(
            "app.services.lm_manager.LMManager.get_lm", return_value=mock_successful_llm
        ):
            result = await service.optimize_prompt(
                original_prompt="Write code",
                provider="ollama",
                model="mistral:7b",
                optimization_method="simple",
            )

        assert result["success"] is True
        assert result["method"] == "simple"

    @pytest.mark.asyncio
    async def test_optimize_prompt_failure(self, service, mock_failing_llm):
        """Test prompt optimization failure handling."""
        with patch(
            "app.services.lm_manager.LMManager.get_lm", return_value=mock_failing_llm
        ):
            result = await service.optimize_prompt(
                original_prompt="Test prompt",
                provider="ollama",
                model="llama3.2:latest",
            )

        assert result["success"] is False
        assert result["optimized_prompt"] == "Test prompt"  # Fallback to original
        assert "error" in result
        assert result["improvement_score"] == 0.0

    @pytest.mark.asyncio
    async def test_dspy_method_falls_back_without_a_dspy_lm(
        self, service, mock_successful_llm
    ):
        """The DSPy path reports how it produced the prompt."""
        with patch(
            "app.services.lm_manager.LMManager.get_lm", return_value=mock_successful_llm
        ):
            result = await service.optimize_prompt(
                original_prompt="Explain AI",
                provider="ollama",
                model="llama3.2:latest",
                optimization_method="dspy",
            )

        assert result["success"] is True
        assert result["metadata"]["predictor"] == "template_fallback"
        assert "fallback_reason" in result["metadata"]

    @pytest.mark.asyncio
    async def test_history_is_bounded(self, service, mock_successful_llm):
        """The singleton history must not grow without limit."""
        maxlen = service.optimization_history.maxlen
        assert maxlen is not None

        for _ in range(maxlen + 5):
            service.optimization_history.append({"original_prompt": "x"})

        assert len(service.get_optimization_history()) == maxlen

    def test_unchanged_prompt_scores_zero(self, service):
        """An untouched prompt is not a 50% improvement."""
        assert service._calculate_improvement_score("Same prompt", "Same prompt") == 0.0
        assert (
            service._calculate_improvement_score("Same prompt", "  Same prompt  ")
            == 0.0
        )

    def test_score_breakdown_lists_every_criterion(self, service):
        """The breakdown is what the UI shows, so it must sum to the score."""
        original = "Hello"
        optimized = "## Task\nHello, with an example and a clear format.\n- one\n- two"

        breakdown = service._score_breakdown(original, optimized)
        applied = [item for item in breakdown if item["applied"]]

        assert {item["label"] for item in breakdown} >= {
            "Base",
            "Markdown formatting (## or **)",
        }
        assert sum(
            item["points"] for item in applied
        ) == service._calculate_improvement_score(original, optimized)
        assert all({"label", "points", "applied"} <= item.keys() for item in breakdown)

    def test_score_breakdown_for_unchanged_prompt(self, service):
        assert service._score_breakdown("same", "same") == [
            {"label": "Prompt unchanged", "points": 0, "applied": True}
        ]

    @pytest.mark.asyncio
    async def test_result_metadata_carries_breakdown(
        self, service, mock_successful_llm
    ):
        with patch(
            "app.services.lm_manager.LMManager.get_lm", return_value=mock_successful_llm
        ):
            result = await service.optimize_prompt(
                original_prompt="Write code",
                provider="ollama",
                model="llama3.2:latest",
                optimization_method="simple",
            )

        assert result["success"] is True
        assert isinstance(result["metadata"]["score_breakdown"], list)

    def test_calculate_improvement_score(self, service):
        """Test improvement score calculation."""
        original = "Hello"
        optimized = "Hello, this is a much more detailed and structured prompt with examples and clear instructions."

        score = service._calculate_improvement_score(original, optimized)
        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0
        assert score > 50.0  # Should be higher due to length and structure

    def test_constraints_reach_the_meta_prompt(self, service):
        constraints = service._build_constraints("json", "concise", True)

        assert "JSON" in constraints
        assert "concise" in constraints.lower()
        assert "Preserve the original wording" in constraints

        meta_prompt = service._generate_meta_prompt(
            "Write a story", "creative", constraints
        )
        assert "## Constraints:" in meta_prompt
        assert "JSON" in meta_prompt

    def test_no_constraints_means_no_section(self, service):
        assert service._build_constraints("auto", "auto", False) == ""
        assert "## Constraints:" not in service._generate_meta_prompt("x", "general")

    @pytest.mark.asyncio
    async def test_settings_are_echoed_in_metadata(self, service, mock_successful_llm):
        with patch(
            "app.services.lm_manager.LMManager.get_lm", return_value=mock_successful_llm
        ) as get_lm:
            result = await service.optimize_prompt(
                original_prompt="Write code",
                provider="ollama",
                model="llama3.2:latest",
                optimization_method="simple",
                temperature=0.1,
                max_tokens=256,
            )

        assert get_lm.call_args.kwargs["temperature"] == 0.1
        assert get_lm.call_args.kwargs["max_tokens"] == 256
        assert result["metadata"]["settings"]["temperature"] == 0.1
        assert result["metadata"]["settings"]["max_tokens"] == 256

    def test_generate_meta_prompt(self, service):
        """Test meta-prompt generation."""
        meta_prompt = service._generate_meta_prompt("Write a story", "creative")

        assert "Write a story" in meta_prompt
        assert "creative" in meta_prompt
        assert "Optimization Guidelines" in meta_prompt
        assert "Optimized Prompt:" in meta_prompt

    def test_optimization_history(self, service):
        """Test optimization history tracking."""
        # Initially empty
        assert len(service.get_optimization_history()) == 0

        # Add some mock history
        service.optimization_history.append(
            {
                "original_prompt": "test",
                "optimized_prompt": "improved test",
                "method": "simple",
            }
        )

        history = service.get_optimization_history()
        assert len(history) == 1
        assert history[0]["original_prompt"] == "test"

        # Clear history
        service.clear_history()
        assert len(service.get_optimization_history()) == 0
        assert service.optimized_prompt is None

    def test_user_feedback_becomes_a_constraint(self, service):
        constraints = service._build_constraints(
            "auto", "auto", False, ["Too long", "Lost the deadline"]
        )
        assert "Too long" in constraints and "Lost the deadline" in constraints
        assert "address it" in constraints
        assert service._build_constraints("auto", "auto", False, ["", "  "]) == ""

    @pytest.mark.asyncio
    async def test_user_feedback_is_echoed_in_settings(
        self, service, mock_successful_llm
    ):
        with patch(
            "app.services.lm_manager.LMManager.get_lm", return_value=mock_successful_llm
        ):
            result = await service.optimize_prompt(
                original_prompt="Write code",
                provider="ollama",
                model="llama3.2:latest",
                optimization_method="simple",
                user_feedback=["Too long", " "],
            )
        assert result["metadata"]["settings"]["user_feedback"] == ["Too long"]
