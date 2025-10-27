"""
Tests for the optimization service.

This module tests the core prompt optimization functionality.
"""

import pytest
from unittest.mock import patch, MagicMock
from app.services.optimization_service import PromptOptimizationService
from app.core.exceptions import OptimizationError


class TestPromptOptimizationService:
    """Test the PromptOptimizationService class."""
    
    @pytest.fixture
    def service(self):
        """Create a service instance for testing."""
        return PromptOptimizationService()
    
    @pytest.mark.asyncio
    async def test_optimize_prompt_meta_method(self, service, mock_successful_llm):
        """Test prompt optimization using meta-prompt method."""
        with patch('app.services.lm_manager.LMManager.get_lm', return_value=mock_successful_llm):
            result = await service.optimize_prompt(
                original_prompt="Hello world",
                provider="openai",
                model="gpt-3.5-turbo",
                optimization_method="meta_prompt"
            )
        
        assert result["success"] is True
        assert result["original_prompt"] == "Hello world"
        assert result["method"] == "meta_prompt"
        assert result["provider"] == "openai"
        assert result["model"] == "gpt-3.5-turbo"
        assert "optimized_prompt" in result
        assert "improvement_score" in result
        assert "processing_time" in result
    
    @pytest.mark.asyncio
    async def test_optimize_prompt_dspy_method(self, service, mock_successful_llm):
        """Test prompt optimization using DSPy method."""
        with patch('app.services.lm_manager.LMManager.get_lm', return_value=mock_successful_llm):
            result = await service.optimize_prompt(
                original_prompt="Explain AI",
                provider="openai",
                model="gpt-4",
                optimization_method="dspy"
            )
        
        assert result["success"] is True
        assert result["method"] == "dspy"
        assert "Task:" in result["optimized_prompt"]
    
    @pytest.mark.asyncio
    async def test_optimize_prompt_simple_method(self, service, mock_successful_llm):
        """Test prompt optimization using simple method."""
        with patch('app.services.lm_manager.LMManager.get_lm', return_value=mock_successful_llm):
            result = await service.optimize_prompt(
                original_prompt="Write code",
                provider="anthropic",
                model="claude-3",
                optimization_method="simple"
            )
        
        assert result["success"] is True
        assert result["method"] == "simple"
    
    @pytest.mark.asyncio
    async def test_optimize_prompt_failure(self, service, mock_failing_llm):
        """Test prompt optimization failure handling."""
        with patch('app.services.lm_manager.LMManager.get_lm', return_value=mock_failing_llm):
            result = await service.optimize_prompt(
                original_prompt="Test prompt",
                provider="openai",
                model="gpt-3.5-turbo"
            )
        
        assert result["success"] is False
        assert result["optimized_prompt"] == "Test prompt"  # Fallback to original
        assert "error" in result
        assert result["improvement_score"] == 0.0
    
    def test_calculate_improvement_score(self, service):
        """Test improvement score calculation."""
        original = "Hello"
        optimized = "Hello, this is a much more detailed and structured prompt with examples and clear instructions."
        
        score = service._calculate_improvement_score(original, optimized)
        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0
        assert score > 50.0  # Should be higher due to length and structure
    
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
        service.optimization_history.append({
            "original_prompt": "test",
            "optimized_prompt": "improved test",
            "method": "simple"
        })
        
        history = service.get_optimization_history()
        assert len(history) == 1
        assert history[0]["original_prompt"] == "test"
        
        # Clear history
        service.clear_history()
        assert len(service.get_optimization_history()) == 0
        assert service.optimized_prompt is None
