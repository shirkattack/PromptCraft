"""
Tests for the Ollama service.

This module tests the Ollama integration functionality.
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.core.exceptions import OllamaConnectionError, ProviderError
from app.services.ollama_service import OllamaService


class TestOllamaService:
    """Test the OllamaService class."""

    @pytest.fixture
    def service(self):
        """Create a service instance for testing."""
        return OllamaService()

    @pytest.mark.asyncio
    async def test_list_models_success(self, service, mock_ollama_response):
        """Test successful model listing."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_ollama_response

        with patch.object(service.client, "get", return_value=mock_response):
            models = await service.list_models()

        # The embedding-only model is filtered out: it cannot run a rewrite.
        assert [m.id for m in models] == ["llama3.2:latest"]
        model = models[0]
        assert model.name == "Llama3.2"
        assert model.is_free is True
        assert model.cost_per_1k_tokens == 0.0
        # Reported by Ollama, not guessed from the name.
        assert model.context_window == 131072
        assert model.parameter_size == "3.2B"
        assert model.quantization == "Q4_K_M"
        assert model.family == "llama"
        assert model.size_bytes == 2019393189
        assert model.capabilities == ["completion", "tools"]
        assert "tool calling" in model.best_use_case

    @pytest.mark.asyncio
    async def test_list_models_connection_error(self, service):
        """Test model listing with connection error."""
        with patch.object(
            service.client, "get", side_effect=httpx.ConnectError("Connection failed")
        ):
            with pytest.raises(OllamaConnectionError) as exc_info:
                await service.list_models()

            assert "Ollama service not available" in str(exc_info.value)
            assert exc_info.value.error_code == "OLLAMA_CONNECTION_FAILED"

    @pytest.mark.asyncio
    async def test_list_models_other_error(self, service):
        """Test model listing with other errors."""
        with patch.object(
            service.client, "get", side_effect=Exception("Unknown error")
        ):
            with pytest.raises(ProviderError) as exc_info:
                await service.list_models()

            assert "Failed to retrieve Ollama models" in str(exc_info.value)
            assert exc_info.value.error_code == "OLLAMA_LIST_MODELS_FAILED"

    @pytest.mark.asyncio
    async def test_generate_completion_success(self, service):
        """Test successful completion generation."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Hello, world!"}

        with patch.object(service.client, "post", return_value=mock_response):
            result = await service.generate_completion("llama3.2", "Say hello")

        assert result == "Hello, world!"

    @pytest.mark.asyncio
    async def test_generate_completion_connection_error(self, service):
        """Test completion generation with connection error."""
        with patch.object(
            service.client, "post", side_effect=httpx.ConnectError("Connection failed")
        ):
            with pytest.raises(OllamaConnectionError) as exc_info:
                await service.generate_completion("llama3.2", "test prompt")

            assert "Ollama service not available for completion" in str(exc_info.value)
            assert exc_info.value.error_code == "OLLAMA_COMPLETION_CONNECTION_FAILED"

    @pytest.mark.asyncio
    async def test_health_check_success(self, service):
        """Test successful health check."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(service.client, "get", return_value=mock_response):
            result = await service.health_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, service):
        """Test failed health check."""
        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch.object(service.client, "get", return_value=mock_response):
            result = await service.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_exception(self, service):
        """Test health check with exception."""
        with patch.object(
            service.client, "get", side_effect=Exception("Connection error")
        ):
            result = await service.health_check()

        assert result is False

    def test_models_without_capabilities_are_kept(self, service):
        """Older Ollama omits `capabilities`; those models must still be listed."""
        assert service._can_complete({"name": "old:latest"}) is True
        assert (
            service._can_complete({"name": "x", "capabilities": ["embedding"]}) is False
        )

    def test_context_window_falls_back_when_unreported(self, service):
        model = service._to_model_response({"name": "mystery:latest", "details": {}})
        assert model.context_window == 4096
        assert model.parameter_size is None

    def test_speed_rating_from_parameter_size(self, service):
        assert service._speed_rating("3.2B") == 5
        assert service._speed_rating("7B") == 4
        assert service._speed_rating("13B") == 3
        assert service._speed_rating("35B-A3B") == 2
        assert service._speed_rating("70B") == 1
        assert service._speed_rating(None) == 3
        assert service._speed_rating("weird") == 3

    def test_best_use_case(self, service):
        assert "code" in service._best_use_case("codellama:7b", "llama", []).lower()
        assert "chat" in service._best_use_case("llama2-chat:7b", "llama", []).lower()
        assert "general" in service._best_use_case("unknown:model", None, []).lower()
        assert "images" in service._best_use_case("llava", "llama", ["vision"])
