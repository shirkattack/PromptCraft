"""
Tests for the Ollama service.

This module tests the Ollama integration functionality.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import httpx
from app.services.ollama_service import OllamaService
from app.core.exceptions import OllamaConnectionError, ProviderError


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
        
        with patch.object(service.client, 'get', return_value=mock_response):
            models = await service.list_models()
        
        assert len(models) == 1
        assert models[0].id == "llama3.2:latest"
        assert models[0].name == "Llama3.2"
        assert models[0].is_free is True
        assert models[0].cost_per_1k_tokens == 0.0
    
    @pytest.mark.asyncio
    async def test_list_models_connection_error(self, service):
        """Test model listing with connection error."""
        with patch.object(service.client, 'get', side_effect=httpx.ConnectError("Connection failed")):
            with pytest.raises(OllamaConnectionError) as exc_info:
                await service.list_models()
            
            assert "Ollama service not available" in str(exc_info.value)
            assert exc_info.value.error_code == "OLLAMA_CONNECTION_FAILED"
    
    @pytest.mark.asyncio
    async def test_list_models_other_error(self, service):
        """Test model listing with other errors."""
        with patch.object(service.client, 'get', side_effect=Exception("Unknown error")):
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
        
        with patch.object(service.client, 'post', return_value=mock_response):
            result = await service.generate_completion("llama3.2", "Say hello")
        
        assert result == "Hello, world!"
    
    @pytest.mark.asyncio
    async def test_generate_completion_connection_error(self, service):
        """Test completion generation with connection error."""
        with patch.object(service.client, 'post', side_effect=httpx.ConnectError("Connection failed")):
            with pytest.raises(OllamaConnectionError) as exc_info:
                await service.generate_completion("llama3.2", "test prompt")
            
            assert "Ollama service not available for completion" in str(exc_info.value)
            assert exc_info.value.error_code == "OLLAMA_COMPLETION_CONNECTION_FAILED"
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, service):
        """Test successful health check."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        with patch.object(service.client, 'get', return_value=mock_response):
            result = await service.health_check()
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, service):
        """Test failed health check."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        
        with patch.object(service.client, 'get', return_value=mock_response):
            result = await service.health_check()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_health_check_exception(self, service):
        """Test health check with exception."""
        with patch.object(service.client, 'get', side_effect=Exception("Connection error")):
            result = await service.health_check()
        
        assert result is False
    
    def test_get_context_window(self, service):
        """Test context window estimation."""
        assert service._get_context_window("llama2:7b") == 4096
        assert service._get_context_window("llama3.2:latest") == 8192
        assert service._get_context_window("mistral:7b") == 8192
        assert service._get_context_window("unknown:model") == 4096  # default
    
    def test_estimate_speed_rating(self, service):
        """Test speed rating estimation."""
        rating_7b = service._estimate_speed_rating("model:7b")
        rating_13b = service._estimate_speed_rating("model:13b")
        rating_70b = service._estimate_speed_rating("model:70b")
        
        assert 1 <= rating_7b <= 5
        assert 1 <= rating_13b <= 5
        assert 1 <= rating_70b <= 5
        assert rating_7b >= rating_13b >= rating_70b  # Smaller models should be faster
    
    def test_get_best_use_case(self, service):
        """Test best use case determination."""
        assert "code" in service._get_best_use_case("codellama:7b").lower()
        assert "chat" in service._get_best_use_case("llama2-chat:7b").lower()
        assert "general" in service._get_best_use_case("unknown:model").lower()
