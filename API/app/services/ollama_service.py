import httpx
from typing import List, Dict, Any, Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.core.exceptions import OllamaConnectionError, ProviderError
from app.schemas.optimization import AIModelResponse, AIProviderResponse

class OllamaService:
    """Service for interacting with Ollama local AI models."""
    
    def __init__(self) -> None:
        self.base_url = settings.ollama_base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self.logger = get_logger("ollama_service")
    
    async def list_models(self) -> List[AIModelResponse]:
        """Get list of available Ollama models"""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = []
                for model in data.get("models", []):
                    model_info = AIModelResponse(
                        id=model["name"],
                        name=model["name"].split(":")[0].title(),
                        context_window=self._get_context_window(model["name"]),
                        cost_per_1k_tokens=0.0,  # Local models are free
                        speed_rating=self._estimate_speed_rating(model["name"]),
                        best_use_case=self._get_best_use_case(model["name"]),
                        is_free=True
                    )
                    models.append(model_info)
                return models
            return []
        except httpx.ConnectError as e:
            self.logger.error(f"Cannot connect to Ollama at {self.base_url}: {e}")
            raise OllamaConnectionError(
                f"Ollama service not available at {self.base_url}",
                error_code="OLLAMA_CONNECTION_FAILED",
                details={"base_url": self.base_url, "error": str(e)}
            )
        except Exception as e:
            self.logger.error(f"Error listing Ollama models: {e}")
            raise ProviderError(
                "Failed to retrieve Ollama models",
                error_code="OLLAMA_LIST_MODELS_FAILED",
                details={"error": str(e)}
            )
    
    async def generate_completion(self, model: str, prompt: str, **kwargs) -> Optional[str]:
        """Generate completion using Ollama"""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                **kwargs
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "")
            return None
        except httpx.ConnectError as e:
            self.logger.error(f"Cannot connect to Ollama for completion: {e}")
            raise OllamaConnectionError(
                "Ollama service not available for completion",
                error_code="OLLAMA_COMPLETION_CONNECTION_FAILED",
                details={"model": model, "error": str(e)}
            )
        except Exception as e:
            self.logger.error(f"Error generating completion with model {model}: {e}")
            raise ProviderError(
                f"Failed to generate completion with model {model}",
                error_code="OLLAMA_COMPLETION_FAILED",
                details={"model": model, "error": str(e)}
            )
    
    async def health_check(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            is_healthy = response.status_code == 200
            
            if is_healthy:
                self.logger.debug("Ollama health check passed")
            else:
                self.logger.warning(f"Ollama health check failed with status {response.status_code}")
            
            return is_healthy
        except Exception as e:
            self.logger.warning(f"Ollama health check failed: {e}")
            return False
    
    def _get_context_window(self, model_name: str) -> int:
        """Estimate context window based on model name"""
        if "llama2" in model_name.lower():
            return 4096
        elif "llama" in model_name.lower():
            return 8192
        elif "mistral" in model_name.lower():
            return 8192
        elif "codellama" in model_name.lower():
            return 16384
        else:
            return 4096
    
    def _estimate_speed_rating(self, model_name: str) -> int:
        """Estimate speed rating (1-5) based on model size"""
        if "7b" in model_name.lower():
            return 4  # Fast
        elif "13b" in model_name.lower():
            return 3  # Medium
        elif "70b" in model_name.lower():
            return 2  # Slow
        else:
            return 3  # Default medium
    
    def _get_best_use_case(self, model_name: str) -> str:
        """Determine best use case based on model name"""
        name_lower = model_name.lower()
        if "code" in name_lower:
            return "Code generation and analysis"
        elif "chat" in name_lower or "instruct" in name_lower:
            return "Conversational AI and instruction following"
        elif "llama2" in name_lower:
            return "General purpose text generation"
        elif "mistral" in name_lower:
            return "High-quality text generation"
        else:
            return "General purpose AI tasks"

# Global instance
ollama_service = OllamaService()