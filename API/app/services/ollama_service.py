import httpx

from app.core.config import settings
from app.core.exceptions import OllamaConnectionError, ProviderError
from app.core.logging import get_logger
from app.schemas.optimization import AIModelResponse


class OllamaService:
    """Service for interacting with Ollama local AI models."""

    def __init__(self) -> None:
        self.base_url = settings.ollama_base_url
        # Local models are slow to load; a 30s ceiling silently killed
        # completions that the configured timeout allows for.
        self.client = httpx.AsyncClient(timeout=settings.ollama_timeout)
        self.logger = get_logger("ollama_service")

    async def aclose(self) -> None:
        """Release the underlying connection pool (called on app shutdown)."""
        await self.client.aclose()

    async def list_models(self) -> list[AIModelResponse]:
        """Get list of available Ollama models"""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                raise ProviderError(
                    "Ollama returned an unexpected response while listing models",
                    error_code="OLLAMA_LIST_MODELS_FAILED",
                    details={"status_code": response.status_code},
                )

            data = response.json()
            return [
                AIModelResponse(
                    id=model["name"],
                    name=model["name"].split(":")[0].title(),
                    context_window=self._get_context_window(model["name"]),
                    cost_per_1k_tokens=0.0,  # Local models are free
                    speed_rating=self._estimate_speed_rating(model["name"]),
                    best_use_case=self._get_best_use_case(model["name"]),
                    is_free=True,
                )
                for model in data.get("models", [])
            ]
        except httpx.ConnectError as e:
            self.logger.error(f"Cannot connect to Ollama at {self.base_url}: {e}")
            raise OllamaConnectionError(
                f"Ollama service not available at {self.base_url}",
                error_code="OLLAMA_CONNECTION_FAILED",
                details={"base_url": self.base_url, "error": str(e)},
            ) from e
        except ProviderError:
            raise
        except Exception as e:
            self.logger.error(f"Error listing Ollama models: {e}")
            raise ProviderError(
                "Failed to retrieve Ollama models",
                error_code="OLLAMA_LIST_MODELS_FAILED",
                details={"error": str(e)},
            ) from e

    async def generate_completion(
        self, model: str, prompt: str, **kwargs
    ) -> str | None:
        """Generate completion using Ollama"""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": settings.ollama_keep_alive,
                **kwargs,
            }

            response = await self.client.post(
                f"{self.base_url}/api/generate", json=payload
            )

            if response.status_code != 200:
                raise ProviderError(
                    f"Ollama returned HTTP {response.status_code} for model {model}",
                    error_code="OLLAMA_COMPLETION_FAILED",
                    details={"model": model, "status_code": response.status_code},
                )

            return response.json().get("response", "")
        except httpx.ConnectError as e:
            self.logger.error(f"Cannot connect to Ollama for completion: {e}")
            raise OllamaConnectionError(
                "Ollama service not available for completion",
                error_code="OLLAMA_COMPLETION_CONNECTION_FAILED",
                details={"model": model, "error": str(e)},
            ) from e
        except ProviderError:
            raise
        except Exception as e:
            self.logger.error(f"Error generating completion with model {model}: {e}")
            raise ProviderError(
                f"Failed to generate completion with model {model}",
                error_code="OLLAMA_COMPLETION_FAILED",
                details={"model": model, "error": str(e)},
            ) from e

    async def health_check(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            is_healthy = response.status_code == 200

            if is_healthy:
                self.logger.debug("Ollama health check passed")
            else:
                self.logger.warning(
                    f"Ollama health check failed with status {response.status_code}"
                )

            return is_healthy
        except Exception as e:
            self.logger.warning(f"Ollama health check failed: {e}")
            return False

    def _get_context_window(self, model_name: str) -> int:
        """Estimate context window based on model name"""
        name_lower = model_name.lower()
        # Order matters: "codellama" and "llama2" also match the generic
        # "llama" test, so the specific names have to be checked first.
        if "codellama" in name_lower:
            return 16384
        if "llama2" in name_lower:
            return 4096
        if "llama" in name_lower:
            return 8192
        if "mistral" in name_lower:
            return 8192
        return 4096

    def _estimate_speed_rating(self, model_name: str) -> int:
        """Estimate speed rating (1-5) based on model size"""
        name_lower = model_name.lower()
        if "70b" in name_lower:
            return 2  # Slow
        if "13b" in name_lower:
            return 3  # Medium
        if "7b" in name_lower or "3b" in name_lower or "1b" in name_lower:
            return 4  # Fast
        return 3  # Default medium

    def _get_best_use_case(self, model_name: str) -> str:
        """Determine best use case based on model name"""
        name_lower = model_name.lower()
        if "code" in name_lower:
            return "Code generation and analysis"
        if "chat" in name_lower or "instruct" in name_lower:
            return "Chat and instruction following"
        if "llama2" in name_lower:
            return "General purpose text generation"
        if "mistral" in name_lower:
            return "High-quality text generation"
        return "General purpose AI tasks"


# Global instance
ollama_service = OllamaService()
