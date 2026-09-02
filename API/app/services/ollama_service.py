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
                self._to_model_response(model)
                for model in data.get("models", [])
                if self._can_complete(model)
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

    @staticmethod
    def _can_complete(model: dict) -> bool:
        """Embedding-only models (e.g. nomic-embed-text) cannot run a rewrite.

        Older Ollama versions omit `capabilities`; treat those as completion
        models rather than hiding everything.
        """
        capabilities = model.get("capabilities")
        return not capabilities or "completion" in capabilities

    def _to_model_response(self, model: dict) -> AIModelResponse:
        """Build the API view of a model from what /api/tags reports."""
        name = model["name"]
        details = model.get("details") or {}
        parameter_size = details.get("parameter_size")
        capabilities = list(model.get("capabilities") or [])

        return AIModelResponse(
            id=name,
            name=name.split(":")[0].title(),
            context_window=int(
                details.get("context_length") or self._fallback_context_window(name)
            ),
            cost_per_1k_tokens=0.0,  # Local models are free
            speed_rating=self._speed_rating(parameter_size),
            best_use_case=self._best_use_case(
                name, details.get("family"), capabilities
            ),
            is_free=True,
            parameter_size=parameter_size,
            quantization=details.get("quantization_level"),
            family=details.get("family"),
            size_bytes=model.get("size"),
            capabilities=capabilities,
        )

    @staticmethod
    def _fallback_context_window(model_name: str) -> int:
        """Used only when Ollama does not report context_length."""
        return 4096

    @staticmethod
    def _parse_parameter_count(parameter_size: str | None) -> float | None:
        """'3.2B' -> 3.2, '7B' -> 7.0, '35B-A3B' -> 35.0, '500M' -> 0.5."""
        if not parameter_size:
            return None
        head = parameter_size.upper().split("-")[0].strip()
        try:
            if head.endswith("B"):
                return float(head[:-1])
            if head.endswith("M"):
                return float(head[:-1]) / 1000
        except ValueError:
            return None
        return None

    def _speed_rating(self, parameter_size: str | None) -> int:
        """1-5, from the parameter count: smaller models answer faster."""
        params = self._parse_parameter_count(parameter_size)
        if params is None:
            return 3  # Unknown size: assume middling
        if params <= 4:
            return 5
        if params <= 8:
            return 4
        if params <= 14:
            return 3
        if params <= 35:
            return 2
        return 1

    @staticmethod
    def _best_use_case(
        model_name: str, family: str | None, capabilities: list[str]
    ) -> str:
        """Describe what the model is good for from its name, family and capabilities."""
        name_lower = model_name.lower()
        traits = []
        if "code" in name_lower or "coder" in name_lower:
            traits.append("Code generation and analysis")
        elif "chat" in name_lower or "instruct" in name_lower:
            traits.append("Chat and instruction following")
        elif family:
            traits.append(f"General purpose ({family})")
        else:
            traits.append("General purpose AI tasks")
        if "vision" in capabilities:
            traits.append("understands images")
        if "tools" in capabilities:
            traits.append("supports tool calling")
        if "thinking" in capabilities:
            traits.append("extended reasoning")
        return "; ".join(traits)


# Global instance
ollama_service = OllamaService()
