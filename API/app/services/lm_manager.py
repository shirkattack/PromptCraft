from typing import Any

import dspy

from app.core.config import settings
from app.core.exceptions import ModelNotFoundError, ProviderError
from app.core.logging import get_logger

logger = get_logger("lm_manager")


class LMManager:
    """Manages Language Model initialization and configuration for Ollama provider"""

    SUPPORTED_PROVIDERS = {"ollama": "ollama"}

    @classmethod
    def get_lm(
        cls,
        provider: str,
        model_name: str,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        **kwargs: Any,
    ) -> dspy.LM:
        """
        Initialize and return Ollama language model.

        Args:
            provider: The model provider (must be 'ollama')
            model_name: Name of the Ollama model to use
            api_key: Not used for Ollama (kept for compatibility)
            api_base: Ollama base URL (defaults to settings)
            temperature: Sampling temperature
            max_tokens: Maximum tokens for generation
            **kwargs: Additional Ollama-specific arguments

        Returns:
            Initialized Ollama language model instance

        Raises:
            ProviderError: If provider is not 'ollama'
            ModelNotFoundError: If model is not available
        """
        # Handle both string and enum types
        if hasattr(provider, "value"):
            provider = provider.value.lower()
        else:
            provider = provider.lower()

        if provider != "ollama":
            logger.error(f"Unsupported provider: {provider}. Only Ollama is supported.")
            raise ProviderError(
                f"Unsupported provider: {provider}. This version only supports Ollama.",
                error_code="UNSUPPORTED_PROVIDER",
                details={"provider": provider, "supported": ["ollama"]},
            )

        try:
            # Format model name for DSPy compatibility
            if not model_name.startswith("ollama/"):
                formatted_model = f"ollama/{model_name}"
            else:
                formatted_model = model_name

            logger.info(f"Initializing Ollama model: {model_name}")

            return dspy.LM(
                model=formatted_model,
                api_base=api_base or settings.ollama_base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=settings.ollama_timeout,
                **kwargs,
            )

        except Exception as e:
            logger.error(f"Failed to initialize Ollama model {model_name}: {e}")
            raise ModelNotFoundError(
                f"Failed to initialize Ollama model: {model_name}",
                error_code="MODEL_INITIALIZATION_FAILED",
                details={"model": model_name, "error": str(e)},
            ) from e

    @classmethod
    def test_connection(
        cls,
        provider: str = "ollama",
        model_name: str | None = None,
        api_key: str | None = None,
    ) -> bool:
        """
        Test if connection to Ollama works.

        Args:
            provider: The model provider (must be 'ollama')
            model_name: Name of the Ollama model to test (defaults to settings)
            api_key: Not used for Ollama (kept for compatibility)

        Returns:
            True if connection works, False otherwise
        """
        if not model_name:
            model_name = settings.default_model_name

        try:
            lm = cls.get_lm(provider, model_name, api_key, max_tokens=10)
            # Try a simple completion
            lm("Test", max_tokens=5)
            logger.info(f"Connection test successful for {provider}/{model_name}")
            return True
        except Exception as e:
            logger.error(f"Connection test failed for {provider}/{model_name}: {e}")
            return False

    @classmethod
    async def get_available_models(cls) -> list[str]:
        """
        Get list of available Ollama models.

        Returns:
            List of available model names, falling back to the configured
            default when Ollama cannot be reached.
        """
        try:
            from app.services.ollama_service import ollama_service

            models = await ollama_service.list_models()
            return [model.id for model in models]
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return [settings.default_model_name]
