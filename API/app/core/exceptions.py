"""
Custom exception classes for PromptCraft API.

This module defines application-specific exceptions to provide better error handling
and more informative error messages throughout the application.

Each exception carries the HTTP status code it should surface as, so the handler in
``app.main`` can translate an upstream outage into a 503 instead of flattening every
failure into a 400.
"""

from typing import Any


class PromptCraftException(Exception):
    """Base exception class for all PromptCraft-specific errors."""

    status_code: int = 400

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class OptimizationError(PromptCraftException):
    """Raised when prompt optimization fails."""

    status_code = 500


class ProviderError(PromptCraftException):
    """Raised when there's an issue with AI provider integration."""

    status_code = 502


class ModelNotFoundError(ProviderError):
    """Raised when a requested model is not available."""

    status_code = 404


class APIKeyError(ProviderError):
    """Raised when API key is missing or invalid."""

    status_code = 401


class DatabaseError(PromptCraftException):
    """Raised when database operations fail."""

    status_code = 500


class ValidationError(PromptCraftException):
    """Raised when input validation fails."""

    status_code = 422


class ConfigurationError(PromptCraftException):
    """Raised when there's a configuration issue."""

    status_code = 500


class TrainingDataError(PromptCraftException):
    """Raised when training data operations fail."""


class SyntheticDataGenerationError(TrainingDataError):
    """Raised when synthetic data generation fails."""

    status_code = 502


class OllamaConnectionError(ProviderError):
    """Raised when Ollama service is not accessible."""

    status_code = 503


class RateLimitError(ProviderError):
    """Raised when API rate limits are exceeded."""

    status_code = 429


class TimeoutError(PromptCraftException):
    """Raised when operations timeout."""

    status_code = 504
