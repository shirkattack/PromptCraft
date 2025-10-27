"""
Custom exception classes for PromptCraft API.

This module defines application-specific exceptions to provide better error handling
and more informative error messages throughout the application.
"""

from typing import Any, Dict, Optional


class PromptCraftException(Exception):
    """Base exception class for all PromptCraft-specific errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class OptimizationError(PromptCraftException):
    """Raised when prompt optimization fails."""
    pass


class ProviderError(PromptCraftException):
    """Raised when there's an issue with AI provider integration."""
    pass


class ModelNotFoundError(ProviderError):
    """Raised when a requested model is not available."""
    pass


class APIKeyError(ProviderError):
    """Raised when API key is missing or invalid."""
    pass


class DatabaseError(PromptCraftException):
    """Raised when database operations fail."""
    pass


class ValidationError(PromptCraftException):
    """Raised when input validation fails."""
    pass


class ConfigurationError(PromptCraftException):
    """Raised when there's a configuration issue."""
    pass


class TrainingDataError(PromptCraftException):
    """Raised when training data operations fail."""
    pass


class SyntheticDataGenerationError(TrainingDataError):
    """Raised when synthetic data generation fails."""
    pass


class OllamaConnectionError(ProviderError):
    """Raised when Ollama service is not accessible."""
    pass


class RateLimitError(ProviderError):
    """Raised when API rate limits are exceeded."""
    pass


class TimeoutError(PromptCraftException):
    """Raised when operations timeout."""
    pass
