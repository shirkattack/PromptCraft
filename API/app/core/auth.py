"""
API Key Authentication Module

Provides optional API key authentication for PromptCraft API.
Can be disabled for local development by setting REQUIRE_API_KEY=false
"""

from fastapi import Header, HTTPException, status
from typing import Optional
from app.core.config import settings


async def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> str:
    """
    Verify API key from request header.

    This dependency can be added to any endpoint that requires authentication.
    If REQUIRE_API_KEY is False, authentication is skipped (useful for local dev).

    Args:
        x_api_key: API key from X-API-Key header

    Returns:
        The validated API key

    Raises:
        HTTPException: If API key is required but missing or invalid
    """
    # Skip auth if not required (local development mode)
    if not settings.require_api_key:
        return "auth_disabled"

    # Check if API key is configured
    if not settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key authentication is enabled but API_KEY is not configured"
        )

    # Validate API key
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Include X-API-Key header in your request.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )

    return x_api_key


# Optional: Dependency for endpoints that should always be public (health checks, etc.)
async def optional_verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> Optional[str]:
    """
    Optional API key verification - doesn't fail if key is missing.
    Useful for endpoints that should be public even when auth is enabled.
    """
    if not settings.require_api_key:
        return None

    if x_api_key and x_api_key == settings.api_key:
        return x_api_key

    return None
