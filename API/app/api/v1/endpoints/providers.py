from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import verify_api_key
from app.core.exceptions import ProviderError
from app.core.logging import get_logger
from app.schemas.optimization import AIModelResponse, AIProviderResponse
from app.services.ollama_service import ollama_service

router = APIRouter(dependencies=[Depends(verify_api_key)])
logger = get_logger("providers_endpoints")


@router.get("/", response_model=list[AIProviderResponse])
async def get_providers() -> list[AIProviderResponse]:
    """List the providers this build can drive: Ollama only.

    `available` is False while Ollama is unreachable so clients can explain
    why no model can be selected instead of failing at optimization time.
    """
    providers = []

    try:
        ollama_models = await ollama_service.list_models()
    except ProviderError as e:
        # A local Ollama that is not running should not take the whole
        # provider catalogue down with it.
        logger.warning(f"Ollama unavailable while listing providers: {e.message}")
        ollama_models = []

    providers.append(
        AIProviderResponse(
            id="ollama",
            name="Ollama",
            logo="/logos/ollama.png",
            models=ollama_models,
            available=bool(ollama_models),
            unavailable_reason=None if ollama_models else "Ollama is not reachable.",
        )
    )

    return providers


@router.get("/ollama/health")
async def check_ollama_health() -> dict[str, str | bool]:
    """Check if Ollama is running and accessible"""
    is_healthy = await ollama_service.health_check()
    return {"status": "healthy" if is_healthy else "unavailable", "healthy": is_healthy}


@router.get("/ollama/models", response_model=list[AIModelResponse])
async def get_ollama_models() -> list[AIModelResponse]:
    """Get available Ollama models"""
    models = await ollama_service.list_models()
    if not models:
        raise HTTPException(status_code=503, detail="Ollama service unavailable")
    return models
