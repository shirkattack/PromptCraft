from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import verify_api_key
from app.core.exceptions import ProviderError
from app.core.logging import get_logger
from app.schemas.optimization import AIModelResponse, AIProviderResponse
from app.services.ollama_service import ollama_service

router = APIRouter(dependencies=[Depends(verify_api_key)])
logger = get_logger("providers_endpoints")

# Hosted providers are advertised so clients can show the full catalogue, but
# LMManager only implements Ollama today -- selecting one of these for an
# optimization run would fail. They are returned with available=False rather
# than silently offered alongside working models.
_UNAVAILABLE_REASON = "This build only runs local models through Ollama."

OPENAI_MODELS = [
    AIModelResponse(
        id="gpt-4",
        name="GPT-4",
        context_window=8192,
        cost_per_1k_tokens=0.03,
        speed_rating=3,
        best_use_case="Complex reasoning and analysis",
        is_free=False,
    ),
    AIModelResponse(
        id="gpt-3.5-turbo",
        name="GPT-3.5 Turbo",
        context_window=4096,
        cost_per_1k_tokens=0.002,
        speed_rating=5,
        best_use_case="Fast general purpose tasks",
        is_free=False,
    ),
]

# cost_per_1k_tokens is the input rate; output tokens are billed higher.
ANTHROPIC_MODELS = [
    AIModelResponse(
        id="claude-opus-5",
        name="Claude Opus 5",
        context_window=1_000_000,
        cost_per_1k_tokens=0.005,
        speed_rating=3,
        best_use_case="Complex analysis, coding and agentic tasks",
        is_free=False,
    ),
    AIModelResponse(
        id="claude-sonnet-5",
        name="Claude Sonnet 5",
        context_window=1_000_000,
        cost_per_1k_tokens=0.002,
        speed_rating=4,
        best_use_case="Balanced performance and cost",
        is_free=False,
    ),
    AIModelResponse(
        id="claude-haiku-4-5",
        name="Claude Haiku 4.5",
        context_window=200_000,
        cost_per_1k_tokens=0.001,
        speed_rating=5,
        best_use_case="Fast, high-volume tasks",
        is_free=False,
    ),
]


@router.get("/", response_model=list[AIProviderResponse])
async def get_providers():
    """Get all available AI providers and their models"""
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

    providers.append(
        AIProviderResponse(
            id="openai",
            name="OpenAI",
            logo="/logos/openai.png",
            models=OPENAI_MODELS,
            available=False,
            unavailable_reason=_UNAVAILABLE_REASON,
        )
    )

    providers.append(
        AIProviderResponse(
            id="anthropic",
            name="Anthropic",
            logo="/logos/anthropic.png",
            models=ANTHROPIC_MODELS,
            available=False,
            unavailable_reason=_UNAVAILABLE_REASON,
        )
    )

    return providers


@router.get("/ollama/health")
async def check_ollama_health():
    """Check if Ollama is running and accessible"""
    is_healthy = await ollama_service.health_check()
    return {"status": "healthy" if is_healthy else "unavailable", "healthy": is_healthy}


@router.get("/ollama/models", response_model=list[AIModelResponse])
async def get_ollama_models():
    """Get available Ollama models"""
    models = await ollama_service.list_models()
    if not models:
        raise HTTPException(status_code=503, detail="Ollama service unavailable")
    return models
