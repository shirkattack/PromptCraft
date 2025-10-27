from fastapi import APIRouter, HTTPException
from typing import List
from app.schemas.optimization import AIProviderResponse, AIModelResponse
from app.services.ollama_service import ollama_service

router = APIRouter()

@router.get("/", response_model=List[AIProviderResponse])
async def get_providers():
    """Get all available AI providers and their models"""
    providers = []
    
    # Ollama provider
    ollama_models = await ollama_service.list_models()
    if ollama_models:
        ollama_provider = AIProviderResponse(
            id="ollama",
            name="Ollama",
            logo="/logos/ollama.png",
            models=ollama_models
        )
        providers.append(ollama_provider)
    
    # OpenAI provider (static data for now)
    openai_models = [
        AIModelResponse(
            id="gpt-4",
            name="GPT-4",
            context_window=8192,
            cost_per_1k_tokens=0.03,
            speed_rating=3,
            best_use_case="Complex reasoning and analysis",
            is_free=False
        ),
        AIModelResponse(
            id="gpt-3.5-turbo",
            name="GPT-3.5 Turbo",
            context_window=4096,
            cost_per_1k_tokens=0.002,
            speed_rating=5,
            best_use_case="Fast general purpose tasks",
            is_free=False
        )
    ]
    
    openai_provider = AIProviderResponse(
        id="openai",
        name="OpenAI",
        logo="/logos/openai.png",
        models=openai_models
    )
    providers.append(openai_provider)
    
    # Anthropic provider
    anthropic_models = [
        AIModelResponse(
            id="claude-3-opus",
            name="Claude 3 Opus",
            context_window=200000,
            cost_per_1k_tokens=0.015,
            speed_rating=2,
            best_use_case="Complex analysis and creative tasks",
            is_free=False
        ),
        AIModelResponse(
            id="claude-3-sonnet",
            name="Claude 3 Sonnet",
            context_window=200000,
            cost_per_1k_tokens=0.003,
            speed_rating=4,
            best_use_case="Balanced performance and cost",
            is_free=False
        )
    ]
    
    anthropic_provider = AIProviderResponse(
        id="anthropic",
        name="Anthropic",
        logo="/logos/anthropic.png",
        models=anthropic_models
    )
    providers.append(anthropic_provider)
    
    return providers

@router.get("/ollama/health")
async def check_ollama_health():
    """Check if Ollama is running and accessible"""
    is_healthy = await ollama_service.health_check()
    return {"status": "healthy" if is_healthy else "unavailable", "healthy": is_healthy}

@router.get("/ollama/models", response_model=List[AIModelResponse])
async def get_ollama_models():
    """Get available Ollama models"""
    models = await ollama_service.list_models()
    if not models:
        raise HTTPException(status_code=503, detail="Ollama service unavailable")
    return models