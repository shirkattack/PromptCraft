from fastapi import APIRouter

from app.api.v1.endpoints import datasets, providers, sessions, training

api_router = APIRouter()

api_router.include_router(sessions.router, prefix="/sessions", tags=["sessions"])
api_router.include_router(
    datasets.router, prefix="/datasets", tags=["datasets"]
)  # Keep old datasets for compatibility
api_router.include_router(
    training.router, prefix="/training", tags=["training"]
)  # New enhanced training endpoints
api_router.include_router(providers.router, prefix="/providers", tags=["providers"])
