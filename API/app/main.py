from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app import __version__
from app.api.v1.router import api_router
from app.core.config import settings
from app.core.database import create_tables, engine
from app.core.exceptions import PromptCraftException
from app.core.logging import get_logger, setup_logging
from app.core.migrations import run_migrations
from app.services.ollama_service import ollama_service

# Setup logging
setup_logging()
logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan events."""
    # Startup
    logger.info("Starting PromptCraft API...")

    # Import all models to ensure they're registered
    from app.models import optimization, training  # noqa: F401

    # Migrations first so the schema is current before create_all fills in any
    # table the migrations do not know about (there are none today; it is a
    # safety net for a database wiped between releases).
    run_migrations()
    create_tables()
    logger.info("Database schema migrated and verified")

    yield

    # Shutdown
    logger.info("Shutting down PromptCraft API...")
    await ollama_service.aclose()


app = FastAPI(
    title="PromptCraft API",
    description="FastAPI backend for PromptCraft - AI-powered prompt optimization",
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Security middleware. "*" means "accept any Host", so skip the check entirely
# rather than paying for a middleware that can never reject anything.
if "*" not in settings.allowed_hosts:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")


# Global exception handler
@app.exception_handler(PromptCraftException)
async def promptcraft_exception_handler(
    request: Request, exc: PromptCraftException
) -> JSONResponse:
    """Handle custom PromptCraft exceptions."""
    logger.error(
        f"PromptCraft error: {exc.message}",
        extra={
            "error_code": exc.error_code,
            "details": exc.details,
            "path": request.url.path,
        },
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.message,
            "error_code": exc.error_code,
            "details": exc.details,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions with logging."""
    logger.warning(
        f"HTTP {exc.status_code}: {exc.detail}",
        extra={"path": request.url.path, "status_code": exc.status_code},
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
        headers=getattr(exc, "headers", None),
    )


@app.get("/", tags=["System"])
async def root() -> dict[str, str]:
    """API information endpoint."""
    return {
        "message": "PromptCraft API",
        "version": __version__,
        "description": "AI-powered prompt optimization platform",
        "docs_url": (
            "/docs" if settings.debug else "Documentation disabled in production"
        ),
    }


@app.get("/health", tags=["System"])
async def health_check() -> dict[str, Any]:
    """Health check endpoint for monitoring."""
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy") from e

    return {
        "status": "healthy",
        "timestamp": datetime.now(UTC).isoformat(),
        "version": __version__,
        "database": "connected",
    }
