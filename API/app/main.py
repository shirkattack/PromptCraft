from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.database import create_tables
from app.core.logging import setup_logging, get_logger
from app.core.exceptions import PromptCraftException

# Setup logging
setup_logging()
logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan events."""
    # Startup
    logger.info("Starting PromptCraft API...")

    # Import all models to ensure they're registered
    from app.models import optimization, training
    create_tables()
    logger.info("Database tables created/verified")

    yield

    # Shutdown
    logger.info("Shutting down PromptCraft API...")


app = FastAPI(
    title="PromptCraft API",
    description="FastAPI backend for PromptCraft - AI-powered prompt optimization",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "*.localhost"]
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins + ["http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

# Global exception handler
@app.exception_handler(PromptCraftException)
async def promptcraft_exception_handler(request, exc: PromptCraftException):
    """Handle custom PromptCraft exceptions."""
    logger.error(
        f"PromptCraft error: {exc.message}",
        extra={
            "error_code": exc.error_code,
            "details": exc.details,
            "path": request.url.path
        }
    )
    return JSONResponse(
        status_code=400,
        content={
            "error": exc.message,
            "error_code": exc.error_code,
            "details": exc.details
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions with logging."""
    logger.warning(
        f"HTTP {exc.status_code}: {exc.detail}",
        extra={"path": request.url.path, "status_code": exc.status_code}
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.get("/", tags=["System"])
async def root():
    """API information endpoint."""
    return {
        "message": "PromptCraft API",
        "version": "1.0.0",
        "description": "AI-powered prompt optimization platform",
        "docs_url": "/docs" if settings.debug else "Documentation disabled in production"
    }


@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint for monitoring."""
    try:
        # You could add database connectivity check here
        return {
            "status": "healthy",
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.options("/{full_path:path}", include_in_schema=False)
async def options_handler():
    """Handle CORS preflight requests."""
    return {"message": "OK"}