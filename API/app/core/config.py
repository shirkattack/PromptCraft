from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Database
    database_url: str = "sqlite:///./app.db"
    
    # Ollama Configuration (Primary AI Provider)
    ollama_base_url: str = "http://localhost:11434"
    
    # Default settings optimized for Ollama
    default_model_name: str = "llama3.2:latest"
    default_model_provider: str = "ollama"
    default_temperature: float = 0.7
    default_max_tokens: int = 1000
    default_synthetic_data_size: int = 10
    default_train_ratio: float = 0.8
    
    # Ollama-specific settings
    ollama_timeout: int = 120  # Longer timeout for local models
    ollama_max_retries: int = 3
    ollama_keep_alive: str = "5m"  # Keep model loaded for 5 minutes
    
    # Logging
    log_level: str = "INFO"
    
    # CORS
    cors_origins: list[str] = ["http://localhost:3000"]
    
    # App settings
    app_name: str = "PromptCraft API - Ollama Edition"
    debug: bool = False
    
    class Config:
        env_file = ".env"
        extra = "ignore"  # Ignore extra fields from .env file

settings = Settings()