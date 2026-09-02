from pydantic_settings import BaseSettings, SettingsConfigDict


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
    # Dataset-driven optimization: caps on how many samples are used so a
    # large dataset does not turn one request into hundreds of local model calls.
    eval_max_train_samples: int = 40
    eval_max_dev_samples: int = 20
    eval_max_demos: int = 8

    # Ollama-specific settings
    ollama_timeout: int = 120  # Longer timeout for local models
    ollama_max_retries: int = 3
    ollama_keep_alive: str = "5m"  # Keep model loaded for 5 minutes

    # Logging
    log_level: str = "INFO"

    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    # Trusted hosts for the Host header check. "*" disables the check, which is
    # the sensible default for a locally bound dev server; restrict it (e.g.
    # ALLOWED_HOSTS=["api.example.com"]) when deploying behind a public name.
    allowed_hosts: list[str] = ["*"]

    # App settings
    app_name: str = "PromptCraft API - Ollama Edition"
    debug: bool = False

    # Pagination guard rails
    max_page_size: int = 500

    # How many optimization results to keep in the in-process history buffer.
    optimization_history_size: int = 100

    # Security
    api_key: str | None = None  # If set, API key authentication is required
    require_api_key: bool = False  # Set to True to enforce API key auth

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",  # Ignore extra fields from .env file
    )


settings = Settings()
