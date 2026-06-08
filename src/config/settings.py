"""
Configuration management via environment variables.
NO API keys are hardcoded — all secrets come from .env or system env vars.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # LLM Configuration
    llm_provider: str = Field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "deepseek")
    )

    # DeepSeek
    deepseek_api_key: str = Field(
        default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", "")
    )
    deepseek_base_url: str = Field(
        default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )
    deepseek_model: str = Field(
        default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    )

    # OpenAI (alternative)
    openai_api_key: str = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    openai_base_url: str = Field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    openai_model: str = Field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o")
    )

    # Application
    log_level: str = Field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    data_dir: str = Field(
        default_factory=lambda: os.getenv("DATA_DIR", str(_PROJECT_ROOT / "data"))
    )

    # Model parameters
    temperature: float = 0.7
    max_tokens: int = 2048

    @property
    def llm_api_key(self) -> str:
        """Get the active LLM API key."""
        if self.llm_provider == "openai":
            return self.openai_api_key
        return self.deepseek_api_key

    @property
    def llm_base_url(self) -> str:
        """Get the active LLM base URL."""
        if self.llm_provider == "openai":
            return self.openai_base_url
        return self.deepseek_base_url

    @property
    def llm_model(self) -> str:
        """Get the active LLM model name."""
        if self.llm_provider == "openai":
            return self.openai_model
        return self.deepseek_model

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton
settings = Settings()
