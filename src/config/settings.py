"""
Enterprise-grade configuration management via environment variables.
NO API keys are hardcoded — all secrets come from .env or system env vars.

Supports:
- Multi-provider LLM (DeepSeek, OpenAI, Anthropic, Ollama)
- ChromaDB vector store
- Langfuse observability
- MCP server configuration
- Rate limiting & retry policies
- Feature flags
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    """Application settings loaded from environment variables with validation."""

    # ═══════════════════════════════════════════════════════════════════════════
    # LLM Provider Configuration
    # ═══════════════════════════════════════════════════════════════════════════

    llm_provider: Literal["deepseek", "openai", "anthropic", "ollama"] = Field(
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

    # OpenAI
    openai_api_key: str = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    openai_base_url: str = Field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    openai_model: str = Field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o")
    )

    # Anthropic
    anthropic_api_key: str = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )
    anthropic_model: str = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    )

    # Ollama (local)
    ollama_base_url: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    )
    ollama_model: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    )

    @property
    def llm_api_key(self) -> str:
        """Get the active LLM API key."""
        provider_map = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "ollama": "ollama",  # Ollama doesn't need a key
            "deepseek": self.deepseek_api_key,
        }
        return provider_map.get(self.llm_provider, self.deepseek_api_key)

    @property
    def llm_base_url(self) -> str:
        """Get the active LLM base URL."""
        provider_map = {
            "openai": self.openai_base_url,
            "anthropic": "https://api.anthropic.com",
            "ollama": self.ollama_base_url,
            "deepseek": self.deepseek_base_url,
        }
        return provider_map.get(self.llm_provider, self.deepseek_base_url)

    @property
    def llm_model(self) -> str:
        """Get the active LLM model name."""
        provider_map = {
            "openai": self.openai_model,
            "anthropic": self.anthropic_model,
            "ollama": self.ollama_model,
            "deepseek": self.deepseek_model,
        }
        return provider_map.get(self.llm_provider, self.deepseek_model)

    # ═══════════════════════════════════════════════════════════════════════════
    # Model Parameters
    # ═══════════════════════════════════════════════════════════════════════════

    temperature: float = Field(
        default_factory=lambda: float(os.getenv("TEMPERATURE", "0.7"))
    )
    max_tokens: int = Field(
        default_factory=lambda: int(os.getenv("MAX_TOKENS", "4096"))
    )
    streaming: bool = Field(
        default_factory=lambda: os.getenv("STREAMING", "true").lower() == "true"
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # Memory & Vector Store
    # ═══════════════════════════════════════════════════════════════════════════

    data_dir: str = Field(
        default_factory=lambda: os.getenv("DATA_DIR", str(_PROJECT_ROOT / "data"))
    )

    # ChromaDB
    chroma_persist_dir: str = Field(
        default_factory=lambda: os.getenv(
            "CHROMA_PERSIST_DIR", str(_PROJECT_ROOT / "data" / "chroma")
        )
    )
    chroma_collection_name: str = Field(
        default_factory=lambda: os.getenv("CHROMA_COLLECTION", "health_memory")
    )
    embedding_model: str = Field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    )

    # Memory limits
    short_term_memory_limit: int = Field(
        default_factory=lambda: int(os.getenv("SHORT_TERM_MEMORY_LIMIT", "20"))
    )
    vector_memory_top_k: int = Field(
        default_factory=lambda: int(os.getenv("VECTOR_MEMORY_TOP_K", "5"))
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # Observability (Langfuse)
    # ═══════════════════════════════════════════════════════════════════════════

    langfuse_enabled: bool = Field(
        default_factory=lambda: os.getenv("LANGFUSE_ENABLED", "false").lower() == "true"
    )
    langfuse_public_key: str = Field(
        default_factory=lambda: os.getenv("LANGFUSE_PUBLIC_KEY", "")
    )
    langfuse_secret_key: str = Field(
        default_factory=lambda: os.getenv("LANGFUSE_SECRET_KEY", "")
    )
    langfuse_host: str = Field(
        default_factory=lambda: os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # MCP (Model Context Protocol)
    # ═══════════════════════════════════════════════════════════════════════════

    mcp_enabled: bool = Field(
        default_factory=lambda: os.getenv("MCP_ENABLED", "true").lower() == "true"
    )
    mcp_health_knowledge_server: str = Field(
        default_factory=lambda: os.getenv(
            "MCP_HEALTH_KNOWLEDGE_SERVER", "http://localhost:8001"
        )
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # Resilience & Middleware
    # ═══════════════════════════════════════════════════════════════════════════

    max_retries: int = Field(
        default_factory=lambda: int(os.getenv("MAX_RETRIES", "3"))
    )
    retry_backoff: float = Field(
        default_factory=lambda: float(os.getenv("RETRY_BACKOFF", "2.0"))
    )
    retry_max_delay: float = Field(
        default_factory=lambda: float(os.getenv("RETRY_MAX_DELAY", "60.0"))
    )
    request_timeout: float = Field(
        default_factory=lambda: float(os.getenv("REQUEST_TIMEOUT", "120.0"))
    )
    rate_limit_per_minute: int = Field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # Agent Configuration
    # ═══════════════════════════════════════════════════════════════════════════

    agent_max_tool_rounds: int = Field(
        default_factory=lambda: int(os.getenv("AGENT_MAX_TOOL_ROUNDS", "5"))
    )
    agent_reflection_enabled: bool = Field(
        default_factory=lambda: os.getenv("AGENT_REFLECTION_ENABLED", "true").lower() == "true"
    )
    agent_confidence_threshold: float = Field(
        default_factory=lambda: float(os.getenv("AGENT_CONFIDENCE_THRESHOLD", "0.6"))
    )
    agent_parallel_execution: bool = Field(
        default_factory=lambda: os.getenv("AGENT_PARALLEL_EXECUTION", "true").lower() == "true"
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # API Server
    # ═══════════════════════════════════════════════════════════════════════════

    api_host: str = Field(
        default_factory=lambda: os.getenv("API_HOST", "0.0.0.0")
    )
    api_port: int = Field(
        default_factory=lambda: int(os.getenv("API_PORT", "8000"))
    )
    api_workers: int = Field(
        default_factory=lambda: int(os.getenv("API_WORKERS", "4"))
    )
    cors_origins: str = Field(
        default_factory=lambda: os.getenv("CORS_ORIGINS", "*")
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # Feature Flags
    # ═══════════════════════════════════════════════════════════════════════════

    feature_rag: bool = Field(
        default_factory=lambda: os.getenv("FEATURE_RAG", "true").lower() == "true"
    )
    feature_mcp: bool = Field(
        default_factory=lambda: os.getenv("FEATURE_MCP", "true").lower() == "true"
    )
    feature_streaming: bool = Field(
        default_factory=lambda: os.getenv("FEATURE_STREAMING", "true").lower() == "true"
    )
    feature_evaluation: bool = Field(
        default_factory=lambda: os.getenv("FEATURE_EVALUATION", "true").lower() == "true"
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # Logging
    # ═══════════════════════════════════════════════════════════════════════════

    log_level: str = Field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    log_format: Literal["text", "json"] = Field(
        default_factory=lambda: os.getenv("LOG_FORMAT", "text")
    )
    log_file: Optional[str] = Field(
        default_factory=lambda: os.getenv("LOG_FILE", None)
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # Validators
    # ═══════════════════════════════════════════════════════════════════════════

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        return max(0.0, min(2.0, v))

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int) -> int:
        return max(1, min(128000, v))

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Singleton
settings = Settings()
