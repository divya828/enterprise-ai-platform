"""Typed application settings.

Why a single typed settings object instead of scattered ``os.getenv`` calls:
a config layer is one of the platform concepts we want to teach. Centralizing
configuration gives us one place to document every knob, validate it at startup
(fail fast on a bad value rather than deep inside a request), and swap the
backing store later (env today; a per-tenant config table in Phase 4).

We use ``pydantic-settings`` so that env vars are parsed and validated against
type hints, with a clear precedence: explicit kwargs > environment > .env file
> defaults.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(StrEnum):
    """Supported LLM backends, selected via ``EAIP_LLM_PROVIDER``.

    The provider abstraction is itself a concept we teach (provider strategy /
    avoiding vendor lock-in): the rest of the platform depends only on the
    :class:`~eaip.providers.base.LLMProvider` protocol, never on a concrete SDK.
    """

    STUB = "stub"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class Settings(BaseSettings):
    """All runtime configuration in one validated object.

    Env vars are prefixed ``EAIP_`` (except third-party credentials such as
    ``ANTHROPIC_API_KEY`` which keep their conventional names so existing SDK
    docs apply).
    """

    model_config = SettingsConfigDict(
        env_prefix="EAIP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # tolerate unrelated env vars in the shell
    )

    # --- LLM provider ---
    llm_provider: LLMProvider = Field(
        default=LLMProvider.STUB,
        description="Which LLM backend to use. Defaults to the offline stub.",
    )
    llm_model: str = Field(
        default="claude-opus-4-8",
        description="Model name handed to the active provider (ignored by stub).",
    )
    ollama_base_url: str = Field(default="http://localhost:11434")

    # Third-party credentials. Aliased to their conventional names so the env
    # var is e.g. ANTHROPIC_API_KEY, not EAIP_ANTHROPIC_API_KEY.
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")

    # --- Runtime ---
    data_dir: Path = Field(default=Path("./data"))
    log_level: str = Field(default="INFO")


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance.

    Cached so configuration is parsed once. Tests that need a different config
    can call ``get_settings.cache_clear()`` after setting env vars.
    """
    return Settings()
