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


class EmbedderName(StrEnum):
    """Supported dense embedders, selected via ``EAIP_EMBEDDER``.

    ``hashing`` is offline + deterministic (default, used by tests). ``bge`` is a
    real sentence-transformers model for actual semantic quality.
    """

    HASHING = "hashing"
    BGE = "bge"


class RerankerName(StrEnum):
    """Supported rerankers, selected via ``EAIP_RERANKER``.

    ``lexical`` is offline + deterministic (default, used by tests). ``bge`` is a
    real cross-encoder for actual relevance quality.
    """

    LEXICAL = "lexical"
    BGE = "bge"


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

    # --- Embeddings ---
    embedder: EmbedderName = Field(
        default=EmbedderName.HASHING,
        description="Dense embedder backend. Defaults to the offline hashing embedder.",
    )
    embedding_dim: int = Field(
        default=256,
        description="Vector dimension for the hashing embedder (ignored by BGE).",
    )
    bge_model: str = Field(default="BAAI/bge-small-en-v1.5")

    # --- Retrieval / reranking ---
    reranker: RerankerName = Field(
        default=RerankerName.LEXICAL,
        description="Reranker backend. Defaults to the offline lexical reranker.",
    )
    bge_reranker_model: str = Field(default="BAAI/bge-reranker-base")
    retrieval_shortlist: int = Field(
        default=20,
        description="Top-N candidates pulled from each retrieval arm and fed to the reranker.",
    )
    retrieval_top_k: int = Field(
        default=5,
        description="Final number of reranked chunks used to ground the answer.",
    )
    answer_min_score: float = Field(
        default=0.05,
        description="Minimum top reranked score to attempt an answer; below it, abstain "
        "('I don't know') instead of grounding on weak context.",
    )

    # --- Vector store (Qdrant) ---
    qdrant_path: Path = Field(
        default=Path("./data/qdrant"),
        description="On-disk path for the embedded Qdrant store. Use ':memory:' for ephemeral.",
    )
    qdrant_collection: str = Field(
        default="eaip_chunks",
        description="Collection name *prefix*. Each tenant gets its own collection "
        "named '<prefix>__<tenant_id>' for physical isolation (Phase 4).",
    )

    # --- State store (SQLite by default; Postgres-swappable) ---
    state_db_path: Path = Field(
        default=Path("./data/eaip.db"),
        description="SQLite database file for durable platform state (ingestion watermarks, "
        "and in later phases audit logs, prompt registry, tenant config).",
    )

    # --- Orchestration (Phase 3) ---
    checkpoint_db_path: Path = Field(
        default=Path("./data/checkpoints.sqlite"),
        description="SQLite file for the LangGraph checkpointer (durable graph state + HITL).",
    )
    agent_max_iterations: int = Field(
        default=6,
        description="Hard cap on agent-loop steps before the run is stopped (loop safety).",
    )
    agent_token_budget: int = Field(
        default=20_000,
        description="Per-run token budget; exceeding it stops the run (loop safety).",
    )
    agent_time_budget_s: float = Field(
        default=30.0,
        description="Per-run wall-clock budget in seconds; exceeding it stops the run.",
    )
    critic_max_revisions: int = Field(
        default=2,
        description="Max draft->critic revision cycles before finalizing (critic loop bound).",
    )
    approval_ttl_s: float = Field(
        default=3600.0,
        description="How long a pending HITL approval stays valid before it expires.",
    )

    # --- Platform capabilities (Phase 4) ---
    default_tenant: str = Field(
        default="acme",
        description="Tenant id used when a request doesn't specify one (single-tenant dev).",
    )
    tenant_requests_per_minute: int = Field(
        default=60,
        description="Default per-tenant request rate limit (requests/min).",
    )
    tenant_daily_token_budget: int = Field(
        default=1_000_000,
        description="Default per-tenant daily token budget; over it, requests are throttled.",
    )

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
