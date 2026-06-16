"""FastAPI application — the platform's HTTP surface.

Phase 0 keeps this deliberately tiny: a health endpoint and a ``/hello`` endpoint
that proves the LLM provider abstraction works end to end. Later phases add
ingestion, retrieval, orchestration, and platform-capability routers here.

The app uses FastAPI's dependency injection for the provider so tests can
override it with a scripted stub without touching global state.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from eaip import __version__
from eaip.config import get_settings
from eaip.providers import LLMProvider, Message, Role, get_provider


@lru_cache
def _provider() -> LLMProvider:
    """Cached provider dependency (constructed once per process)."""
    return get_provider()


def create_app() -> FastAPI:
    """Application factory.

    A factory (rather than a module-level ``app``) lets tests build a fresh app
    and override dependencies cleanly — a small thing now that pays off as the
    surface grows.
    """
    app = FastAPI(
        title="Enterprise AI Platform (learning reference)",
        version=__version__,
        summary="A runnable, laptop-scale reference implementation of an enterprise AI agent platform.",
    )

    @app.get("/health", tags=["meta"])
    def health() -> dict[str, str]:
        """Liveness probe — does not call the LLM."""
        settings = get_settings()
        return {
            "status": "ok",
            "version": __version__,
            "provider": settings.llm_provider.value,
        }

    @app.post("/hello", tags=["meta"])
    def hello(req: HelloRequest, provider: LLMProvider = Depends(_provider)) -> HelloResponse:
        """Round-trip a prompt through the configured provider.

        This is the Phase 0 'definition of done' in HTTP form: with the default
        stub provider it returns a deterministic completion with no model or key.
        """
        completion = provider.complete(
            [Message(role=Role.USER, content=req.message)],
            max_tokens=req.max_tokens,
        )
        return HelloResponse(
            reply=completion.text,
            provider=provider.name,
            model=completion.model,
            total_tokens=completion.usage.total_tokens,
        )

    return app


class HelloRequest(BaseModel):
    """Input for the ``/hello`` smoke endpoint."""

    message: str
    max_tokens: int = 256


class HelloResponse(BaseModel):
    """Output of the ``/hello`` smoke endpoint."""

    reply: str
    provider: str
    model: str
    total_tokens: int


# Module-level app instance for `uvicorn eaip.app:app`.
app = create_app()
