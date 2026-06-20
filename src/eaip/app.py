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
from eaip.retrieval import Principal
from eaip.retrieval.service import RetrievalService


@lru_cache
def _provider() -> LLMProvider:
    """Cached provider dependency (constructed once per process)."""
    return get_provider()


@lru_cache
def _retrieval_service() -> RetrievalService:
    """Cached retrieval service (opens the index + builds the stack once)."""
    return RetrievalService.from_settings()


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

    @app.post("/ask", tags=["retrieval"])
    def ask(
        req: AskRequest,
        service: RetrievalService = Depends(_retrieval_service),
    ) -> AskResponse:
        """Answer a question with permission-aware, grounded RAG.

        The request carries the asking principal (user + groups). Retrieval is
        scoped to that principal's ACL, so the response can only cite documents
        the user is allowed to see. (Phase 4 replaces the body-supplied principal
        with authenticated identity + RBAC.)
        """
        principal = Principal.of(user=req.user, groups=req.groups)
        result = service.ask(req.query, principal)
        answer = result.answer
        return AskResponse(
            answer=answer.text,
            abstained=answer.abstained,
            top_score=round(answer.top_score, 4),
            citations=[
                CitationModel(label=c.label, doc_id=c.doc_id, title=c.title, source=c.source)
                for c in answer.citations
            ],
            timings_ms={k: round(v, 2) for k, v in result.retrieval.timings_ms.items()},
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


class AskRequest(BaseModel):
    """Input for ``/ask``: the question and the asking principal."""

    query: str
    user: str
    groups: list[str] = []


class CitationModel(BaseModel):
    """A cited source passage (API shape)."""

    label: int
    doc_id: str
    title: str
    source: str


class AskResponse(BaseModel):
    """Grounded answer plus citations, abstention flag, and retrieval timings."""

    answer: str
    abstained: bool
    top_score: float
    citations: list[CitationModel]
    timings_ms: dict[str, float]


# Module-level app instance for `uvicorn eaip.app:app`.
app = create_app()
