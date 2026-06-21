"""FastAPI application — the platform's HTTP surface.

Phase 0 keeps this deliberately tiny: a health endpoint and a ``/hello`` endpoint
that proves the LLM provider abstraction works end to end. Later phases add
ingestion, retrieval, orchestration, and platform-capability routers here.

The app uses FastAPI's dependency injection for the provider so tests can
override it with a scripted stub without touching global state.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from eaip import __version__
from eaip.config import get_settings
from eaip.platform.limits import RateLimiter, TokenBudget
from eaip.platform.rbac import Capability, PermissionDenied, require
from eaip.providers import LLMProvider, Message, Role, get_provider
from eaip.retrieval import Principal
from eaip.retrieval.service import RetrievalService
from eaip.storage import AuditEvent, SqliteStateStore


@lru_cache
def _provider() -> LLMProvider:
    """Cached provider dependency (constructed once per process)."""
    return get_provider()


@lru_cache
def _retrieval_service() -> RetrievalService:
    """Cached retrieval service (opens the index + builds the stack once)."""
    return RetrievalService.from_settings()


@dataclass
class PlatformContext:
    """The per-process platform governance dependencies (store + limits)."""

    store: SqliteStateStore
    rate_limiter: RateLimiter
    token_budget: TokenBudget


@lru_cache
def _platform() -> PlatformContext:
    """Cached platform context: durable store + per-tenant rate limit + budget."""
    settings = get_settings()
    store = SqliteStateStore(settings.state_db_path)
    return PlatformContext(
        store=store,
        rate_limiter=RateLimiter(settings.tenant_requests_per_minute),
        token_budget=TokenBudget(store, settings.tenant_daily_token_budget),
    )


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
        platform: PlatformContext = Depends(_platform),
    ) -> AskResponse:
        """Answer a question with tenant-scoped, RBAC-checked, grounded RAG.

        The governance pipeline, in order:
        1. RBAC — the role must have the ASK capability (else 403).
        2. Rate limit + token budget — over either, the tenant is throttled (429).
        3. Retrieve + answer — scoped to the principal's tenant collection + ACL.
        4. Record usage (cost attribution) and append an audit event.

        (Identity is body-supplied here for the demo; a real deployment derives
        tenant/user/role from an authenticated token.)
        """
        principal = Principal.of(user=req.user, groups=req.groups, tenant=req.tenant, role=req.role)
        day = datetime.now(UTC).date().isoformat()

        # 1. RBAC
        try:
            require(principal.role, Capability.ASK)
        except PermissionDenied as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

        # 2. Rate limit + token budget
        rl = platform.rate_limiter.check_and_record(principal.tenant)
        if not rl.allowed:
            raise HTTPException(status_code=429, detail=rl.reason)
        budget = platform.token_budget.check(principal.tenant, day)
        if not budget.allowed:
            raise HTTPException(status_code=429, detail=budget.reason)

        # 3. Retrieve + answer
        result = service.ask(req.query, principal)
        answer = result.answer

        # 4. Cost attribution + audit (append-only)
        tokens = _estimate_tokens(req.query, answer.text)
        platform.token_budget.record(principal.tenant, tokens=tokens, day=day)
        now = datetime.now(UTC).isoformat()
        platform.store.append_event(
            AuditEvent(
                tenant=principal.tenant,
                actor=principal.user,
                action="ask",
                target=req.query[:80],
                detail=f"abstained={answer.abstained} tokens={tokens}",
                created_at=now,
            )
        )

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


def _estimate_tokens(query: str, answer: str) -> int:
    """Rough token estimate (~4 chars/token) for cost attribution."""
    return (len(query) + len(answer)) // 4 + 1


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
    tenant: str = "acme"
    role: str = "viewer"


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
