"""Tests for the /ask HTTP endpoint (permission-aware grounded RAG over HTTP)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from eaip.app import PlatformContext, _platform, _retrieval_service, create_app
from eaip.index.resolver import SingleIndexResolver
from eaip.platform.limits import RateLimiter, TokenBudget
from eaip.providers.stub import StubProvider
from eaip.providers.types import Completion
from eaip.retrieval.answerer import GroundedAnswerer
from eaip.retrieval.pipeline import HybridRetriever
from eaip.retrieval.reranker import LexicalReranker
from eaip.retrieval.service import RetrievalService
from eaip.storage import InMemoryStateStore


def _client(ingested_index, embedder, provider=None, *, rpm=60, daily_tokens=1_000_000):
    """An app whose retrieval + platform deps use in-memory stores.

    Returns ``(client, platform_context)`` so governance tests can inspect the
    audit log / usage and tune limits.
    """
    retriever = HybridRetriever(
        SingleIndexResolver(ingested_index),
        embedder,
        LexicalReranker(),
        shortlist=20,
        top_k=5,
    )
    answerer = GroundedAnswerer(provider or StubProvider(), min_score=0.05)
    service = RetrievalService(retriever, answerer)

    store = InMemoryStateStore()
    platform = PlatformContext(
        store=store,
        rate_limiter=RateLimiter(rpm),
        token_budget=TokenBudget(store, daily_tokens),
    )

    app = create_app()
    app.dependency_overrides[_retrieval_service] = lambda: service
    app.dependency_overrides[_platform] = lambda: platform
    return TestClient(app), platform


def test_ask_returns_answer_with_citations_and_timings(ingested_index, embedder):
    provider = StubProvider([Completion(text="Download GlobalConnect [1].")])
    client, _platform_ctx = _client(ingested_index, embedder, provider)

    resp = client.post(
        "/ask",
        json={"query": "how do I set up the vpn", "user": "u@acme.test", "groups": ["everyone"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Download GlobalConnect [1]."
    assert not body["abstained"]
    assert len(body["citations"]) >= 1
    assert "rerank_ms" in body["timings_ms"]


def test_ask_does_not_leak_restricted_doc_over_http(ingested_index, embedder):
    """An 'everyone' user asking a finance question never gets a finance citation."""
    client, _platform_ctx = _client(ingested_index, embedder)
    resp = client.post(
        "/ask",
        json={
            "query": "what is the FY26 revenue forecast",
            "user": "intern@acme.test",
            "groups": ["everyone"],
        },
    )
    assert resp.status_code == 200
    cited = {c["doc_id"] for c in resp.json()["citations"]}
    assert "CONF-7" not in cited


def test_ask_authorized_user_can_reach_restricted_doc(ingested_index, embedder):
    provider = StubProvider([Completion(text="ARR is about 42 million [1].")])
    client, _platform_ctx = _client(ingested_index, embedder, provider)
    resp = client.post(
        "/ask",
        json={
            "query": "what is the FY26 revenue forecast annual recurring revenue",
            "user": "cfo@acme.test",
            "groups": ["finance"],
        },
    )
    assert resp.status_code == 200
    cited = {c["doc_id"] for c in resp.json()["citations"]}
    assert "CONF-7" in cited


# --- Phase 4 governance over HTTP -------------------------------------------


def test_ask_records_an_audit_event(ingested_index, embedder):
    client, platform = _client(ingested_index, embedder)
    client.post("/ask", json={"query": "vpn", "user": "u@acme.test", "groups": ["everyone"]})
    events = platform.store.events(tenant="acme")
    assert any(e.action == "ask" and e.actor == "u@acme.test" for e in events)


def test_ask_attributes_token_usage_to_the_tenant(ingested_index, embedder):
    from datetime import UTC, datetime

    client, platform = _client(ingested_index, embedder)
    client.post("/ask", json={"query": "vpn", "user": "u@acme.test", "tenant": "acme"})
    day = datetime.now(UTC).date().isoformat()
    totals = platform.store.usage_for_day("acme", day)
    assert totals.requests == 1
    assert totals.tokens > 0  # cost attribution recorded


def test_ask_rate_limit_returns_429(ingested_index, embedder):
    client, _platform_ctx = _client(ingested_index, embedder, rpm=1)
    body = {"query": "vpn", "user": "u@acme.test", "groups": ["everyone"]}
    assert client.post("/ask", json=body).status_code == 200
    resp = client.post("/ask", json=body)  # second within the minute
    assert resp.status_code == 429
    assert "rate limit" in resp.json()["detail"]


def test_ask_token_budget_returns_429(ingested_index, embedder):
    client, _platform_ctx = _client(ingested_index, embedder, daily_tokens=1)
    # First request records usage that blows the tiny budget; the next is throttled.
    client.post("/ask", json={"query": "vpn", "user": "u@acme.test"})
    resp = client.post("/ask", json={"query": "vpn again", "user": "u@acme.test"})
    assert resp.status_code == 429
    assert "token budget" in resp.json()["detail"]


def test_ask_denies_a_role_without_the_ask_capability(ingested_index, embedder):
    """An unknown/invalid role can't ask — defense at the API boundary (403)."""
    client, _platform_ctx = _client(ingested_index, embedder)
    resp = client.post(
        "/ask",
        json={"query": "vpn", "user": "u@acme.test", "role": "guest"},  # not a real role
    )
    # 'guest' has no privileges (fail closed) -> a clean 403, not a 500.
    assert resp.status_code == 403
