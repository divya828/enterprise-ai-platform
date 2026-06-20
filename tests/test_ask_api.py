"""Tests for the /ask HTTP endpoint (permission-aware grounded RAG over HTTP)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from eaip.app import _retrieval_service, create_app
from eaip.providers.stub import StubProvider
from eaip.providers.types import Completion
from eaip.retrieval.answerer import GroundedAnswerer
from eaip.retrieval.dense import DenseRetriever
from eaip.retrieval.pipeline import HybridRetriever
from eaip.retrieval.reranker import LexicalReranker
from eaip.retrieval.service import RetrievalService
from eaip.retrieval.sparse import SparseRetriever


def _client(ingested_index, embedder, provider=None):
    """An app whose retrieval service uses the in-memory ingested index."""
    retriever = HybridRetriever(
        DenseRetriever(ingested_index, embedder),
        SparseRetriever(ingested_index),
        LexicalReranker(),
        shortlist=20,
        top_k=5,
    )
    answerer = GroundedAnswerer(provider or StubProvider(), min_score=0.05)
    service = RetrievalService(ingested_index, retriever, answerer)

    app = create_app()
    app.dependency_overrides[_retrieval_service] = lambda: service
    return TestClient(app)


def test_ask_returns_answer_with_citations_and_timings(ingested_index, embedder):
    provider = StubProvider([Completion(text="Download GlobalConnect [1].")])
    client = _client(ingested_index, embedder, provider)

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
    client = _client(ingested_index, embedder)
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
    client = _client(ingested_index, embedder, provider)
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
