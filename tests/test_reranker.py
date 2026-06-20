"""Tests for the lexical reranker and the shortlist→rerank pattern."""

from __future__ import annotations

from datetime import UTC, datetime

from eaip.index.store import ScoredChunk
from eaip.ingestion.models import ACL, Chunk, SourceType
from eaip.retrieval.reranker import LexicalReranker


def _sc(cid: str, text: str, score: float = 0.0) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=cid,
        doc_id=cid,
        source=SourceType.CONFLUENCE,
        title=cid,
        text=text,
        ordinal=0,
        last_modified=datetime(2026, 1, 1, tzinfo=UTC),
        acl=ACL.of(groups=["everyone"]),
    )
    return ScoredChunk(chunk=chunk, score=score)


def test_reranker_reorders_by_query_relevance():
    """A cross-encoder-shaped reranker should promote the on-topic candidate."""
    candidates = [
        _sc("off", "the cafeteria menu changes weekly", score=0.9),
        _sc("on", "to reset your vpn password open the client and sign in", score=0.1),
    ]
    reranked = LexicalReranker().rerank("how do I reset my vpn password", candidates)
    # Despite a lower initial score, the on-topic chunk now ranks first.
    assert reranked[0].chunk.chunk_id == "on"


def test_reranker_is_deterministic():
    r = LexicalReranker()
    cands = [_sc("a", "vpn client setup"), _sc("b", "expense policy")]
    assert [c.chunk.chunk_id for c in r.rerank("vpn", cands)] == [
        c.chunk.chunk_id for c in r.rerank("vpn", cands)
    ]


def test_reranker_handles_empty():
    assert LexicalReranker().rerank("anything", []) == []


def test_pipeline_only_reranks_shortlist_and_logs_latency(retriever):
    """Rerank runs on top-N and returns top-k; per-stage latency is recorded."""
    from eaip.retrieval import Principal

    result = retriever.retrieve("vpn setup client", Principal.of("u", ["everyone"]))
    # top_k cap honored.
    assert len(result.chunks) <= 5
    # The fused shortlist (what the reranker saw) is at least as large as top-k.
    assert len(result.fused) >= len(result.chunks)
    # Latency for every stage, including the reranking cost, is surfaced.
    assert {"dense_ms", "sparse_ms", "fusion_ms", "rerank_ms"} <= set(result.timings_ms)
    assert all(v >= 0.0 for v in result.timings_ms.values())
