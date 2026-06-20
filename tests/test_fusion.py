"""Unit tests for Reciprocal Rank Fusion — proving the formula behaves."""

from __future__ import annotations

from datetime import UTC, datetime

from eaip.index.store import ScoredChunk
from eaip.ingestion.models import ACL, Chunk, SourceType
from eaip.retrieval.fusion import reciprocal_rank_fusion


def _chunk(cid: str) -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id=cid.split("::")[0],
        source=SourceType.CONFLUENCE,
        title=cid,
        text=cid,
        ordinal=0,
        last_modified=datetime(2026, 1, 1, tzinfo=UTC),
        acl=ACL.of(groups=["everyone"]),
    )


def _sc(cid: str, score: float) -> ScoredChunk:
    return ScoredChunk(chunk=_chunk(cid), score=score)


def test_rrf_rewards_agreement_between_lists():
    """A chunk ranked highly by BOTH arms should beat one ranked highly by only one."""
    dense = [_sc("A::0", 0.9), _sc("B::0", 0.8), _sc("C::0", 0.7)]
    sparse = [_sc("B::0", 5.0), _sc("A::0", 4.0), _sc("D::0", 3.0)]
    fused = reciprocal_rank_fusion({"dense": dense, "bm25": sparse})

    by_id = {f.chunk_id: f for f in fused}
    # A (ranks 1 & 2) and B (ranks 2 & 1) appear in both -> top two.
    assert {fused[0].chunk_id, fused[1].chunk_id} == {"A::0", "B::0"}
    # C and D appear in only one list -> below.
    assert by_id["C::0"].fused_score < by_id["A::0"].fused_score
    assert by_id["D::0"].fused_score < by_id["B::0"].fused_score


def test_rrf_uses_rank_not_raw_score():
    """BM25's huge magnitudes must not dominate cosine's small ones — rank only."""
    dense = [_sc("X::0", 0.99)]  # tiny score, rank 1
    sparse = [_sc("Y::0", 1000.0)]  # huge score, rank 1
    fused = reciprocal_rank_fusion({"dense": dense, "bm25": sparse})
    # Both are rank 1 in their list -> equal fused score despite wild score gap.
    assert fused[0].fused_score == fused[1].fused_score


def test_rrf_formula_value():
    """Explicit check of the 1/(k+rank) contribution."""
    dense = [_sc("A::0", 1.0), _sc("B::0", 0.5)]
    fused = reciprocal_rank_fusion({"dense": dense}, k=60)
    a = next(f for f in fused if f.chunk_id == "A::0")
    assert abs(a.fused_score - 1.0 / (60 + 1)) < 1e-9


def test_rrf_records_per_source_ranks():
    dense = [_sc("A::0", 0.9)]
    sparse = [_sc("B::0", 1.0), _sc("A::0", 0.5)]
    fused = {f.chunk_id: f for f in reciprocal_rank_fusion({"dense": dense, "bm25": sparse})}
    assert fused["A::0"].ranks == {"dense": 1, "bm25": 2}
    assert fused["B::0"].ranks == {"bm25": 1}


def test_rrf_empty_input():
    assert reciprocal_rank_fusion({"dense": [], "bm25": []}) == []
