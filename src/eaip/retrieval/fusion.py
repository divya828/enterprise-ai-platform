"""Reciprocal Rank Fusion (RRF) — combining dense and sparse rankings.

Dense and sparse retrieval produce scores on completely different scales (cosine
similarity in [-1, 1] vs. unbounded BM25 magnitudes), so you cannot just add or
average them. RRF sidesteps the scale problem by throwing the *scores* away and
fusing on *rank position* instead. Each result contributes ``1 / (k + rank)`` to
its document's fused score, summed across the ranked lists it appears in. A
document near the top of either list gets a large contribution; a document near
the top of *both* gets the largest — which is exactly the behavior we want from a
hybrid retriever (agreement between methods is a strong relevance signal).

The constant ``k`` (conventionally 60) dampens the influence of the very top
ranks so that, e.g., rank 1 isn't wildly more valuable than rank 2; a larger ``k``
flattens the curve. We implement RRF explicitly (rather than relying on a library)
because seeing the formula is one of the points of this exercise.

Reference: Cormack, Clarke, Buettcher (2009), "Reciprocal Rank Fusion outperforms
Condorcet and individual rank learning methods."
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from eaip.index.store import ScoredChunk

DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class FusedChunk:
    """A chunk after fusion, carrying its fused score and per-source ranks.

    ``ranks`` maps a source name (e.g. "dense", "bm25") to this chunk's 1-based
    rank in that source's list — kept for explainability/debugging so you can see
    *why* a chunk fused high.
    """

    chunk_id: str
    scored: ScoredChunk
    fused_score: float
    ranks: dict[str, int]


def reciprocal_rank_fusion(
    ranked_lists: dict[str, Sequence[ScoredChunk]],
    *,
    k: int = DEFAULT_RRF_K,
) -> list[FusedChunk]:
    """Fuse several ranked lists into one, sorted by fused score (desc).

    ``ranked_lists`` maps a source name to that source's results in rank order
    (best first). A chunk appearing in multiple lists accumulates a contribution
    from each. Chunks are identified across lists by ``chunk_id``.
    """
    fused_scores: dict[str, float] = {}
    ranks_by_chunk: dict[str, dict[str, int]] = {}
    representative: dict[str, ScoredChunk] = {}

    for source, results in ranked_lists.items():
        for rank, scored in enumerate(results, start=1):
            cid = scored.chunk.chunk_id
            fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (k + rank)
            ranks_by_chunk.setdefault(cid, {})[source] = rank
            representative.setdefault(cid, scored)

    fused = [
        FusedChunk(
            chunk_id=cid,
            scored=representative[cid],
            fused_score=score,
            ranks=ranks_by_chunk[cid],
        )
        for cid, score in fused_scores.items()
    ]
    fused.sort(key=lambda f: f.fused_score, reverse=True)
    return fused
