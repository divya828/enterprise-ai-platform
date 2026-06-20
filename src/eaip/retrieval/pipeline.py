"""Hybrid retrieval pipeline: dense + sparse -> RRF -> cross-encoder rerank.

This is the RAG core. For a query and a requesting *principal* (a user plus their
groups):

1. **Permission filter** is computed once from the principal and applied to BOTH
   retrieval arms, so forbidden chunks never enter either candidate set.
2. **Dense** (vector) and **sparse** (BM25) each return a top-N shortlist.
3. **RRF fusion** combines the two rankings into one (rank-based, scale-free).
4. **Cross-encoder rerank** re-scores ONLY the fused shortlist (top-N) and keeps
   the top-k — the bi-encoder→cross-encoder shortlist pattern. Reranking latency
   is measured and returned so the cost of step 4 is visible.

The output (a :class:`RetrievalResult`) carries the reranked top-k plus timings
and the fused shortlist, so callers can both ground an answer and inspect *how*
retrieval behaved. Whether the result is confident enough to answer is decided by
the answerer, not here — this module's job is to return the best permitted
evidence and the numbers describing it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from eaip.index.acl_filter import access_filter
from eaip.index.store import ScoredChunk
from eaip.retrieval.dense import DenseRetriever
from eaip.retrieval.fusion import FusedChunk, reciprocal_rank_fusion
from eaip.retrieval.reranker import Reranker
from eaip.retrieval.sparse import SparseRetriever


@dataclass(frozen=True)
class Principal:
    """The identity a retrieval request is made on behalf of.

    Retrieval is always scoped to a principal — there is no "search as everyone".
    The ``groups`` drive the ACL filter alongside the ``user`` id.
    """

    user: str
    groups: frozenset[str] = frozenset()

    @classmethod
    def of(cls, user: str, groups: list[str] | None = None) -> Principal:
        return cls(user=user, groups=frozenset(groups or ()))


@dataclass(frozen=True)
class RetrievalResult:
    """Everything a retrieval produced: the reranked top-k plus diagnostics."""

    query: str
    principal: Principal
    chunks: list[ScoredChunk]  # final reranked top-k
    fused: list[FusedChunk] = field(default_factory=list)  # pre-rerank shortlist
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def top_score(self) -> float:
        """The best reranked score (0.0 if nothing was retrieved)."""
        return self.chunks[0].score if self.chunks else 0.0


class HybridRetriever:
    """Dense + sparse retrieval, RRF fusion, and cross-encoder reranking."""

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        reranker: Reranker,
        *,
        shortlist: int = 20,
        top_k: int = 5,
    ) -> None:
        self._dense = dense
        self._sparse = sparse
        self._reranker = reranker
        self._shortlist = shortlist
        self._top_k = top_k

    def retrieve(self, query: str, principal: Principal) -> RetrievalResult:
        """Run the full hybrid pipeline for ``query`` as ``principal``."""
        timings: dict[str, float] = {}
        # One ACL filter, applied to both arms — the single permission gate.
        acl = access_filter(user=principal.user, groups=principal.groups)

        with _timed(timings, "dense_ms"):
            dense_hits = self._dense.search(query, limit=self._shortlist, acl_filter=acl)
        with _timed(timings, "sparse_ms"):
            sparse_hits = self._sparse.search(query, limit=self._shortlist, acl_filter=acl)

        with _timed(timings, "fusion_ms"):
            fused = reciprocal_rank_fusion({"dense": dense_hits, "bm25": sparse_hits})

        # Rerank ONLY the fused shortlist (top-N), then keep top-k. This is where
        # the cross-encoder cost lives, so we measure it explicitly.
        shortlist = [f.scored for f in fused[: self._shortlist]]
        with _timed(timings, "rerank_ms"):
            reranked = self._reranker.rerank(query, shortlist)
        top_k = reranked[: self._top_k]

        return RetrievalResult(
            query=query,
            principal=principal,
            chunks=top_k,
            fused=fused,
            timings_ms=timings,
        )


class _timed:
    """Context manager that records elapsed wall-clock ms into ``store[key]``."""

    def __init__(self, store: dict[str, float], key: str) -> None:
        self._store = store
        self._key = key

    def __enter__(self) -> _timed:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self._store[self._key] = (time.perf_counter() - self._start) * 1000.0
