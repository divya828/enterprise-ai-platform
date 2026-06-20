"""Reranker abstraction: bi-encoder shortlist -> cross-encoder rerank.

The "shortlist then rerank" pattern is the central efficiency/quality trade in
modern RAG. Dense + sparse retrieval are *bi-encoder* style: the query and each
document are embedded independently, so retrieval is a fast nearest-neighbour
lookup but the model never sees the query and document *together*. A
*cross-encoder* does: it takes the (query, document) pair as one input and scores
their relevance directly, which is far more accurate — but it must run the model
once per candidate, so it's too slow to run over the whole corpus. The resolution
is to retrieve a cheap top-N shortlist with the bi-encoders, then rerank only
that shortlist with the cross-encoder and keep the top-k. We log the reranking
latency so the cost side of that trade is visible.

As with embeddings and the LLM provider, the reranker is an interface with an
offline deterministic default (lexical overlap) and a real opt-in backend (a
sentence-transformers cross-encoder), so the shortlist→rerank machinery and its
latency accounting are testable in CI with no model download.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from eaip.index.store import ScoredChunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Reranker(Protocol):
    """Re-scores (query, chunk) pairs and returns them in descending relevance."""

    name: str

    def rerank(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        """Return ``candidates`` re-scored and sorted most-relevant first."""
        ...


class LexicalReranker:
    """Deterministic, offline reranker — the default for tests and CI.

    Scores each candidate by token-overlap with the query (a Jaccard-style
    measure). This is genuinely cross-encoder-*shaped* — it scores the pair
    together rather than via independent embeddings — so it meaningfully reorders
    a shortlist and lets us test the top-N→top-k flow and latency logging without
    a model. It is not as accurate as a real cross-encoder; ``EAIP_RERANKER=bge``
    switches to one for quality.
    """

    name = "lexical"

    def rerank(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        q_tokens = set(_TOKEN_RE.findall(query.lower()))
        rescored = [
            ScoredChunk(chunk=c.chunk, score=self._overlap(q_tokens, c.chunk.text))
            for c in candidates
        ]
        rescored.sort(key=lambda c: c.score, reverse=True)
        return rescored

    @staticmethod
    def _overlap(q_tokens: set[str], text: str) -> float:
        d_tokens = set(_TOKEN_RE.findall(text.lower()))
        if not q_tokens or not d_tokens:
            return 0.0
        return len(q_tokens & d_tokens) / len(q_tokens | d_tokens)


class CrossEncoderReranker:
    """Real cross-encoder reranker via sentence-transformers (opt-in, lazy import).

    Defaults to BAAI/bge-reranker-base. ``predict`` scores a batch of
    (query, document) pairs; we attach those scores and sort. Downloads the model
    on first use, then runs offline.
    """

    name = "bge"

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        from sentence_transformers import CrossEncoder  # noqa: PLC0415

        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        if not candidates:
            return []
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self._model.predict(pairs)
        rescored = [
            ScoredChunk(chunk=c.chunk, score=float(s))
            for c, s in zip(candidates, scores, strict=True)
        ]
        rescored.sort(key=lambda c: c.score, reverse=True)
        return rescored
