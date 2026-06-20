"""Sparse retrieval (BM25) over ACL-permitted candidates.

Dense retrieval matches on *meaning* (vector similarity); sparse retrieval
matches on *exact terms* (keyword overlap, weighted by term rarity). They fail in
different ways — dense can miss a rare identifier or acronym the query spells out
exactly; sparse can miss a paraphrase that shares no words — so fusing both
(Phase 2's RRF) is more robust than either alone. BM25 is the classic sparse
scorer: it rewards query terms that appear in a document, discounts very common
terms, and normalizes for document length.

**Permission posture (important).** ``rank_bm25`` scores an in-memory corpus; it
has no notion of Qdrant's ACL-filtered nearest-neighbour search. So we build the
BM25 corpus from *only the chunks the requesting principal may read* — pulled
from the index with the same ACL filter the dense arm uses. Permission filtering
therefore happens *before* sparse scoring: a forbidden chunk never enters the
corpus, never influences IDF, and can never be returned. Filtering-after would be
the riskier posture; this is filtering-before, the safe one.

We rebuild the corpus per query here. At this learning scale that's trivial; a
production system would cache a per-tenant index and invalidate on ACL changes.
"""

from __future__ import annotations

import re

from qdrant_client import models as qm
from rank_bm25 import BM25Okapi

from eaip.index.store import ChunkIndex, ScoredChunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class SparseRetriever:
    """BM25 retrieval restricted to ACL-permitted chunks."""

    name = "bm25"

    def __init__(self, index: ChunkIndex) -> None:
        self._index = index

    def search(
        self, query: str, *, limit: int, acl_filter: qm.Filter | None = None
    ) -> list[ScoredChunk]:
        """Return the top ``limit`` chunks by BM25 score, ACL-permitted only."""
        # Pull exactly the chunks this principal may see; build the corpus from them.
        permitted = self._index.scroll_chunks(acl_filter=acl_filter)
        if not permitted:
            return []

        corpus_tokens = [_tokenize(c.text) for c in permitted]
        bm25 = BM25Okapi(corpus_tokens)
        scores = bm25.get_scores(_tokenize(query))

        ranked = sorted(
            (ScoredChunk(chunk=c, score=float(s)) for c, s in zip(permitted, scores, strict=True)),
            key=lambda sc: sc.score,
            reverse=True,
        )
        # Drop zero-score hits (no query term present) — they carry no signal.
        return [sc for sc in ranked[:limit] if sc.score > 0.0]
