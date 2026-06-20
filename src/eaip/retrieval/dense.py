"""Dense (vector) retrieval — semantic nearest-neighbour search.

A thin wrapper that embeds the query and runs an ACL-filtered nearest-neighbour
search against the Qdrant index. The ACL filter is applied *inside* the vector
search (Qdrant constrains candidates before ranking), so forbidden chunks are
never even scored — the same permission posture as the sparse arm.
"""

from __future__ import annotations

from qdrant_client import models as qm

from eaip.embeddings.base import Embedder
from eaip.index.store import ChunkIndex, ScoredChunk


class DenseRetriever:
    """Embeds the query and runs ACL-filtered vector search."""

    name = "dense"

    def __init__(self, index: ChunkIndex, embedder: Embedder) -> None:
        self._index = index
        self._embedder = embedder

    def search(
        self, query: str, *, limit: int, acl_filter: qm.Filter | None = None
    ) -> list[ScoredChunk]:
        query_vector = self._embedder.embed_query(query)
        return self._index.search(query_vector, limit=limit, acl_filter=acl_filter)
