"""Qdrant-backed chunk index.

This wraps Qdrant (in local/embedded mode — no Docker) behind a small interface
expressed in our domain types. It is responsible for three things that the
ingestion edge cases hinge on:

1. **Idempotent upsert.** A chunk's point id is a stable UUID derived from its
   deterministic ``chunk_id``, so re-indexing the same chunk overwrites the same
   point rather than creating a duplicate. This is what makes "re-index updates
   instead of duplicating" true at the storage layer.

2. **ACL travels into the payload.** Each point stores ``allowed_groups`` and
   ``allowed_users`` alongside the text and metadata, so Phase 2 can filter by
   the requesting user at query time. Every chunk carries its document's ACL.

3. **Delete-by-document.** Deleting a source document must remove *all* its
   chunks. We store ``doc_id`` on every point and delete by a ``doc_id`` filter,
   so a tombstone cleanly removes every derived chunk (no orphans left to leak).

Why a stable UUID instead of using the string id directly: Qdrant point ids must
be an unsigned int or a UUID. We keep the human-readable ``chunk_id`` in the
payload and use ``uuid5(NAMESPACE, chunk_id)`` as the point id — deterministic, so
the mapping is stable across runs.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from eaip.ingestion.models import ACL, Chunk, SourceType

# Fixed namespace so chunk_id -> point id is stable across processes.
_POINT_NAMESPACE = uuid.UUID("00000000-0000-0000-0000-00000000ea1b")


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk returned from search, with its similarity score."""

    chunk: Chunk
    score: float


class ChunkIndex:
    """A thin, domain-typed wrapper over a Qdrant collection of chunks."""

    def __init__(self, client: QdrantClient, collection: str, dim: int) -> None:
        self._client = client
        self._collection = collection
        self._dim = dim
        self._ensure_collection()

    @classmethod
    def open(cls, *, path: str, collection: str, dim: int) -> ChunkIndex:
        """Open (or create) an on-disk embedded index.

        Pass ``path=":memory:"`` for an ephemeral in-process index (tests).
        """
        if path == ":memory:":
            client = QdrantClient(location=":memory:")
        else:
            client = QdrantClient(path=path)
        return cls(client, collection, dim)

    def _ensure_collection(self) -> None:
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qm.VectorParams(size=self._dim, distance=qm.Distance.COSINE),
            )

    # --- writes ---
    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Insert or overwrite chunks (idempotent on ``chunk_id``)."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")
        points = [
            qm.PointStruct(
                id=_point_id(chunk.chunk_id),
                vector=vector,
                payload=_payload(chunk),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        if points:
            self._client.upsert(collection_name=self._collection, points=points)

    def delete_document(self, doc_id: str) -> None:
        """Remove every chunk derived from ``doc_id`` (tombstone)."""
        self._client.delete(
            collection_name=self._collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))]
                )
            ),
        )

    # --- reads ---
    def count(self) -> int:
        """Total number of indexed chunks."""
        return self._client.count(self._collection, exact=True).count

    def count_for_document(self, doc_id: str) -> int:
        """Number of chunks currently indexed for a document."""
        result = self._client.count(
            self._collection,
            exact=True,
            count_filter=qm.Filter(
                must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))]
            ),
        )
        return result.count

    def search(
        self,
        query_vector: list[float],
        *,
        limit: int = 10,
        acl_filter: qm.Filter | None = None,
    ) -> list[ScoredChunk]:
        """Dense nearest-neighbour search, optionally constrained by an ACL filter."""
        response = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=limit,
            query_filter=acl_filter,
            with_payload=True,
        )
        return [
            ScoredChunk(chunk=_chunk_from_payload(p.payload), score=p.score)
            for p in response.points
        ]


def _payload(chunk: Chunk) -> dict:
    """Serialize a chunk to a Qdrant payload (ACL lists are stored explicitly)."""
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "source": str(chunk.source),
        "title": chunk.title,
        "text": chunk.text,
        "ordinal": chunk.ordinal,
        "last_modified": chunk.last_modified.isoformat(),
        "allowed_groups": sorted(chunk.acl.allowed_groups),
        "allowed_users": sorted(chunk.acl.allowed_users),
        "extra": dict(chunk.extra),
    }


def _chunk_from_payload(payload: dict) -> Chunk:
    """Rebuild a :class:`Chunk` from a Qdrant payload."""
    from datetime import datetime

    return Chunk(
        chunk_id=payload["chunk_id"],
        doc_id=payload["doc_id"],
        source=SourceType(payload["source"]),
        title=payload["title"],
        text=payload["text"],
        ordinal=payload["ordinal"],
        last_modified=datetime.fromisoformat(payload["last_modified"]),
        acl=ACL.of(
            groups=list(payload.get("allowed_groups", [])),
            users=list(payload.get("allowed_users", [])),
        ),
        extra=dict(payload.get("extra", {})),
    )
