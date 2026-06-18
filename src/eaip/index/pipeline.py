"""The ingestion pipeline: connectors -> chunker -> embedder -> index.

This ties Phase 1 together and is where the three required edge cases live:

* **Incremental re-indexing (watermark).** The pipeline keeps a per-source
  watermark = the newest ``last_modified`` it has indexed. On each sync it asks
  the connector for only documents modified after that watermark, so an
  unchanged corpus does no work and an edited document is re-processed. Because
  chunk ids are deterministic, re-processing *overwrites* the document's chunks
  rather than duplicating them.

* **Deletions / tombstones.** The pipeline remembers which doc ids it has indexed
  per source. On sync it compares that against the connector's *current* ids; any
  id that disappeared is deleted from the index (all of its chunks). A deleted
  source document therefore becomes unretrievable.

* **ACL preservation.** Chunking copies the document ACL onto every chunk and the
  index stores it in each point's payload — verified end to end by the tests.

The watermark/seen-id state lives in a ``SyncState`` object. Tests keep it in
memory; the app persists it to JSON (``SyncState.save``/``load``) so syncs are
incremental across process restarts. In production this JSON file would be a row
in the abstracted storage layer (SQLite/Postgres) — same shape, different store.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from eaip.embeddings.base import Embedder
from eaip.index.store import ChunkIndex
from eaip.ingestion.chunker import ChunkConfig, chunk_document
from eaip.ingestion.connectors import Connector
from eaip.ingestion.models import SourceType


@dataclass
class SyncState:
    """Per-source sync bookkeeping (watermark + indexed doc ids).

    Tests use it purely in memory. The app persists it to a small JSON file via
    :meth:`save`/:meth:`load` so the watermark survives across process restarts —
    which is what makes the second CLI run a true no-op rather than a full
    re-sync. In production this JSON file would be a row in the abstracted storage
    layer (SQLite/Postgres); the shape is identical, only the backing store
    changes. Kept as a separate object so that swap doesn't touch pipeline logic.
    """

    watermarks: dict[SourceType, datetime] = field(default_factory=dict)
    indexed_ids: dict[SourceType, set[str]] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        """Persist watermarks + indexed ids to a JSON file."""
        payload = {
            "watermarks": {s.value: ts.isoformat() for s, ts in self.watermarks.items()},
            "indexed_ids": {s.value: sorted(ids) for s, ids in self.indexed_ids.items()},
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> SyncState:
        """Load state from a JSON file, or return empty state if absent."""
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text())
        return cls(
            watermarks={
                SourceType(s): datetime.fromisoformat(ts)
                for s, ts in data.get("watermarks", {}).items()
            },
            indexed_ids={SourceType(s): set(ids) for s, ids in data.get("indexed_ids", {}).items()},
        )


@dataclass(frozen=True)
class SyncReport:
    """What a single sync did — surfaced so callers (and tests) can assert on it."""

    source: SourceType
    upserted_docs: int
    upserted_chunks: int
    deleted_docs: int

    def __str__(self) -> str:
        return (
            f"[{self.source}] upserted {self.upserted_docs} docs "
            f"({self.upserted_chunks} chunks), deleted {self.deleted_docs} docs"
        )


class IngestionPipeline:
    """Drives connectors through chunking, embedding, and indexing."""

    def __init__(
        self,
        index: ChunkIndex,
        embedder: Embedder,
        *,
        chunk_config: ChunkConfig | None = None,
        state: SyncState | None = None,
    ) -> None:
        self._index = index
        self._embedder = embedder
        self._chunk_config = chunk_config or ChunkConfig()
        self._state = state or SyncState()

    @property
    def state(self) -> SyncState:
        return self._state

    def sync(self, connector: Connector, *, full: bool = False) -> SyncReport:
        """Run one sync for a connector.

        ``full=True`` ignores the watermark and re-processes everything (a forced
        rebuild). Otherwise only documents newer than the source's watermark are
        processed. Deletions are always reconciled.
        """
        source = connector.source
        watermark = None if full else self._state.watermarks.get(source)

        # 1. Upsert new/changed documents.
        changed = connector.fetch_since(watermark)
        upserted_chunks = 0
        newest = watermark
        for doc in changed:
            chunks = chunk_document(doc, self._chunk_config)
            vectors = self._embedder.embed_documents([c.text for c in chunks])
            # Re-chunking may produce fewer chunks than before (e.g. doc shrank);
            # clear the doc's old chunks first so no stale chunk lingers.
            self._index.delete_document(doc.doc_id)
            self._index.upsert_chunks(chunks, vectors)
            upserted_chunks += len(chunks)
            if newest is None or doc.last_modified > newest:
                newest = doc.last_modified

        # 2. Reconcile deletions (tombstones).
        seen = self._state.indexed_ids.setdefault(source, set())
        current = connector.current_ids()
        deleted = seen - current
        for doc_id in deleted:
            self._index.delete_document(doc_id)

        # 3. Update sync state.
        if newest is not None:
            self._state.watermarks[source] = newest
        self._state.indexed_ids[source] = set(current)

        return SyncReport(
            source=source,
            upserted_docs=len(changed),
            upserted_chunks=upserted_chunks,
            deleted_docs=len(deleted),
        )

    def sync_all(
        self, connectors: dict[SourceType, Connector], *, full: bool = False
    ) -> list[SyncReport]:
        """Sync every connector and return one report per source."""
        return [self.sync(c, full=full) for c in connectors.values()]
