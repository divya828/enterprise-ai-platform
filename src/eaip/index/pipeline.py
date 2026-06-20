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

The watermark/seen-id state lives in a :class:`~eaip.storage.SyncState` and is
persisted through a :class:`~eaip.storage.StateStore` — the storage abstraction.
Tests use the in-memory store; the app uses SQLite, so syncs are incremental
across process restarts. Swapping to Postgres later means a new store class, not
a pipeline change.
"""

from __future__ import annotations

from dataclasses import dataclass

from eaip.embeddings.base import Embedder
from eaip.index.store import ChunkIndex
from eaip.ingestion.chunker import ChunkConfig, chunk_document
from eaip.ingestion.connectors import Connector
from eaip.ingestion.models import SourceType
from eaip.storage import InMemoryStateStore, StateStore, SyncState


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
    """Drives connectors through chunking, embedding, and indexing.

    Persistence is delegated to a :class:`StateStore`. The pipeline loads the
    current :class:`SyncState` at the start of each sync and saves it at the end,
    so a single ``sync`` call is atomic with respect to the stored watermark.
    """

    def __init__(
        self,
        index: ChunkIndex,
        embedder: Embedder,
        *,
        chunk_config: ChunkConfig | None = None,
        store: StateStore | None = None,
    ) -> None:
        self._index = index
        self._embedder = embedder
        self._chunk_config = chunk_config or ChunkConfig()
        self._store = store or InMemoryStateStore()

    @property
    def store(self) -> StateStore:
        return self._store

    def state(self) -> SyncState:
        """Return the currently persisted sync state (for inspection/tests)."""
        return self._store.load_state()

    def sync(self, connector: Connector, *, full: bool = False) -> SyncReport:
        """Run one sync for a connector.

        ``full=True`` ignores the watermark and re-processes everything (a forced
        rebuild). Otherwise only documents newer than the source's watermark are
        processed. Deletions are always reconciled. The updated state is persisted
        through the store before returning.
        """
        source = connector.source
        state = self._store.load_state()
        watermark = None if full else state.watermarks.get(source)

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
        seen = state.indexed_ids.get(source, set())
        current = connector.current_ids()
        deleted = seen - current
        for doc_id in deleted:
            self._index.delete_document(doc_id)

        # 3. Update + persist sync state.
        if newest is not None:
            state.watermarks[source] = newest
        state.indexed_ids[source] = set(current)
        self._store.save_state(state)

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
