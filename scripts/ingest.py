"""Ingest the synthetic corpus into the embedded Qdrant index.

Run with: ``uv run python scripts/ingest.py``

This is the runnable Phase 1 demo: it loads the corpus through the mock
connectors, chunks + embeds + indexes everything, and prints a per-source sync
report plus the total chunk count. Re-running it is a no-op for unchanged docs
(watermark) and never duplicates chunks (deterministic ids) — run it twice to see
that the second run upserts zero documents.

Uses whatever provider/embedder the environment selects; defaults (hashing
embedder, on-disk Qdrant at ./data/qdrant) need no model and no API key.
"""

from __future__ import annotations

from eaip.config import get_settings
from eaip.embeddings import get_embedder
from eaip.index import ChunkIndex, IngestionPipeline, SyncState
from eaip.ingestion import connectors_from_corpus


def main() -> None:
    settings = get_settings()
    corpus_path = settings.data_dir / "corpus" / "documents.json"
    state_path = settings.qdrant_path / "sync_state.json"

    embedder = get_embedder(settings)
    index = ChunkIndex.open(
        path=str(settings.qdrant_path),
        collection=settings.qdrant_collection,
        dim=embedder.dim,
    )
    # Load persisted sync state so the watermark survives across runs — a second
    # run is then a genuine no-op rather than a full re-sync.
    state = SyncState.load(state_path)
    pipeline = IngestionPipeline(index, embedder, state=state)
    connectors = connectors_from_corpus(corpus_path)

    print(f"Embedder: {embedder.name} (dim={embedder.dim})")
    print(f"Index:    {settings.qdrant_path} / {settings.qdrant_collection}\n")
    for report in pipeline.sync_all(connectors):
        print(report)
    pipeline.state.save(state_path)
    print(f"\nTotal indexed chunks: {index.count()}")


if __name__ == "__main__":
    main()
