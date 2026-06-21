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
from eaip.index import IngestionPipeline
from eaip.index.resolver import TenantIndexResolver
from eaip.ingestion import connectors_from_corpus
from eaip.storage import SqliteStateStore


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Ingest the corpus for a tenant.")
    parser.add_argument("--tenant", default=None, help="Tenant id (default from settings).")
    args = parser.parse_args()

    settings = get_settings()
    tenant = args.tenant or settings.default_tenant
    corpus_path = settings.data_dir / "corpus" / "documents.json"

    embedder = get_embedder(settings)
    # Resolve this tenant's own collection — data is physically isolated per tenant.
    resolver = TenantIndexResolver(
        path=str(settings.qdrant_path),
        collection_prefix=settings.qdrant_collection,
        dim=embedder.dim,
    )
    index = resolver.for_tenant(tenant)
    # The SQLite state store persists watermarks across runs, so a second run is a
    # genuine no-op rather than a full re-sync. (Same DB file later phases use.)
    store = SqliteStateStore(settings.state_db_path)
    pipeline = IngestionPipeline(index, embedder, store=store)
    connectors = connectors_from_corpus(corpus_path)

    print(f"Tenant:   {tenant}")
    print(f"Embedder: {embedder.name} (dim={embedder.dim})")
    print(f"Index:    {settings.qdrant_path} / {settings.qdrant_collection}__{tenant}")
    print(f"State:    {settings.state_db_path} (SQLite)\n")
    for report in pipeline.sync_all(connectors):
        print(report)
    print(f"\nTotal indexed chunks: {index.count()}")
    store.close()


if __name__ == "__main__":
    main()
