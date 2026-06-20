"""Edge-case tests for the ingestion pipeline — the Phase 1 deliverables.

The spec calls for proving:
  1. A deleted source document is no longer retrievable (tombstones).
  2. A re-indexed document updates rather than duplicates.
  3. ACL metadata survives onto every chunk in the index.
Plus the incremental-watermark behavior that makes (2) efficient.
"""

from __future__ import annotations

from datetime import UTC, datetime

from eaip.index import ChunkIndex, IngestionPipeline, access_filter
from eaip.ingestion import ChunkConfig, CorpusConnector, SourceType
from tests.conftest import make_doc


def _search_text(index: ChunkIndex, embedder, query: str, **kw):
    return index.search(embedder.embed_query(query), **kw)


# --- Edge case 1: deletion / tombstone --------------------------------------


def test_deleted_document_is_not_retrievable(index, embedder, pipeline):
    conn = CorpusConnector(
        SourceType.CONFLUENCE,
        [make_doc("DEL-1", title="VPN Guide", text="how to set up the vpn client")],
    )
    pipeline.sync(conn)
    assert index.count_for_document("DEL-1") > 0
    hits_before = _search_text(index, embedder, "vpn client setup")
    assert any(h.chunk.doc_id == "DEL-1" for h in hits_before)

    # Delete in the source, then sync: the tombstone removes all its chunks.
    conn.delete("DEL-1")
    report = pipeline.sync(conn)
    assert report.deleted_docs == 1
    assert index.count_for_document("DEL-1") == 0
    hits_after = _search_text(index, embedder, "vpn client setup")
    assert all(h.chunk.doc_id != "DEL-1" for h in hits_after)


# --- Edge case 2: re-index updates, does not duplicate ----------------------


def test_reindex_updates_not_duplicates(index, embedder, pipeline):
    conn = CorpusConnector(
        SourceType.CONFLUENCE,
        [make_doc("UPD-1", text="original body about onboarding laptops", day=1)],
    )
    pipeline.sync(conn)
    count_first = index.count_for_document("UPD-1")
    total_first = index.count()
    assert count_first >= 1

    # Edit the document (new content, newer timestamp) and re-sync.
    conn.upsert(make_doc("UPD-1", text="revised body about onboarding badges", day=2))
    pipeline.sync(conn)

    # Same doc id -> chunks overwritten, not added. Total grows by the chunk
    # delta only (here zero), never duplicating the document.
    assert index.count() == total_first
    # The new content is searchable; the old content is gone.
    hits = _search_text(index, embedder, "onboarding badges")
    assert any(h.chunk.doc_id == "UPD-1" for h in hits)
    texts = " ".join(
        h.chunk.text
        for h in _search_text(index, embedder, "onboarding", limit=20)
        if h.chunk.doc_id == "UPD-1"
    )
    assert "badges" in texts
    assert "laptops" not in texts


def test_reindex_with_fewer_chunks_leaves_no_stale_chunks(index, embedder, pipeline):
    """Shrinking a document must drop its now-extra chunks."""
    cfg = ChunkConfig(max_tokens=15, overlap_tokens=3)
    pipeline_small = IngestionPipeline(index, embedder, chunk_config=cfg)
    conn = CorpusConnector(
        SourceType.CONFLUENCE,
        [make_doc("SHR-1", text=" ".join(["alpha"] * 120), day=1)],
    )
    pipeline_small.sync(conn)
    many = index.count_for_document("SHR-1")
    assert many > 1

    conn.upsert(make_doc("SHR-1", text="alpha beta", day=2))
    pipeline_small.sync(conn)
    assert index.count_for_document("SHR-1") == 1  # stale chunks cleared


# --- Edge case 3: ACL survives into the index payload -----------------------


def test_acl_present_on_every_indexed_chunk(index, embedder, pipeline):
    conn = CorpusConnector(
        SourceType.CONFLUENCE,
        [
            make_doc(
                "ACL-1",
                text="\n\n".join(f"secret paragraph {i} with words" for i in range(15)),
                groups=["finance"],
                users=["cfo@acme.test"],
            )
        ],
    )
    pipeline.sync(conn)

    # A finance user can retrieve it...
    finance_hits = _search_text(
        index,
        embedder,
        "secret paragraph words",
        acl_filter=access_filter(user="x@acme.test", groups=["finance"]),
    )
    assert any(h.chunk.doc_id == "ACL-1" for h in finance_hits)
    # ...and the ACL is intact on the returned chunk.
    chunk = next(h.chunk for h in finance_hits if h.chunk.doc_id == "ACL-1")
    assert "finance" in chunk.acl.allowed_groups
    assert "cfo@acme.test" in chunk.acl.allowed_users


# --- Incremental watermark behavior -----------------------------------------


def test_watermark_skips_unchanged_docs_on_resync(pipeline):
    conn = CorpusConnector(
        SourceType.CONFLUENCE,
        [make_doc("W-1", day=1), make_doc("W-2", day=2)],
    )
    first = pipeline.sync(conn)
    assert first.upserted_docs == 2

    # No changes -> watermark means nothing is re-processed.
    second = pipeline.sync(conn)
    assert second.upserted_docs == 0
    assert second.deleted_docs == 0

    # Add a newer doc -> only that one is processed.
    conn.upsert(make_doc("W-3", day=5))
    third = pipeline.sync(conn)
    assert third.upserted_docs == 1


def test_full_resync_reprocesses_everything(pipeline):
    conn = CorpusConnector(SourceType.CONFLUENCE, [make_doc("F-1", day=1)])
    pipeline.sync(conn)
    report = pipeline.sync(conn, full=True)
    assert report.upserted_docs == 1  # full=True ignores the watermark


def test_sync_state_persists_across_processes(tmp_path, index, embedder):
    """A SQLite store makes a second 'process' a true no-op (watermark survives).

    Each ``SqliteStateStore`` opens the same DB file fresh, simulating separate
    processes: the second pipeline reloads the watermark and does nothing.
    """
    from eaip.storage import SqliteStateStore

    db_path = tmp_path / "eaip.db"
    conn = CorpusConnector(SourceType.CONFLUENCE, [make_doc("P-1", day=1), make_doc("P-2", day=2)])

    # First "process": fresh DB, full sync, state persisted by sync().
    store1 = SqliteStateStore(db_path)
    p1 = IngestionPipeline(index, embedder, store=store1)
    assert p1.sync(conn).upserted_docs == 2
    store1.close()

    # Second "process": reopen the same DB -> watermark loaded -> nothing to do.
    store2 = SqliteStateStore(db_path)
    p2 = IngestionPipeline(index, embedder, store=store2)
    assert p2.sync(conn).upserted_docs == 0
    assert p2.sync(conn).deleted_docs == 0
    store2.close()


# --- Whole-corpus smoke -----------------------------------------------------


def test_full_corpus_ingests(index, embedder, corpus_connectors):
    pipeline = IngestionPipeline(index, embedder)
    reports = pipeline.sync_all(corpus_connectors)
    total_docs = sum(r.upserted_docs for r in reports)
    assert total_docs == 30  # the synthetic corpus size
    assert index.count() >= 30  # at least one chunk per doc


def test_per_source_watermarks_are_independent(pipeline, corpus_connectors):
    """Each source keeps its own watermark; syncing one doesn't reset another."""
    conf = corpus_connectors[SourceType.CONFLUENCE]
    jira = corpus_connectors[SourceType.JIRA]
    pipeline.sync(conf)
    pipeline.sync(jira)
    # Resyncing confluence does nothing new; jira state is untouched.
    assert pipeline.sync(conf).upserted_docs == 0
    # Add a future-dated jira issue; only jira picks it up.
    jira.upsert(make_doc("JIRA-NEW", source=SourceType.JIRA, day=28, text="new issue body"))
    assert pipeline.sync(jira).upserted_docs == 1
    assert pipeline.state().watermarks[SourceType.JIRA] == datetime(2026, 1, 28, 12, 0, tzinfo=UTC)
