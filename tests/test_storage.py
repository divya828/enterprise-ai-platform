"""Tests for the storage layer (StateStore + SQLite/in-memory backends)."""

from __future__ import annotations

from datetime import UTC, datetime

from eaip.ingestion import SourceType
from eaip.storage import InMemoryStateStore, SqliteStateStore, StateStore, SyncState


def _sample_state() -> SyncState:
    return SyncState(
        watermarks={
            SourceType.CONFLUENCE: datetime(2026, 1, 10, 12, 0, tzinfo=UTC),
            SourceType.JIRA: datetime(2026, 1, 5, 12, 0, tzinfo=UTC),
        },
        indexed_ids={
            SourceType.CONFLUENCE: {"CONF-1", "CONF-2"},
            SourceType.JIRA: {"JIRA-1"},
        },
    )


def test_both_backends_satisfy_the_protocol():
    assert isinstance(InMemoryStateStore(), StateStore)


def test_empty_store_returns_empty_state(tmp_path):
    store = SqliteStateStore(tmp_path / "eaip.db")
    state = store.load_state()
    assert state.watermarks == {}
    assert state.indexed_ids == {}
    store.close()


def test_sqlite_roundtrip_preserves_state(tmp_path):
    store = SqliteStateStore(tmp_path / "eaip.db")
    store.save_state(_sample_state())
    loaded = store.load_state()
    assert loaded.watermarks[SourceType.CONFLUENCE] == datetime(2026, 1, 10, 12, 0, tzinfo=UTC)
    assert loaded.indexed_ids[SourceType.CONFLUENCE] == {"CONF-1", "CONF-2"}
    assert loaded.indexed_ids[SourceType.JIRA] == {"JIRA-1"}
    store.close()


def test_sqlite_persists_across_connections(tmp_path):
    db = tmp_path / "eaip.db"
    s1 = SqliteStateStore(db)
    s1.save_state(_sample_state())
    s1.close()

    # Reopen (simulating a new process) — state is still there.
    s2 = SqliteStateStore(db)
    loaded = s2.load_state()
    assert loaded.indexed_ids[SourceType.JIRA] == {"JIRA-1"}
    s2.close()


def test_save_replaces_prior_contents(tmp_path):
    store = SqliteStateStore(tmp_path / "eaip.db")
    store.save_state(_sample_state())

    # A new, smaller state must fully replace the old one (no leftover rows).
    store.save_state(
        SyncState(
            watermarks={SourceType.JIRA: datetime(2026, 2, 1, 12, 0, tzinfo=UTC)},
            indexed_ids={SourceType.JIRA: {"JIRA-9"}},
        )
    )
    loaded = store.load_state()
    assert set(loaded.watermarks) == {SourceType.JIRA}
    assert loaded.indexed_ids[SourceType.JIRA] == {"JIRA-9"}
    assert SourceType.CONFLUENCE not in loaded.indexed_ids
    store.close()


def test_sqlite_uses_normalized_tables(tmp_path):
    """The schema is real relational rows, not a single serialized blob."""
    import sqlite3

    db = tmp_path / "eaip.db"
    store = SqliteStateStore(db)
    store.save_state(_sample_state())
    store.close()

    conn = sqlite3.connect(db)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"sync_watermark", "indexed_doc"} <= tables
    # Queryable per-source, as a relational store should be.
    rows = conn.execute("SELECT doc_id FROM indexed_doc WHERE source='confluence'").fetchall()
    assert {r[0] for r in rows} == {"CONF-1", "CONF-2"}
    conn.close()


def test_in_memory_store_roundtrip():
    store = InMemoryStateStore()
    store.save_state(_sample_state())
    assert store.load_state().indexed_ids[SourceType.JIRA] == {"JIRA-1"}
