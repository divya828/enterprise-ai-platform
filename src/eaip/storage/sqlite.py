"""SQLite-backed StateStore — the durable default for the running app.

Uses Python's stdlib ``sqlite3`` (zero extra dependencies, no server). State is
stored in **normalized tables**, not a serialized blob, so it's the realistic
relational shape an enterprise system would use and is directly queryable
(``SELECT doc_id FROM indexed_doc WHERE source = 'jira'``):

    sync_watermark(source PRIMARY KEY, watermark)   -- one row per source
    indexed_doc(source, doc_id, PRIMARY KEY(source, doc_id))  -- one row per indexed doc

One ``sqlite3.Connection`` is held open for the store's lifetime and shared by
all operations. Later phases (audit log, prompt registry) add their own tables
and methods to this same store/connection rather than opening their own database.

What would change for Postgres (noted for the "drop-in swap" learning): swap the
driver (``psycopg``), use ``%s`` placeholders and ``ON CONFLICT`` upserts (both
already used here in SQLite-compatible form), and manage a connection pool
instead of a single connection. The :class:`StateStore` interface and every call
site stay identical.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from eaip.ingestion.models import SourceType
from eaip.storage.base import StateStore, SyncState

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sync_watermark (
    source     TEXT PRIMARY KEY,
    watermark  TEXT NOT NULL          -- ISO 8601 timestamp
);
CREATE TABLE IF NOT EXISTS indexed_doc (
    source  TEXT NOT NULL,
    doc_id  TEXT NOT NULL,
    PRIMARY KEY (source, doc_id)
);
"""


class SqliteStateStore(StateStore):
    """A durable StateStore backed by a SQLite database file."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False keeps this usable from FastAPI's threadpool in
        # later phases; we serialize our own writes within single calls.
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()

    # --- StateStore interface ---
    def load_state(self) -> SyncState:
        state = SyncState()
        for row in self._conn.execute("SELECT source, watermark FROM sync_watermark"):
            state.watermarks[SourceType(row["source"])] = datetime.fromisoformat(row["watermark"])
        for row in self._conn.execute("SELECT source, doc_id FROM indexed_doc"):
            state.indexed_ids.setdefault(SourceType(row["source"]), set()).add(row["doc_id"])
        return state

    def save_state(self, state: SyncState) -> None:
        """Persist the full state in one transaction (replace prior contents).

        We rewrite the tables wholesale rather than diffing. The state is small
        (one row per source + one row per indexed document) and a full rewrite
        inside a single transaction is simplest and keeps the stored state an
        exact mirror of the in-memory state.
        """
        with self._conn:  # transaction: commit on success, rollback on error
            self._conn.execute("DELETE FROM sync_watermark")
            self._conn.executemany(
                "INSERT INTO sync_watermark (source, watermark) VALUES (?, ?)",
                [(s.value, ts.isoformat()) for s, ts in state.watermarks.items()],
            )
            self._conn.execute("DELETE FROM indexed_doc")
            self._conn.executemany(
                "INSERT INTO indexed_doc (source, doc_id) VALUES (?, ?)",
                [
                    (source.value, doc_id)
                    for source, ids in state.indexed_ids.items()
                    for doc_id in ids
                ],
            )
