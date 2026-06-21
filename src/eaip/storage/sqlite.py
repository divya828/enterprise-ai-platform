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

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from eaip.ingestion.models import SourceType
from eaip.storage.base import (
    AgentDefinition,
    AgentStore,
    AuditEvent,
    AuditStore,
    Episode,
    EpisodicStore,
    LifecycleState,
    ProceduralStore,
    PromptStore,
    PromptVersion,
    StateStore,
    SyncState,
    UsageStore,
    UsageTotals,
)

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
-- Episodic memory: one row per completed agent run.
CREATE TABLE IF NOT EXISTS episode (
    thread_id   TEXT PRIMARY KEY,
    user        TEXT NOT NULL,
    query       TEXT NOT NULL,
    route       TEXT NOT NULL,
    outcome     TEXT NOT NULL,
    created_at  TEXT NOT NULL          -- ISO 8601
);
-- Procedural memory: learned rules/policies as namespaced key/value text.
CREATE TABLE IF NOT EXISTS procedural_rule (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);
-- Phase 4: append-only audit log (no update/delete API).
CREATE TABLE IF NOT EXISTS audit_event (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant      TEXT NOT NULL,
    actor       TEXT NOT NULL,
    action      TEXT NOT NULL,
    target      TEXT NOT NULL,
    detail      TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
-- Phase 4: versioned prompt registry.
CREATE TABLE IF NOT EXISTS prompt_version (
    tenant      TEXT NOT NULL,
    name        TEXT NOT NULL,
    version     INTEGER NOT NULL,
    text        TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (tenant, name, version)
);
CREATE TABLE IF NOT EXISTS prompt_active (
    tenant   TEXT NOT NULL,
    name     TEXT NOT NULL,
    version  INTEGER NOT NULL,       -- the pinned/active version
    PRIMARY KEY (tenant, name)
);
-- Phase 4: agent definitions + lifecycle state.
CREATE TABLE IF NOT EXISTS agent_definition (
    tenant       TEXT NOT NULL,
    agent_id     TEXT NOT NULL,
    name         TEXT NOT NULL,
    prompt_name  TEXT NOT NULL,
    tools        TEXT NOT NULL,      -- JSON array of tool names
    state        TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    PRIMARY KEY (tenant, agent_id)
);
-- Phase 4: per-tenant daily usage (cost attribution + budgets).
CREATE TABLE IF NOT EXISTS usage_daily (
    tenant    TEXT NOT NULL,
    day       TEXT NOT NULL,
    requests  INTEGER NOT NULL DEFAULT 0,
    tokens    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (tenant, day)
);
"""


class SqliteStateStore(
    StateStore,
    EpisodicStore,
    ProceduralStore,
    AuditStore,
    PromptStore,
    AgentStore,
    UsageStore,
):
    """A durable store backed by a SQLite database file.

    Implements every platform store interface over one connection — the "one
    store, many capabilities" shape the storage abstraction was built for:
    ingestion state, the Phase 3 memory tiers, and the Phase 4 governance stores
    (audit log, prompt registry, agent definitions, usage counters).
    """

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

    # --- EpisodicStore interface ---
    def record_episode(self, episode: Episode) -> None:
        with self._conn:
            # Upsert on thread_id so re-recording a resumed run replaces, not dupes.
            self._conn.execute(
                "INSERT INTO episode (thread_id, user, query, route, outcome, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(thread_id) DO UPDATE SET "
                "user=excluded.user, query=excluded.query, route=excluded.route, "
                "outcome=excluded.outcome, created_at=excluded.created_at",
                (
                    episode.thread_id,
                    episode.user,
                    episode.query,
                    episode.route,
                    episode.outcome,
                    episode.created_at,
                ),
            )

    def recent_episodes(self, *, user: str | None = None, limit: int = 5) -> list[Episode]:
        if user is None:
            rows = self._conn.execute(
                "SELECT * FROM episode ORDER BY created_at DESC LIMIT ?", (limit,)
            )
        else:
            rows = self._conn.execute(
                "SELECT * FROM episode WHERE user = ? ORDER BY created_at DESC LIMIT ?",
                (user, limit),
            )
        return [
            Episode(
                thread_id=r["thread_id"],
                user=r["user"],
                query=r["query"],
                route=r["route"],
                outcome=r["outcome"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    # --- ProceduralStore interface ---
    def get_rule(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM procedural_rule WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def set_rule(self, key: str, value: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO procedural_rule (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def all_rules(self) -> dict[str, str]:
        return {
            r["key"]: r["value"]
            for r in self._conn.execute("SELECT key, value FROM procedural_rule")
        }

    # --- AuditStore (append-only) ---
    def append_event(self, event: AuditEvent) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO audit_event (tenant, actor, action, target, detail, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    event.tenant,
                    event.actor,
                    event.action,
                    event.target,
                    event.detail,
                    event.created_at,
                ),
            )

    def events(self, *, tenant: str, limit: int = 100) -> list[AuditEvent]:
        rows = self._conn.execute(
            "SELECT * FROM audit_event WHERE tenant = ? ORDER BY id DESC LIMIT ?",
            (tenant, limit),
        )
        return [
            AuditEvent(
                tenant=r["tenant"],
                actor=r["actor"],
                action=r["action"],
                target=r["target"],
                detail=r["detail"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    # --- PromptStore (versioned registry) ---
    def add_version(self, tenant: str, name: str, text: str, created_at: str) -> PromptVersion:
        with self._conn:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(version), 0) AS v FROM prompt_version "
                "WHERE tenant = ? AND name = ?",
                (tenant, name),
            ).fetchone()
            version = row["v"] + 1
            self._conn.execute(
                "INSERT INTO prompt_version (tenant, name, version, text, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (tenant, name, version, text, created_at),
            )
            # The newest version becomes active by default.
            self._conn.execute(
                "INSERT INTO prompt_active (tenant, name, version) VALUES (?, ?, ?) "
                "ON CONFLICT(tenant, name) DO UPDATE SET version=excluded.version",
                (tenant, name, version),
            )
        return PromptVersion(tenant, name, version, text, created_at)

    def get_active(self, tenant: str, name: str) -> PromptVersion | None:
        row = self._conn.execute(
            "SELECT pv.* FROM prompt_active pa "
            "JOIN prompt_version pv ON pv.tenant = pa.tenant AND pv.name = pa.name "
            "AND pv.version = pa.version "
            "WHERE pa.tenant = ? AND pa.name = ?",
            (tenant, name),
        ).fetchone()
        return _prompt_from_row(row) if row else None

    def history(self, tenant: str, name: str) -> list[PromptVersion]:
        rows = self._conn.execute(
            "SELECT * FROM prompt_version WHERE tenant = ? AND name = ? ORDER BY version DESC",
            (tenant, name),
        )
        return [_prompt_from_row(r) for r in rows]

    def pin(self, tenant: str, name: str, version: int) -> PromptVersion:
        row = self._conn.execute(
            "SELECT * FROM prompt_version WHERE tenant = ? AND name = ? AND version = ?",
            (tenant, name, version),
        ).fetchone()
        if row is None:
            raise KeyError(f"no version {version} of prompt {name!r} for tenant {tenant!r}")
        with self._conn:
            self._conn.execute(
                "INSERT INTO prompt_active (tenant, name, version) VALUES (?, ?, ?) "
                "ON CONFLICT(tenant, name) DO UPDATE SET version=excluded.version",
                (tenant, name, version),
            )
        return _prompt_from_row(row)

    # --- AgentStore (definitions + lifecycle) ---
    def upsert_agent(self, agent: AgentDefinition) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO agent_definition "
                "(tenant, agent_id, name, prompt_name, tools, state, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(tenant, agent_id) DO UPDATE SET "
                "name=excluded.name, prompt_name=excluded.prompt_name, "
                "tools=excluded.tools, state=excluded.state, updated_at=excluded.updated_at",
                (
                    agent.tenant,
                    agent.agent_id,
                    agent.name,
                    agent.prompt_name,
                    json.dumps(agent.tools),
                    agent.state.value,
                    agent.updated_at,
                ),
            )

    def get_agent(self, tenant: str, agent_id: str) -> AgentDefinition | None:
        row = self._conn.execute(
            "SELECT * FROM agent_definition WHERE tenant = ? AND agent_id = ?",
            (tenant, agent_id),
        ).fetchone()
        return _agent_from_row(row) if row else None

    def list_agents(self, tenant: str) -> list[AgentDefinition]:
        rows = self._conn.execute(
            "SELECT * FROM agent_definition WHERE tenant = ? ORDER BY agent_id", (tenant,)
        )
        return [_agent_from_row(r) for r in rows]

    # --- UsageStore (per-tenant daily counters) ---
    def record_usage(self, tenant: str, *, tokens: int, day: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO usage_daily (tenant, day, requests, tokens) VALUES (?, ?, 1, ?) "
                "ON CONFLICT(tenant, day) DO UPDATE SET "
                "requests = requests + 1, tokens = tokens + excluded.tokens",
                (tenant, day, tokens),
            )

    def usage_for_day(self, tenant: str, day: str) -> UsageTotals:
        row = self._conn.execute(
            "SELECT requests, tokens FROM usage_daily WHERE tenant = ? AND day = ?",
            (tenant, day),
        ).fetchone()
        if row is None:
            return UsageTotals(tenant=tenant, requests=0, tokens=0)
        return UsageTotals(tenant=tenant, requests=row["requests"], tokens=row["tokens"])


def _prompt_from_row(row: sqlite3.Row) -> PromptVersion:
    return PromptVersion(
        tenant=row["tenant"],
        name=row["name"],
        version=row["version"],
        text=row["text"],
        created_at=row["created_at"],
    )


def _agent_from_row(row: sqlite3.Row) -> AgentDefinition:
    return AgentDefinition(
        tenant=row["tenant"],
        agent_id=row["agent_id"],
        name=row["name"],
        prompt_name=row["prompt_name"],
        tools=json.loads(row["tools"]),
        state=LifecycleState(row["state"]),
        updated_at=row["updated_at"],
    )
