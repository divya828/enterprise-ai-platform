"""The storage-layer abstraction.

A platform needs durable state: ingestion watermarks now; audit logs, the prompt
registry, and tenant config in later phases. Rather than let each feature reach
for ``sqlite3`` directly, we define a thin *storage seam* — interfaces here, one
concrete backend (SQLite) alongside. Every feature depends on an interface, so
swapping the backend (SQLite → Postgres) means writing one new implementation
class, not editing call sites. This mirrors the provider and embedder
abstractions and is exactly the "abstract the storage layer so Postgres is a
drop-in swap" requirement.

This module defines :class:`StateStore`, the interface for ingestion sync state.
Later phases add sibling interfaces (e.g. ``AuditStore``, ``PromptStore``) in
this package, and the SQLite backend implements all of them over one connection.

``SyncState`` is the in-memory transfer object passed across the seam: the
pipeline works with it; the store loads and saves it. Keeping it free of any
persistence logic is what lets the same pipeline run against an in-memory store
(tests) or SQLite (the app) with no code change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

from eaip.ingestion.models import SourceType


@dataclass
class SyncState:
    """Per-source ingestion bookkeeping: a watermark and the set of indexed ids.

    Pure data — no I/O. A :class:`StateStore` is responsible for persisting it.
    """

    watermarks: dict[SourceType, datetime] = field(default_factory=dict)
    indexed_ids: dict[SourceType, set[str]] = field(default_factory=dict)


@runtime_checkable
class StateStore(Protocol):
    """Durable store for ingestion sync state.

    Implementations persist a :class:`SyncState` and return it on load. The
    pipeline depends only on this protocol, never on a concrete database.
    """

    def load_state(self) -> SyncState:
        """Return the persisted sync state (empty if nothing stored yet)."""
        ...

    def save_state(self, state: SyncState) -> None:
        """Persist the full sync state, replacing any prior contents."""
        ...
