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


@dataclass(frozen=True)
class Episode:
    """One completed agent run, recorded for the *episodic* memory tier.

    Episodic memory is "what happened before": past runs the agent can recall to
    inform new ones. We keep a compact record — who asked what, the route taken,
    the outcome — keyed by ``thread_id`` so it lines up with the graph checkpoint.
    """

    thread_id: str
    user: str
    query: str
    route: str
    outcome: str
    created_at: str  # ISO 8601 (passed in; the store does not call the clock)


@runtime_checkable
class EpisodicStore(Protocol):
    """Durable store of past agent runs (episodic memory)."""

    def record_episode(self, episode: Episode) -> None:
        """Persist one completed run."""
        ...

    def recent_episodes(self, *, user: str | None = None, limit: int = 5) -> list[Episode]:
        """Return the most recent episodes, optionally filtered to one user."""
        ...


@runtime_checkable
class ProceduralStore(Protocol):
    """Durable key→value store of learned rules/policies (procedural memory).

    Procedural memory is "how to do things": durable guidance the agent applies
    across runs (e.g. tone rules, which tool to prefer). Modeled as simple
    namespaced key/value text so it's easy to read, edit, and version later.
    """

    def get_rule(self, key: str) -> str | None:
        """Return a rule's value, or ``None`` if unset."""
        ...

    def set_rule(self, key: str, value: str) -> None:
        """Create or update a rule."""
        ...

    def all_rules(self) -> dict[str, str]:
        """Return every rule as a mapping."""
        ...
