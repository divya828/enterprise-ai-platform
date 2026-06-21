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
from enum import StrEnum
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


# ---------------------------------------------------------------------------
# Phase 4 — platform governance: audit, prompt registry, agent lifecycle, usage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditEvent:
    """One append-only audit record: who did what, to what, when, for which tenant.

    The audit log is the platform's tamper-evident history. It is *append-only* —
    there is no update or delete API — so it can answer "who accessed this / who
    changed that / who approved which action" after the fact.
    """

    tenant: str
    actor: str  # the user id responsible
    action: str  # e.g. "ask", "prompt.rollback", "agent.publish", "approval.grant"
    target: str  # what was acted on (a prompt name, agent id, doc id, ...)
    detail: str  # free-form context (JSON or text)
    created_at: str  # ISO 8601 (passed in; the store does not call the clock)


@runtime_checkable
class AuditStore(Protocol):
    """Append-only store of audit events, scoped per tenant."""

    def append_event(self, event: AuditEvent) -> None:
        """Record one event. There is intentionally no update/delete."""
        ...

    def events(self, *, tenant: str, limit: int = 100) -> list[AuditEvent]:
        """Return the most recent events for a tenant, newest first."""
        ...


@dataclass(frozen=True)
class PromptVersion:
    """One immutable version of a named prompt, scoped per tenant."""

    tenant: str
    name: str
    version: int
    text: str
    created_at: str  # ISO 8601


@runtime_checkable
class PromptStore(Protocol):
    """Versioned prompt registry: append versions, pin one as active, roll back.

    Prompts are versioned because changing a prompt changes model behavior, and
    you need history, the ability to pin a known-good version, and a one-step
    rollback when a change regresses (proven by the Phase 5 eval gate).
    """

    def add_version(self, tenant: str, name: str, text: str, created_at: str) -> PromptVersion:
        """Append a new immutable version (auto-incrementing version number).

        The newly added version becomes the active (pinned) one.
        """
        ...

    def get_active(self, tenant: str, name: str) -> PromptVersion | None:
        """Return the currently pinned version of a prompt, or None."""
        ...

    def history(self, tenant: str, name: str) -> list[PromptVersion]:
        """Return all versions of a prompt, newest first."""
        ...

    def pin(self, tenant: str, name: str, version: int) -> PromptVersion:
        """Pin a specific existing version as active (rollback = pin an older one)."""
        ...


class LifecycleState(StrEnum):
    """The lifecycle of an agent definition.

    A defined progression — draft → test → published → deprecated — so an agent
    is built and validated before it serves traffic and is retired safely.
    Transitions are restricted (see :data:`ALLOWED_TRANSITIONS`).
    """

    DRAFT = "draft"
    TEST = "test"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"


# Allowed lifecycle transitions (forward, plus published->deprecated and a
# deprecated->published "un-retire"). Anything else is rejected.
ALLOWED_TRANSITIONS: dict[LifecycleState, set[LifecycleState]] = {
    LifecycleState.DRAFT: {LifecycleState.TEST},
    LifecycleState.TEST: {LifecycleState.PUBLISHED, LifecycleState.DRAFT},
    LifecycleState.PUBLISHED: {LifecycleState.DEPRECATED},
    LifecycleState.DEPRECATED: {LifecycleState.PUBLISHED},
}


@dataclass(frozen=True)
class AgentDefinition:
    """A declarative agent definition (knowledge sources, tools, prompt, state).

    This is the "define an agent, have the runtime execute it" abstraction: a
    named, tenant-scoped record describing an agent's configuration plus its
    current lifecycle state. The runtime reads a *published* definition to run.
    """

    tenant: str
    agent_id: str
    name: str
    prompt_name: str  # which registered prompt this agent uses
    tools: list[str]  # tool names from the catalog
    state: LifecycleState
    updated_at: str  # ISO 8601


@runtime_checkable
class AgentStore(Protocol):
    """Store of agent definitions + their lifecycle state, per tenant."""

    def upsert_agent(self, agent: AgentDefinition) -> None:
        """Create or replace an agent definition."""
        ...

    def get_agent(self, tenant: str, agent_id: str) -> AgentDefinition | None:
        """Return an agent definition, or None."""
        ...

    def list_agents(self, tenant: str) -> list[AgentDefinition]:
        """Return all agent definitions for a tenant."""
        ...


@dataclass(frozen=True)
class UsageTotals:
    """Aggregated per-tenant usage for cost attribution and budget checks."""

    tenant: str
    requests: int
    tokens: int


@runtime_checkable
class UsageStore(Protocol):
    """Per-tenant usage counters (requests + tokens) for budgets + cost attribution."""

    def record_usage(self, tenant: str, *, tokens: int, day: str) -> None:
        """Add one request's usage to a tenant's daily counter."""
        ...

    def usage_for_day(self, tenant: str, day: str) -> UsageTotals:
        """Return a tenant's totals for a given day (zeros if none)."""
        ...
