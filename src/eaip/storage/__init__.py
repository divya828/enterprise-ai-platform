"""Storage-layer abstraction (SQLite by default, Postgres-swappable).

Public surface: the store interfaces, the transfer objects, and the two backends.
Consumers depend on an interface, never on a concrete backend.
"""

from eaip.storage.base import ALLOWED_TRANSITIONS as ALLOWED_TRANSITIONS
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
from eaip.storage.memory import InMemoryStateStore
from eaip.storage.sqlite import SqliteStateStore

__all__ = [
    "StateStore",
    "SyncState",
    "Episode",
    "EpisodicStore",
    "ProceduralStore",
    "AuditEvent",
    "AuditStore",
    "PromptVersion",
    "PromptStore",
    "AgentDefinition",
    "AgentStore",
    "LifecycleState",
    "ALLOWED_TRANSITIONS",
    "UsageTotals",
    "UsageStore",
    "InMemoryStateStore",
    "SqliteStateStore",
]
