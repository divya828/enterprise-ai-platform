"""Storage-layer abstraction (SQLite by default, Postgres-swappable).

Public surface: the store interfaces, the transfer objects, and the two backends.
Consumers depend on an interface, never on a concrete backend.
"""

from eaip.storage.base import (
    Episode,
    EpisodicStore,
    ProceduralStore,
    StateStore,
    SyncState,
)
from eaip.storage.memory import InMemoryStateStore
from eaip.storage.sqlite import SqliteStateStore

__all__ = [
    "StateStore",
    "SyncState",
    "Episode",
    "EpisodicStore",
    "ProceduralStore",
    "InMemoryStateStore",
    "SqliteStateStore",
]
