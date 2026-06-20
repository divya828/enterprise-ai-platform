"""Storage-layer abstraction (SQLite by default, Postgres-swappable).

Public surface: the :class:`StateStore` interface, the :class:`SyncState`
transfer object, and the two backends. The pipeline depends on ``StateStore``,
never on a concrete backend.
"""

from eaip.storage.base import StateStore, SyncState
from eaip.storage.memory import InMemoryStateStore
from eaip.storage.sqlite import SqliteStateStore

__all__ = [
    "StateStore",
    "SyncState",
    "InMemoryStateStore",
    "SqliteStateStore",
]
