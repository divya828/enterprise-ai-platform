"""In-memory StateStore — the default for tests.

Holds the :class:`SyncState` in process memory. Returns the *same* instance from
``load_state`` so a pipeline mutating the loaded state and re-saving is cheap and
obvious in tests. Not durable across processes — that's what the SQLite store is
for. Its existence is the payoff of the storage abstraction: the pipeline runs
unchanged against either backend.
"""

from __future__ import annotations

from eaip.storage.base import StateStore, SyncState


class InMemoryStateStore(StateStore):
    """A non-durable StateStore backed by an in-process object."""

    def __init__(self, state: SyncState | None = None) -> None:
        self._state = state or SyncState()

    def load_state(self) -> SyncState:
        return self._state

    def save_state(self, state: SyncState) -> None:
        self._state = state
