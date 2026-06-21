"""In-memory stores — the default for tests.

Holds all platform state (sync state, episodes, procedural rules) in process
memory. Returns the *same* state instance from ``load_state`` so a pipeline
mutating and re-saving is cheap and obvious in tests. Not durable across
processes — that's what the SQLite store is for. Its existence is the payoff of
the storage abstraction: consumers run unchanged against either backend.
"""

from __future__ import annotations

from eaip.storage.base import (
    Episode,
    EpisodicStore,
    ProceduralStore,
    StateStore,
    SyncState,
)


class InMemoryStateStore(StateStore, EpisodicStore, ProceduralStore):
    """A non-durable store backed by in-process objects."""

    def __init__(self, state: SyncState | None = None) -> None:
        self._state = state or SyncState()
        self._episodes: list[Episode] = []
        self._rules: dict[str, str] = {}

    # --- StateStore ---
    def load_state(self) -> SyncState:
        return self._state

    def save_state(self, state: SyncState) -> None:
        self._state = state

    # --- EpisodicStore ---
    def record_episode(self, episode: Episode) -> None:
        # Replace any existing episode with the same thread_id (resume case).
        self._episodes = [e for e in self._episodes if e.thread_id != episode.thread_id]
        self._episodes.append(episode)

    def recent_episodes(self, *, user: str | None = None, limit: int = 5) -> list[Episode]:
        eps = [e for e in self._episodes if user is None or e.user == user]
        return sorted(eps, key=lambda e: e.created_at, reverse=True)[:limit]

    # --- ProceduralStore ---
    def get_rule(self, key: str) -> str | None:
        return self._rules.get(key)

    def set_rule(self, key: str, value: str) -> None:
        self._rules[key] = value

    def all_rules(self) -> dict[str, str]:
        return dict(self._rules)
