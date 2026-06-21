"""In-memory stores — the default for tests.

Holds all platform state in process memory and implements every store interface,
so tests run unchanged against this or the SQLite backend. Not durable across
processes — that's what the SQLite store is for.
"""

from __future__ import annotations

from eaip.storage.base import (
    AgentDefinition,
    AgentStore,
    AuditEvent,
    AuditStore,
    Episode,
    EpisodicStore,
    ProceduralStore,
    PromptStore,
    PromptVersion,
    StateStore,
    SyncState,
    UsageStore,
    UsageTotals,
)


class InMemoryStateStore(
    StateStore,
    EpisodicStore,
    ProceduralStore,
    AuditStore,
    PromptStore,
    AgentStore,
    UsageStore,
):
    """A non-durable store backed by in-process objects (all interfaces)."""

    def __init__(self, state: SyncState | None = None) -> None:
        self._state = state or SyncState()
        self._episodes: list[Episode] = []
        self._rules: dict[str, str] = {}
        self._audit: list[AuditEvent] = []
        # prompts: {(tenant, name): [versions...]}; active: {(tenant, name): version}
        self._prompts: dict[tuple[str, str], list[PromptVersion]] = {}
        self._active: dict[tuple[str, str], int] = {}
        self._agents: dict[tuple[str, str], AgentDefinition] = {}
        self._usage: dict[tuple[str, str], UsageTotals] = {}

    # --- StateStore ---
    def load_state(self) -> SyncState:
        return self._state

    def save_state(self, state: SyncState) -> None:
        self._state = state

    # --- EpisodicStore ---
    def record_episode(self, episode: Episode) -> None:
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

    # --- AuditStore (append-only) ---
    def append_event(self, event: AuditEvent) -> None:
        self._audit.append(event)

    def events(self, *, tenant: str, limit: int = 100) -> list[AuditEvent]:
        scoped = [e for e in self._audit if e.tenant == tenant]
        return list(reversed(scoped))[:limit]  # newest first

    # --- PromptStore ---
    def add_version(self, tenant: str, name: str, text: str, created_at: str) -> PromptVersion:
        key = (tenant, name)
        versions = self._prompts.setdefault(key, [])
        version = (versions[-1].version + 1) if versions else 1
        pv = PromptVersion(tenant, name, version, text, created_at)
        versions.append(pv)
        self._active[key] = version
        return pv

    def get_active(self, tenant: str, name: str) -> PromptVersion | None:
        key = (tenant, name)
        active = self._active.get(key)
        if active is None:
            return None
        return next(v for v in self._prompts[key] if v.version == active)

    def history(self, tenant: str, name: str) -> list[PromptVersion]:
        return sorted(self._prompts.get((tenant, name), []), key=lambda v: v.version, reverse=True)

    def pin(self, tenant: str, name: str, version: int) -> PromptVersion:
        key = (tenant, name)
        match = next((v for v in self._prompts.get(key, []) if v.version == version), None)
        if match is None:
            raise KeyError(f"no version {version} of prompt {name!r} for tenant {tenant!r}")
        self._active[key] = version
        return match

    # --- AgentStore ---
    def upsert_agent(self, agent: AgentDefinition) -> None:
        self._agents[(agent.tenant, agent.agent_id)] = agent

    def get_agent(self, tenant: str, agent_id: str) -> AgentDefinition | None:
        return self._agents.get((tenant, agent_id))

    def list_agents(self, tenant: str) -> list[AgentDefinition]:
        return sorted(
            (a for (t, _), a in self._agents.items() if t == tenant),
            key=lambda a: a.agent_id,
        )

    # --- UsageStore ---
    def record_usage(self, tenant: str, *, tokens: int, day: str) -> None:
        key = (tenant, day)
        cur = self._usage.get(key, UsageTotals(tenant, 0, 0))
        self._usage[key] = UsageTotals(tenant, cur.requests + 1, cur.tokens + tokens)

    def usage_for_day(self, tenant: str, day: str) -> UsageTotals:
        return self._usage.get((tenant, day), UsageTotals(tenant, 0, 0))
