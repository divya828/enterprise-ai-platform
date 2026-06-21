"""Governed prompt registry and agent lifecycle.

This layer composes the durable stores with the cross-cutting governance
concerns: every mutation is **RBAC-checked** (raises on insufficient role),
**tenant-scoped** (the principal's tenant is the only one touched), and
**audited** (an append-only event records who did what). Lifecycle transitions
are validated against the allowed state machine.

Keeping this composition in one place — rather than scattering RBAC/audit checks
across the API handlers — means the rules are enforced uniformly and tested once.
"""

from __future__ import annotations

from eaip.platform.rbac import Capability, require
from eaip.security import Principal
from eaip.storage.base import (
    ALLOWED_TRANSITIONS,
    AgentDefinition,
    AgentStore,
    AuditEvent,
    AuditStore,
    LifecycleState,
    PromptStore,
    PromptVersion,
)


class LifecycleError(ValueError):
    """Raised on a disallowed agent lifecycle transition."""


class PromptRegistry:
    """RBAC-checked, audited, tenant-scoped prompt versioning."""

    def __init__(self, prompts: PromptStore, audit: AuditStore, *, now: str) -> None:
        self._prompts = prompts
        self._audit = audit
        self._now = now  # ISO timestamp injected by the caller (no clock here)

    def add_version(self, principal: Principal, name: str, text: str) -> PromptVersion:
        require(principal.role, Capability.MANAGE_PROMPTS)
        pv = self._prompts.add_version(principal.tenant, name, text, self._now)
        self._audit.append_event(
            AuditEvent(
                principal.tenant,
                principal.user,
                "prompt.add_version",
                name,
                f"version={pv.version}",
                self._now,
            )
        )
        return pv

    def rollback(self, principal: Principal, name: str, version: int) -> PromptVersion:
        """Pin an earlier version as active — the one-step rollback."""
        require(principal.role, Capability.MANAGE_PROMPTS)
        pv = self._prompts.pin(principal.tenant, name, version)
        self._audit.append_event(
            AuditEvent(
                principal.tenant,
                principal.user,
                "prompt.rollback",
                name,
                f"pinned version={version}",
                self._now,
            )
        )
        return pv

    def active(self, principal: Principal, name: str) -> PromptVersion | None:
        require(principal.role, Capability.ASK)  # reading the active prompt is a low bar
        return self._prompts.get_active(principal.tenant, name)

    def history(self, principal: Principal, name: str) -> list[PromptVersion]:
        require(principal.role, Capability.MANAGE_PROMPTS)
        return self._prompts.history(principal.tenant, name)


class AgentRegistry:
    """RBAC-checked, audited agent definitions with a validated lifecycle."""

    def __init__(self, agents: AgentStore, audit: AuditStore, *, now: str) -> None:
        self._agents = agents
        self._audit = audit
        self._now = now

    def create(
        self,
        principal: Principal,
        agent_id: str,
        *,
        name: str,
        prompt_name: str,
        tools: list[str],
    ) -> AgentDefinition:
        """Create a new agent in the DRAFT state."""
        require(principal.role, Capability.MANAGE_AGENTS)
        agent = AgentDefinition(
            tenant=principal.tenant,
            agent_id=agent_id,
            name=name,
            prompt_name=prompt_name,
            tools=tools,
            state=LifecycleState.DRAFT,
            updated_at=self._now,
        )
        self._agents.upsert_agent(agent)
        self._audit.append_event(
            AuditEvent(
                principal.tenant, principal.user, "agent.create", agent_id, "draft", self._now
            )
        )
        return agent

    def transition(
        self, principal: Principal, agent_id: str, to_state: LifecycleState
    ) -> AgentDefinition:
        """Move an agent to a new lifecycle state, if the transition is allowed."""
        require(principal.role, Capability.MANAGE_AGENTS)
        agent = self._agents.get_agent(principal.tenant, agent_id)
        if agent is None:
            raise KeyError(f"no agent {agent_id!r} for tenant {principal.tenant!r}")
        if to_state not in ALLOWED_TRANSITIONS.get(agent.state, set()):
            raise LifecycleError(
                f"cannot transition agent {agent_id} from {agent.state} to {to_state}"
            )
        updated = AgentDefinition(
            tenant=agent.tenant,
            agent_id=agent.agent_id,
            name=agent.name,
            prompt_name=agent.prompt_name,
            tools=agent.tools,
            state=to_state,
            updated_at=self._now,
        )
        self._agents.upsert_agent(updated)
        self._audit.append_event(
            AuditEvent(
                principal.tenant,
                principal.user,
                "agent.transition",
                agent_id,
                f"{agent.state}->{to_state}",
                self._now,
            )
        )
        return updated

    def get(self, principal: Principal, agent_id: str) -> AgentDefinition | None:
        require(principal.role, Capability.ASK)
        return self._agents.get_agent(principal.tenant, agent_id)
