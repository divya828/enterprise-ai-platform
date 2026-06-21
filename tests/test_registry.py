"""Tests for the governed prompt registry and agent lifecycle.

Covers the Phase 4 deliverables: prompt versioning + rollback, lifecycle
transition rules, RBAC enforcement on mutations, tenant isolation, and the
append-only audit trail these operations write.
"""

from __future__ import annotations

import pytest

from eaip.platform.rbac import PermissionDenied
from eaip.platform.registry import AgentRegistry, LifecycleError, PromptRegistry
from eaip.security import Principal
from eaip.storage import InMemoryStateStore, LifecycleState

_NOW = "2026-01-01T00:00:00+00:00"


def _builder(tenant="acme"):
    return Principal.of("dev@acme.test", tenant=tenant, role="builder")


def _viewer(tenant="acme"):
    return Principal.of("user@acme.test", tenant=tenant, role="viewer")


# --- prompt registry -------------------------------------------------------


def test_prompt_versioning_and_rollback():
    store = InMemoryStateStore()
    reg = PromptRegistry(store, store, now=_NOW)
    p = _builder()

    reg.add_version(p, "greeting", "v1: hello")
    reg.add_version(p, "greeting", "v2: hi there")
    assert reg.active(p, "greeting").version == 2  # newest is active

    # Roll back to v1 — one step.
    rolled = reg.rollback(p, "greeting", 1)
    assert rolled.version == 1
    assert reg.active(p, "greeting").text == "v1: hello"
    # History is intact, newest first.
    assert [v.version for v in reg.history(p, "greeting")] == [2, 1]


def test_prompt_mutations_require_builder_role():
    store = InMemoryStateStore()
    reg = PromptRegistry(store, store, now=_NOW)
    with pytest.raises(PermissionDenied):
        reg.add_version(_viewer(), "greeting", "nope")


def test_prompt_registry_is_tenant_isolated():
    store = InMemoryStateStore()
    reg = PromptRegistry(store, store, now=_NOW)
    reg.add_version(_builder("acme"), "greeting", "acme prompt")
    # A different tenant sees nothing for the same name.
    assert reg.active(_viewer("globex"), "greeting") is None


def test_prompt_operations_are_audited():
    store = InMemoryStateStore()
    reg = PromptRegistry(store, store, now=_NOW)
    p = _builder()
    reg.add_version(p, "greeting", "v1")
    reg.rollback(p, "greeting", 1)
    actions = [e.action for e in store.events(tenant="acme")]
    assert "prompt.add_version" in actions
    assert "prompt.rollback" in actions


# --- agent lifecycle -------------------------------------------------------


def test_agent_lifecycle_happy_path():
    store = InMemoryStateStore()
    reg = AgentRegistry(store, store, now=_NOW)
    p = _builder()

    reg.create(p, "support-bot", name="Support Bot", prompt_name="greeting", tools=["send_email"])
    assert reg.get(p, "support-bot").state is LifecycleState.DRAFT

    reg.transition(p, "support-bot", LifecycleState.TEST)
    reg.transition(p, "support-bot", LifecycleState.PUBLISHED)
    assert reg.get(p, "support-bot").state is LifecycleState.PUBLISHED
    reg.transition(p, "support-bot", LifecycleState.DEPRECATED)
    assert reg.get(p, "support-bot").state is LifecycleState.DEPRECATED


def test_illegal_lifecycle_transition_rejected():
    store = InMemoryStateStore()
    reg = AgentRegistry(store, store, now=_NOW)
    p = _builder()
    reg.create(p, "bot", name="Bot", prompt_name="greeting", tools=[])
    # draft -> published is not allowed (must go through test).
    with pytest.raises(LifecycleError):
        reg.transition(p, "bot", LifecycleState.PUBLISHED)


def test_agent_mutations_require_builder_role():
    store = InMemoryStateStore()
    reg = AgentRegistry(store, store, now=_NOW)
    with pytest.raises(PermissionDenied):
        reg.create(_viewer(), "bot", name="Bot", prompt_name="greeting", tools=[])


def test_agent_lifecycle_transitions_are_audited():
    store = InMemoryStateStore()
    reg = AgentRegistry(store, store, now=_NOW)
    p = _builder()
    reg.create(p, "bot", name="Bot", prompt_name="greeting", tools=[])
    reg.transition(p, "bot", LifecycleState.TEST)
    details = [e.detail for e in store.events(tenant="acme") if e.action == "agent.transition"]
    assert any("draft->test" in d for d in details)
