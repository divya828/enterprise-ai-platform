"""Demonstrate the Phase 4 platform capabilities, offline and self-contained.

Run with: ``uv run python scripts/governance.py``

Shows, against an in-memory store:
  * RBAC — a viewer is denied a builder-only operation (403-equivalent).
  * Prompt registry — versioning + one-step rollback.
  * Agent lifecycle — draft → test → published, and a rejected illegal jump.
  * Multi-tenancy — prompts/agents are invisible across tenants.
  * Audit log — the append-only trail of who did what.

No services or API key required; this exercises the governance layer directly.
"""

from __future__ import annotations

from eaip.platform.rbac import PermissionDenied
from eaip.platform.registry import AgentRegistry, LifecycleError, PromptRegistry
from eaip.security import Principal
from eaip.storage import InMemoryStateStore, LifecycleState

NOW = "2026-06-21T12:00:00+00:00"


def main() -> None:
    store = InMemoryStateStore()
    prompts = PromptRegistry(store, store, now=NOW)
    agents = AgentRegistry(store, store, now=NOW)

    builder = Principal.of("dev@acme.test", tenant="acme", role="builder")
    viewer = Principal.of("user@acme.test", tenant="acme", role="viewer")
    other_tenant = Principal.of("dev@globex.test", tenant="globex", role="builder")

    print("=== RBAC ===")
    try:
        prompts.add_version(viewer, "support_greeting", "hi!")
    except PermissionDenied as exc:
        print(f"viewer denied prompt edit: {exc}")

    print("\n=== Prompt registry (versioning + rollback) ===")
    prompts.add_version(builder, "support_greeting", "v1: Hello, how can I help?")
    prompts.add_version(builder, "support_greeting", "v2: Hey! What's up?")
    print(f"active after 2 adds: v{prompts.active(builder, 'support_greeting').version}")
    prompts.rollback(builder, "support_greeting", 1)
    active = prompts.active(builder, "support_greeting")
    print(f"after rollback to v1: v{active.version} — {active.text!r}")

    print("\n=== Agent lifecycle ===")
    agents.create(
        builder,
        "support-bot",
        name="Support Bot",
        prompt_name="support_greeting",
        tools=["send_email"],
    )
    print(f"created: {agents.get(builder, 'support-bot').state}")
    agents.transition(builder, "support-bot", LifecycleState.TEST)
    agents.transition(builder, "support-bot", LifecycleState.PUBLISHED)
    print(f"after test->published: {agents.get(builder, 'support-bot').state}")
    try:
        # Illegal: published can only go to deprecated.
        agents.transition(builder, "support-bot", LifecycleState.DRAFT)
    except LifecycleError as exc:
        print(f"illegal transition rejected: {exc}")

    print("\n=== Multi-tenancy isolation ===")
    print(f"globex sees acme's prompt? {prompts.active(other_tenant, 'support_greeting')}")
    print(f"globex sees acme's agent?  {agents.get(other_tenant, 'support-bot')}")

    print("\n=== Audit log (append-only, acme) ===")
    for e in reversed(store.events(tenant="acme")):  # oldest first for readability
        print(f"  {e.actor:18} {e.action:20} {e.target:18} {e.detail}")


if __name__ == "__main__":
    main()
