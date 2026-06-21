"""Tests for per-tenant rate limiting, token budgets, and the audit log."""

from __future__ import annotations

from eaip.platform.limits import RateLimiter, TokenBudget
from eaip.storage import AuditEvent, InMemoryStateStore

# --- rate limiting (sliding window, fake clock) -----------------------------


def test_rate_limiter_throttles_over_the_cap():
    now = {"t": 0.0}
    rl = RateLimiter(requests_per_minute=2, clock=lambda: now["t"])
    assert rl.check_and_record("acme").allowed
    assert rl.check_and_record("acme").allowed
    third = rl.check_and_record("acme")  # 3rd within the minute
    assert not third.allowed
    assert "rate limit" in third.reason


def test_rate_limit_window_slides():
    now = {"t": 0.0}
    rl = RateLimiter(requests_per_minute=1, clock=lambda: now["t"])
    assert rl.check_and_record("acme").allowed
    assert not rl.check_and_record("acme").allowed  # blocked at t=0
    now["t"] = 61.0  # a minute later
    assert rl.check_and_record("acme").allowed  # window cleared


def test_rate_limit_is_per_tenant():
    now = {"t": 0.0}
    rl = RateLimiter(requests_per_minute=1, clock=lambda: now["t"])
    assert rl.check_and_record("acme").allowed
    # A different tenant has its own window.
    assert rl.check_and_record("globex").allowed


# --- token budget (durable) -------------------------------------------------


def test_token_budget_throttles_when_exceeded():
    store = InMemoryStateStore()
    budget = TokenBudget(store, daily_budget=100)
    day = "2026-01-01"
    assert budget.check("acme", day).allowed  # nothing spent yet
    budget.record("acme", tokens=120, day=day)  # blow the budget
    decision = budget.check("acme", day)
    assert not decision.allowed
    assert "token budget" in decision.reason


def test_token_budget_is_per_tenant_and_per_day():
    store = InMemoryStateStore()
    budget = TokenBudget(store, daily_budget=100)
    budget.record("acme", tokens=120, day="2026-01-01")
    # Other tenant unaffected; same tenant on another day unaffected.
    assert budget.check("globex", "2026-01-01").allowed
    assert budget.check("acme", "2026-01-02").allowed


def test_usage_supports_cost_attribution():
    store = InMemoryStateStore()
    budget = TokenBudget(store, daily_budget=10_000)
    budget.record("acme", tokens=300, day="2026-01-01")
    budget.record("acme", tokens=200, day="2026-01-01")
    totals = store.usage_for_day("acme", "2026-01-01")
    assert totals.requests == 2
    assert totals.tokens == 500  # → cost = 500 * price-per-token


# --- audit log (append-only) ------------------------------------------------


def test_audit_log_is_append_only_and_tenant_scoped():
    store = InMemoryStateStore()
    store.append_event(AuditEvent("acme", "u1", "ask", "q1", "", "2026-01-01T00:00:00+00:00"))
    store.append_event(AuditEvent("acme", "u2", "ask", "q2", "", "2026-01-01T00:01:00+00:00"))
    store.append_event(AuditEvent("globex", "u3", "ask", "q3", "", "2026-01-01T00:02:00+00:00"))

    acme = store.events(tenant="acme")
    assert len(acme) == 2
    assert acme[0].target == "q2"  # newest first
    # No update/delete API exists — the AuditStore protocol only appends + reads.
    assert not hasattr(store, "delete_event")
    assert not hasattr(store, "update_event")
    # Tenant scoping.
    assert len(store.events(tenant="globex")) == 1
