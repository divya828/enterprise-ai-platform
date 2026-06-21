"""Per-tenant rate limiting and token budgets (with cost attribution).

In a multi-tenant platform, one tenant must not exhaust shared capacity or run up
unbounded cost. Two complementary controls:

* **Rate limit** — a sliding-window cap on requests per minute per tenant. This
  protects latency/throughput from a burst.
* **Token budget** — a daily ceiling on tokens consumed per tenant. This protects
  cost. Usage is recorded per tenant per day (the same counters give us **cost
  attribution**: tokens × a price = each tenant's spend).

A request is admitted only if it passes both checks; otherwise it is *throttled*
(the API returns HTTP 429). The rate-limit window is held in memory (it's
ephemeral by nature); the token budget is checked against the durable per-tenant
daily usage in the store, so it survives restarts.

The clock is injectable so the sliding window is testable without real waiting.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass

from eaip.storage.base import UsageStore


@dataclass(frozen=True)
class Decision:
    """Whether a request is admitted, and why not if throttled."""

    allowed: bool
    reason: str = ""


class RateLimiter:
    """Sliding-window per-tenant request rate limiter (in-memory)."""

    def __init__(self, requests_per_minute: int, *, clock: Callable[[], float] = time.monotonic):
        self._rpm = requests_per_minute
        self._clock = clock
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def check_and_record(self, tenant: str) -> Decision:
        """Admit a request if the tenant is under its per-minute cap, recording it."""
        now = self._clock()
        window = self._events[tenant]
        cutoff = now - 60.0
        while window and window[0] < cutoff:
            window.popleft()
        if len(window) >= self._rpm:
            return Decision(False, f"rate limit exceeded ({self._rpm}/min)")
        window.append(now)
        return Decision(True)


class TokenBudget:
    """Per-tenant daily token budget, backed by durable usage counters."""

    def __init__(self, usage: UsageStore, daily_budget: int):
        self._usage = usage
        self._budget = daily_budget

    def check(self, tenant: str, day: str) -> Decision:
        """Admit a request if the tenant is under its daily token budget."""
        spent = self._usage.usage_for_day(tenant, day).tokens
        if spent >= self._budget:
            return Decision(False, f"daily token budget exceeded ({self._budget})")
        return Decision(True)

    def record(self, tenant: str, *, tokens: int, day: str) -> None:
        """Attribute a request's token usage to the tenant (cost attribution)."""
        self._usage.record_usage(tenant, tokens=tokens, day=day)
