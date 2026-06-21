"""Agent-loop safety: the limits that stop a misbehaving agent.

Autonomous agents loop — plan, act, observe, repeat — and a buggy or adversarial
run can loop forever, burn unbounded tokens/money, or get stuck repeating the same
step. A platform must be able to *stop* such a run. This module centralizes four
independent guardrails so every loop in the system (the main graph and the
draft→critic loop) consults the same budget:

* **max iterations** — a hard cap on loop steps.
* **token budget** — cumulative token spend ceiling.
* **time budget** — wall-clock ceiling for the whole run.
* **loop detection** — trip if the same step signature repeats too often (an
  agent stuck doing the same thing).

Plus a **kill switch**: an external flag that forces the next check to stop,
modeling an operator pulling the plug. ``check()`` returns the reason to stop, or
``None`` to continue — nodes call it and, if it returns a reason, set
``stopped_reason`` and route to the end. Keeping this as a plain object (not graph
nodes) makes the limits unit-testable in isolation, and reusable by any loop.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class LoopBudget:
    """Mutable budget tracker shared across a single agent run."""

    max_iterations: int = 12
    token_budget: int = 20_000
    time_budget_s: float = 30.0
    # Trip if a single node repeats this many times — the "stuck agent" backstop.
    # Set above the legitimate draft/critic revisit count (max_revisions + 1) so
    # the bounded revision loop doesn't false-trip; max_iterations is the harder cap.
    loop_detection_threshold: int = 5

    # --- live counters (updated as the run proceeds) ---
    iterations: int = 0
    tokens_used: int = 0
    _started_at: float = field(default_factory=time.perf_counter)
    _signatures: Counter[str] = field(default_factory=Counter)
    _killed: bool = False

    def tick(self, *, tokens: int = 0, signature: str | None = None) -> None:
        """Record one loop step: bump iteration, add tokens, note its signature."""
        self.iterations += 1
        self.tokens_used += tokens
        if signature is not None:
            self._signatures[signature] += 1

    def kill(self) -> None:
        """Trip the kill switch — the next :meth:`check` will stop the run."""
        self._killed = True

    def elapsed_s(self) -> float:
        return time.perf_counter() - self._started_at

    def check(self) -> str | None:
        """Return a stop reason if any limit is exceeded, else ``None``.

        Checked in priority order so the most urgent signal (kill switch) wins.
        """
        if self._killed:
            return "kill_switch"
        if self.iterations >= self.max_iterations:
            return "max_iterations"
        if self.tokens_used >= self.token_budget:
            return "token_budget_exceeded"
        if self.elapsed_s() >= self.time_budget_s:
            return "time_budget_exceeded"
        repeated = self._signatures.most_common(1)
        if repeated and repeated[0][1] >= self.loop_detection_threshold:
            return "loop_detected"
        return None
