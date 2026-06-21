"""The agent runner — composition root for the orchestration graph.

Wires together the provider, retrieval stack, tools, safety budget, memory stores,
and the LangGraph checkpointer, and exposes a small API:

* ``run(query, user, groups, thread_id)`` — start a run. Returns an
  :class:`RunOutcome` that is either *finished* (with an answer) or *interrupted*
  (the run paused at the HITL gate, awaiting approval).
* ``resume(thread_id, approved, approver_roles)`` — supply a human decision and
  continue a paused run. Idempotent: resuming an already-finished run returns its
  result without re-executing side effects.

The checkpointer makes the paused state durable; the runner keeps the live
:class:`LoopBudget` in the node registry (not in the checkpoint). On completion it
records an episode (episodic memory).
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from eaip.orchestration.graph import build_agent_graph
from eaip.orchestration.nodes import AgentNodes, clear_budget, register_budget
from eaip.orchestration.safety import LoopBudget
from eaip.orchestration.tools import Tool
from eaip.storage.base import Episode, EpisodicStore


@dataclass(frozen=True)
class RunOutcome:
    """Result of a run step: either finished, or paused awaiting approval."""

    thread_id: str
    finished: bool
    answer: str = ""
    interrupt: dict | None = None  # the approval request payload when paused
    route: str = ""
    stopped_reason: str = ""

    @property
    def awaiting_approval(self) -> bool:
        return not self.finished and self.interrupt is not None


class AgentRunner:
    """Compiles and drives the agent graph against a durable checkpointer."""

    def __init__(
        self,
        nodes: AgentNodes,
        tools: dict[str, Tool],
        checkpointer: Any,
        *,
        max_revisions: int = 2,
        approval_ttl_s: float = 3600.0,
        episodic: EpisodicStore | None = None,
        budget_factory=LoopBudget,
    ) -> None:
        builder = build_agent_graph(
            nodes, tools, max_revisions=max_revisions, approval_ttl_s=approval_ttl_s
        )
        self._graph = builder.compile(checkpointer=checkpointer)
        self._episodic = episodic
        self._budget_factory = budget_factory

    def run(
        self,
        query: str,
        *,
        user: str,
        groups: list[str] | None = None,
        thread_id: str,
        budget: LoopBudget | None = None,
    ) -> RunOutcome:
        """Start a new run; returns finished or awaiting-approval."""
        register_budget(thread_id, budget or self._budget_factory())
        initial = {
            "query": query,
            "user": user,
            "groups": groups or [],
            "_thread_id": thread_id,
            "iterations": 0,
            "revisions": 0,
        }
        state = self._graph.invoke(initial, _cfg(thread_id))
        return self._outcome(thread_id, state)

    def resume(
        self,
        thread_id: str,
        *,
        approved: bool,
        approver_roles: list[str],
        requested_at: float | None = None,
    ) -> RunOutcome:
        """Resume a paused run with a human approval decision.

        Idempotent: if the run already finished (e.g. this resume is a duplicate),
        the graph has no pending interrupt and returns the existing final state —
        the sensitive tool is not executed again (also guarded by the tool's
        idempotency key).
        """
        snapshot = self._graph.get_state(_cfg(thread_id))
        if not snapshot.interrupts:
            # Nothing is paused — already resolved. Return the current outcome.
            return self._outcome(thread_id, snapshot.values)

        pending = snapshot.interrupts[0].value
        decision = {
            "approved": approved,
            "approver_roles": approver_roles,
            "requested_at": requested_at
            if requested_at is not None
            else pending.get("requested_at"),
        }
        state = self._graph.invoke(Command(resume=decision), _cfg(thread_id))
        return self._outcome(thread_id, state)

    def pending_approval(self, thread_id: str) -> dict | None:
        """Return the pending approval request for a paused run, or None."""
        snapshot = self._graph.get_state(_cfg(thread_id))
        return snapshot.interrupts[0].value if snapshot.interrupts else None

    # --- internals ---
    def _outcome(self, thread_id: str, state: dict) -> RunOutcome:
        interrupts = state.get("__interrupt__")
        if interrupts:
            return RunOutcome(
                thread_id=thread_id,
                finished=False,
                interrupt=interrupts[0].value,
                route=state.get("route", ""),
            )
        answer = state.get("answer", "")
        route = state.get("route", "")
        stopped = state.get("stopped_reason", "")
        if self._episodic is not None and answer:
            self._episodic.record_episode(
                Episode(
                    thread_id=thread_id,
                    user=state.get("user", ""),
                    query=state.get("query", ""),
                    route=route,
                    outcome=stopped or "completed",
                    created_at=datetime.now(UTC).isoformat(),
                )
            )
        clear_budget(thread_id)
        return RunOutcome(
            thread_id=thread_id,
            finished=True,
            answer=answer,
            route=route,
            stopped_reason=stopped,
        )


def _cfg(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def open_sqlite_checkpointer(path: str) -> AbstractContextManager:
    """Open a durable SQLite checkpointer as a context manager.

    ``SqliteSaver.from_conn_string`` returns a context manager; callers should use
    ``with open_sqlite_checkpointer(path) as cp: runner = AgentRunner(..., cp)``.
    """
    return SqliteSaver.from_conn_string(path)
