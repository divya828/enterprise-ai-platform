"""Graph wiring: topology, the HITL interrupt, and conditional routing.

This assembles the nodes from ``nodes.py`` into a LangGraph ``StateGraph``:

    START → supervisor ─┬─(knowledge)→ retrieve → draft → critic ─┐
                        │                            ▲   (revise)  │
                        │                            └─────────────┘
                        │                                          ↓
                        │                                       finalize → END
                        └─(action)→ propose_action → hitl_gate → execute → finalize

The two interesting edges:

* **critic → draft|finalize** (the bounded revision loop): a conditional edge
  loops back to ``draft`` while the critic wants changes AND the revision/budget
  limits allow, otherwise proceeds to ``finalize``.
* **hitl_gate** (human-in-the-loop): a node that calls ``interrupt()`` to pause
  the run and surface the proposed sensitive action for approval. The graph is
  compiled with a checkpointer, so the paused state is durable; the run resumes
  when the caller sends ``Command(resume=<decision>)``.

Sensitive-tool execution lives in ``execute_action``, which honors the approval
decision, role routing, and TTL, and is idempotent.
"""

from __future__ import annotations

import time

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from eaip.orchestration.nodes import AgentNodes, get_budget
from eaip.orchestration.state import AgentState
from eaip.orchestration.tools import Tool, run_tool

# Roles permitted to approve a sensitive action (role-routed approval).
APPROVER_ROLES = frozenset({"admin", "approver"})


def build_agent_graph(
    nodes: AgentNodes,
    tools: dict[str, Tool],
    *,
    max_revisions: int = 2,
    approval_ttl_s: float = 3600.0,
):
    """Build (uncompiled) the agent ``StateGraph``. Compile it with a checkpointer."""

    def hitl_gate(state: AgentState) -> dict:
        """Pause for human approval of the proposed sensitive action.

        ``interrupt`` suspends the graph and returns its payload to the caller. On
        resume, ``interrupt`` returns the value passed via ``Command(resume=...)``
        — here a dict with the approver's identity and decision. We enforce both
        role routing (only approver roles may decide) and a TTL (a stale approval
        expires).
        """
        decision = interrupt(
            {
                "type": "approval_request",
                "tool": state["tool_name"],
                "args": state["tool_args"],
                "requested_at": time.time(),
            }
        )
        # ``decision`` is whatever the resumer supplied.
        approver_roles = set(decision.get("approver_roles", []))
        approved = bool(decision.get("approved"))
        requested_at = decision.get("requested_at", time.time())

        if time.time() - requested_at > approval_ttl_s:
            return {"approval": "expired"}
        if not (approver_roles & APPROVER_ROLES):
            # The resumer isn't allowed to approve -> treat as denied, not approved.
            return {"approval": "denied"}
        return {"approval": "approved" if approved else "denied"}

    def execute_action(state: AgentState) -> dict:
        """Execute the sensitive tool iff approved — idempotently and resiliently."""
        if state.get("approval") != "approved":
            return {}
        tool = tools[state["tool_name"]]
        result = run_tool(tool, state["tool_args"])
        if result.ok:
            return {"tool_result": result.output}
        return {"tool_result": f"tool failed: {result.error}"}

    builder = StateGraph(AgentState)
    builder.add_node("supervisor", nodes.supervisor)
    builder.add_node("retrieve", nodes.retrieve)
    builder.add_node("draft", nodes.draft)
    builder.add_node("critic", nodes.critic)
    builder.add_node("propose_action", nodes.propose_action)
    builder.add_node("hitl_gate", hitl_gate)
    builder.add_node("execute_action", execute_action)
    builder.add_node("finalize", nodes.finalize)

    builder.add_edge(START, "supervisor")

    # supervisor routes to a specialist subgraph (or short-circuits if stopped).
    def route_from_supervisor(state: AgentState) -> str:
        if state.get("stopped_reason"):
            return "finalize"
        return "retrieve" if state.get("route") == "knowledge" else "propose_action"

    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {"retrieve": "retrieve", "propose_action": "propose_action", "finalize": "finalize"},
    )

    # knowledge path
    builder.add_edge("retrieve", "draft")
    builder.add_edge("draft", "critic")

    def route_from_critic(state: AgentState) -> str:
        """Bounded revision loop: revise again, or finalize.

        Two independent bounds: the revision cap and the safety budget. Either one
        being reached sends the run to ``finalize`` rather than back to ``draft``.
        """
        if state.get("stopped_reason"):
            return "finalize"
        budget = get_budget(state.get("_thread_id", "default"))
        wants_revision = state.get("critique", "") != "looks good"
        under_cap = state.get("revisions", 0) < max_revisions
        if wants_revision and under_cap and budget.check() is None:
            return "draft"
        return "finalize"

    builder.add_conditional_edges(
        "critic", route_from_critic, {"draft": "draft", "finalize": "finalize"}
    )

    # action path
    builder.add_edge("propose_action", "hitl_gate")
    builder.add_edge("hitl_gate", "execute_action")
    builder.add_edge("execute_action", "finalize")

    builder.add_edge("finalize", END)
    return builder
