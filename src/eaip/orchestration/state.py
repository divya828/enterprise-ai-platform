"""The agent's shared state — the typed backbone of the LangGraph graph.

LangGraph is a *stateful* state machine: every node receives the current state
and returns a partial update, which LangGraph merges in. Making the state an
explicit, typed object (rather than an opaque dict passed around) is half the
value of using a graph framework — you can see exactly what flows between nodes,
the checkpointer can persist it, and a human-in-the-loop interrupt can pause on it
and resume from it.

This ``AgentState`` is also where the **in-context memory tier** lives: the
running conversation/scratch for one run. The other three tiers sit outside it
(episodic = past runs in SQLite; semantic = the RAG corpus; procedural = learned
rules) and are pulled into the state by nodes as needed.

A note on reducers: most fields here are *replaced* on each update (the default).
``messages`` uses LangGraph's ``add_messages`` reducer so node updates *append*
to the transcript instead of overwriting it — that's what makes it a growing log.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """State threaded through every node of the agent graph.

    ``total=False`` so nodes may return partial updates (only the keys they
    change). Field groups:

    * **input** — what the run is about.
    * **routing** — the supervisor's decision.
    * **knowledge** — retrieval results and the grounded draft/critique.
    * **action** — a requested sensitive tool call and its approval/result.
    * **control** — loop-safety counters and the terminal outcome.
    """

    # --- input ---
    query: str
    user: str
    groups: list[str]

    # --- conversation / in-context memory (append-reducer) ---
    messages: Annotated[list[Any], add_messages]

    # --- routing (supervisor) ---
    route: str  # "knowledge" | "action"
    plan: list[str]

    # --- knowledge path ---
    citations: list[dict]  # serialized Citation dicts for provenance
    draft: str
    critique: str
    revisions: int  # how many draft->critic cycles have run

    # --- action path (sensitive tools, HITL) ---
    tool_name: str
    tool_args: dict
    approval: str  # "pending" | "approved" | "denied" | "expired"
    tool_result: str

    # --- control / loop safety ---
    iterations: int
    stopped_reason: str  # set when a safety limit halts the run
    answer: str  # the final user-facing answer

    # --- internal plumbing ---
    # The thread id, stamped by the runner so nodes can find their (un-checkpointed)
    # LoopBudget in the per-run registry. Declared here so LangGraph carries it.
    _thread_id: str
