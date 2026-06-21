"""Graph nodes — the agent's behaviors, as plain functions over ``AgentState``.

Each node takes the state and returns a partial update. They're kept here,
separate from graph wiring (``graph.py``), so the *behavior* of each step reads
independently of the *topology* that connects them. The nodes implement:

* **supervisor** — an LLM-router that classifies the request and sets ``route``
  to "knowledge" (answer from the corpus) or "action" (perform a sensitive tool).
  This is the multi-agent *role separation* concept: one node decides, specialist
  subgraphs do the work.
* **knowledge path** — ``plan`` → ``retrieve`` (the Phase 2 RAG stack) → ``draft``
  → ``critic`` → (revise loop) → ``finalize``. The draft→critic loop is the
  collaboration/revision concept: a writer produces a draft, a critic reviews it,
  and the writer revises — bounded so it can't revise forever.
* **action path** — ``propose_action`` → (HITL interrupt, in ``graph.py``) →
  ``execute_action``. Sensitive tools are gated by human approval.

Every node first consults the run's :class:`LoopBudget` (looked up by thread id):
if a safety limit has tripped, it records ``stopped_reason`` and the graph routes
straight to ``finalize``. The budget is a live, time-aware object kept in a
per-run registry rather than in the checkpointed state (it isn't cleanly
serializable, and resuming should not reset the clock semantics we test).
"""

from __future__ import annotations

import json

from eaip.orchestration.safety import LoopBudget
from eaip.orchestration.state import AgentState
from eaip.providers.base import LLMProvider
from eaip.providers.types import Message, Role
from eaip.retrieval.answerer import GroundedAnswerer
from eaip.retrieval.pipeline import HybridRetriever, Principal

# Per-run live objects keyed by thread id (budgets aren't checkpointed).
_BUDGETS: dict[str, LoopBudget] = {}


def register_budget(thread_id: str, budget: LoopBudget) -> None:
    _BUDGETS[thread_id] = budget


def get_budget(thread_id: str) -> LoopBudget:
    return _BUDGETS.setdefault(thread_id, LoopBudget())


def clear_budget(thread_id: str) -> None:
    _BUDGETS.pop(thread_id, None)


# ---------------------------------------------------------------------------
# Routing keywords (used to keep the stub-LLM router deterministic in tests).
# A real LLM router would classify free-form; we give it a crisp rubric so the
# offline stub can drive it reproducibly.
# ---------------------------------------------------------------------------
_ACTION_HINTS = ("send email", "send an email", "delete", "remove records", "email to")


class AgentNodes:
    """Bundles the dependencies the nodes need and exposes them as callables."""

    def __init__(
        self,
        provider: LLMProvider,
        retriever: HybridRetriever,
        answerer: GroundedAnswerer,
        *,
        max_revisions: int = 2,
    ) -> None:
        self._provider = provider
        self._retriever = retriever
        self._answerer = answerer
        self._max_revisions = max_revisions

    # --- helpers ---
    @staticmethod
    def _thread_id(state: AgentState) -> str:
        # The runner stamps the thread id into state so nodes can find their budget.
        return state.get("_thread_id", "default")

    def _tripped(self, state: AgentState, node: str, *, tokens: int = 0) -> str | None:
        """Tick the run's budget for ``node`` and return a stop reason if tripped.

        The loop-detection signature is the node name, so the backstop fires when
        the *same* node repeats too often (a stuck agent) — distinct from the
        deliberately-bounded draft↔critic revision loop, which the revision cap
        governs separately.
        """
        budget = get_budget(self._thread_id(state))
        budget.tick(tokens=tokens, signature=node)
        return budget.check()

    # --- supervisor ---
    def supervisor(self, state: AgentState) -> dict:
        """LLM-router: decide whether this is a knowledge or an action request."""
        if reason := self._tripped(state, "supervisor"):
            return {"stopped_reason": reason, "route": "knowledge"}

        query = state["query"].lower()
        # Ask the provider to classify; the stub returns a scripted/echoed reply,
        # so we fall back to a keyword rubric to stay deterministic offline.
        self._provider.complete(
            [
                Message(
                    role=Role.SYSTEM, content="Classify the request as 'knowledge' or 'action'."
                ),
                Message(role=Role.USER, content=state["query"]),
            ],
            max_tokens=8,
        )
        route = "action" if any(h in query for h in _ACTION_HINTS) else "knowledge"
        return {"route": route, "plan": _plan_for(route)}

    # --- knowledge path ---
    def retrieve(self, state: AgentState) -> dict:
        """Semantic memory: pull permitted, ranked context for the query.

        Stores only JSON-serializable citation dicts in the state (not the live
        RetrievalResult) so the whole state checkpoints cleanly to SQLite.
        """
        if reason := self._tripped(state, "retrieve"):
            return {"stopped_reason": reason}
        principal = Principal.of(state["user"], state.get("groups", []))
        result = self._retriever.retrieve(state["query"], principal)
        answer = self._answerer.answer(result)
        citations = [
            {"label": c.label, "doc_id": c.doc_id, "title": c.title, "snippet": c.snippet}
            for c in answer.citations
        ]
        # The answerer already produced a grounded first draft; keep it for the
        # writer/critic loop to refine. Abstention shows up as the IDK text here.
        return {"citations": citations, "draft": answer.text}

    def draft(self, state: AgentState) -> dict:
        """Writer: refine the grounded answer, folding in any critic feedback."""
        if reason := self._tripped(state, "draft"):
            return {"stopped_reason": reason}
        draft = state.get("draft", "")
        critique = state.get("critique", "")
        # Apply the critique deterministically: append a citation marker if the
        # critic asked for one and we have citations to point at.
        wants_citation = critique and "citation" in critique
        if wants_citation and state.get("citations") and "[1]" not in draft:
            draft = f"{draft} [1]"
        return {"draft": draft}

    def critic(self, state: AgentState) -> dict:
        """Critic: review the draft and decide whether a revision is warranted.

        Returns a critique and bumps the revision counter. The graph's conditional
        edge uses ``revisions`` + the budget to decide loop vs. finalize, so this
        loop is bounded twice over (revision cap AND safety budget).
        """
        if reason := self._tripped(state, "critic"):
            return {"stopped_reason": reason}
        revisions = state.get("revisions", 0)
        draft = state.get("draft", "")
        # A crisp, deterministic critique: flag a missing citation marker once.
        needs_work = (
            bool(state.get("citations")) and "[1]" not in draft and "I don't know" not in draft
        )
        critique = "add an explicit citation marker" if needs_work else "looks good"
        return {"critique": critique, "revisions": revisions + 1}

    # --- action path ---
    def propose_action(self, state: AgentState) -> dict:
        """Parse the requested sensitive tool call from the query (mock parser)."""
        if reason := self._tripped(state, "propose_action"):
            return {"stopped_reason": reason}
        tool_name, tool_args = _parse_action(state["query"])
        return {"tool_name": tool_name, "tool_args": tool_args, "approval": "pending"}

    # --- finalize ---
    def finalize(self, state: AgentState) -> dict:
        """Produce the final user-facing answer from whichever path ran."""
        if reason := state.get("stopped_reason"):
            return {"answer": f"Run stopped: {reason}."}
        if state.get("route") == "action":
            result = state.get("tool_result")
            if state.get("approval") == "approved" and result:
                return {"answer": f"Done. {result}"}
            if state.get("approval") == "denied":
                return {"answer": "The requested action was not approved."}
            if state.get("approval") == "expired":
                return {"answer": "The approval request expired; no action taken."}
            return {"answer": "No action was performed."}
        # knowledge path
        return {
            "answer": state.get("draft", "") or "I don't know based on the available information."
        }


# ---------------------------------------------------------------------------
# small pure helpers
# ---------------------------------------------------------------------------


def _plan_for(route: str) -> list[str]:
    if route == "action":
        return ["identify the requested action", "obtain human approval", "execute if approved"]
    return ["retrieve relevant context", "draft a grounded answer", "review and finalize"]


def _parse_action(query: str) -> tuple[str, dict]:
    """Very small natural-language → tool-call parser for the demo/tests.

    Real systems use the LLM's tool-calling; here a deterministic parser keeps the
    action path testable offline. The idempotency key is derived from the query so
    the same request maps to the same key (so a resume can't double-execute).
    """
    import hashlib

    key = hashlib.blake2b(query.lower().encode(), digest_size=8).hexdigest()
    q = query.lower()
    if "delete" in q or "remove records" in q:
        return "delete_records", {"filter": query, "idempotency_key": key}
    # default sensitive action: email
    to = "team@acme.test"
    for token in query.split():
        if "@" in token:
            to = token.strip(".,")
            break
    return "send_email", {"to": to, "subject": "Automated message", "idempotency_key": key}


def serialize_args(args: dict) -> str:
    """Stable JSON for logging/auditing a tool call."""
    return json.dumps(args, sort_keys=True)
