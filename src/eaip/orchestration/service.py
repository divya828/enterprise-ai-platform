"""Build an :class:`AgentRunner` from settings (composition root).

Mirrors ``retrieval.service`` — one function that assembles the whole agent
(provider, retrieval stack, tools, memory, budget, durable checkpointer) so the
app, the demo, and tests don't each wire eight components by hand.

The SQLite checkpointer is a context manager, so this returns the checkpointer's
context manager alongside the builder; callers do::

    with build_runner_cm(settings) as runner:
        runner.run(...)
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from eaip.config import Settings, get_settings
from eaip.orchestration.nodes import AgentNodes
from eaip.orchestration.runner import AgentRunner, open_sqlite_checkpointer
from eaip.orchestration.safety import LoopBudget
from eaip.orchestration.tools import build_default_tools
from eaip.providers import get_provider
from eaip.retrieval.service import RetrievalService
from eaip.storage import SqliteStateStore


@contextmanager
def build_runner_cm(settings: Settings | None = None) -> Iterator[AgentRunner]:
    """Yield an :class:`AgentRunner` backed by the durable SQLite checkpointer."""
    settings = settings or get_settings()
    provider = get_provider(settings)
    retrieval = RetrievalService.from_settings(settings)
    tools, _log = build_default_tools()
    episodic = SqliteStateStore(settings.state_db_path)

    nodes = AgentNodes(
        provider,
        retrieval.retriever,  # reuse the assembled hybrid retriever
        retrieval.answerer,
        max_revisions=settings.critic_max_revisions,
    )

    def budget_factory() -> LoopBudget:
        return LoopBudget(
            max_iterations=settings.agent_max_iterations,
            token_budget=settings.agent_token_budget,
            time_budget_s=settings.agent_time_budget_s,
        )

    with open_sqlite_checkpointer(str(settings.checkpoint_db_path)) as checkpointer:
        yield AgentRunner(
            nodes,
            tools,
            checkpointer,
            max_revisions=settings.critic_max_revisions,
            approval_ttl_s=settings.approval_ttl_s,
            episodic=episodic,
            budget_factory=budget_factory,
        )
    episodic.close()
