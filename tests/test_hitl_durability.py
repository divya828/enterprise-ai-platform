"""HITL durability: a paused approval survives a process restart (SqliteSaver).

Each ``with SqliteSaver.from_conn_string(...)`` block opens a fresh connection to
the same DB file, simulating separate processes. The run pauses for approval in
one block and resumes in another — proving the interrupt state is durable, not
just in-memory.
"""

from __future__ import annotations

from langgraph.checkpoint.sqlite import SqliteSaver

from eaip.orchestration import AgentNodes, AgentRunner, build_default_tools
from eaip.providers.stub import StubProvider
from eaip.retrieval.answerer import GroundedAnswerer


def test_hitl_state_survives_a_simulated_restart(tmp_path, retriever):
    db = str(tmp_path / "checkpoints.sqlite")
    tools, log = build_default_tools()
    answerer = GroundedAnswerer(StubProvider(), min_score=0.0)
    nodes = AgentNodes(StubProvider(), retriever, answerer, max_revisions=2)

    # "Process 1": start the run; it pauses awaiting approval.
    with SqliteSaver.from_conn_string(db) as cp:
        runner = AgentRunner(nodes, tools, cp, max_revisions=2)
        out = runner.run(
            "send email to ceo@acme.test", user="u@x", groups=["eng"], thread_id="dur1"
        )
        assert out.awaiting_approval

    # "Process 2": reopen the DB, rebuild the runner, resume from the durable state.
    with SqliteSaver.from_conn_string(db) as cp:
        runner = AgentRunner(nodes, tools, cp, max_revisions=2)
        pending = runner.pending_approval("dur1")
        assert pending is not None and pending["tool"] == "send_email"
        out = runner.resume("dur1", approved=True, approver_roles=["admin"])
        assert out.finished
        assert "email sent" in out.answer
    assert len(log.performed) == 1
