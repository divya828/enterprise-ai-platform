"""Integration tests for the agent graph: routing, critic loop, HITL, safety."""

from __future__ import annotations

from eaip.orchestration.safety import LoopBudget

# --- supervisor routing -----------------------------------------------------


def test_supervisor_routes_knowledge_question(agent_runner):
    runner, _log, _store = agent_runner
    out = runner.run("how do I set up the vpn", user="u@x", groups=["everyone"], thread_id="r1")
    assert out.finished
    assert out.route == "knowledge"


def test_supervisor_routes_action_request(agent_runner):
    runner, _log, _store = agent_runner
    out = runner.run(
        "send email to team@acme.test", user="u@x", groups=["everyone"], thread_id="r2"
    )
    assert out.awaiting_approval  # action path pauses at the HITL gate
    assert out.route == "action"


# --- knowledge path + bounded critic loop -----------------------------------


def test_knowledge_path_produces_an_answer(agent_runner):
    runner, _log, _store = agent_runner
    out = runner.run("how does retrieval work", user="u@x", groups=["everyone"], thread_id="k1")
    assert out.finished
    assert out.answer  # non-empty


def test_critic_loop_is_bounded(agent_runner):
    """The draft->critic loop must not exceed max_revisions even if the critic
    keeps asking for changes."""
    runner, _log, _store = agent_runner
    out = runner.run("how do I set up the vpn", user="u@x", groups=["everyone"], thread_id="k2")
    # The run terminates (doesn't hang); revisions are capped at 2 in the fixture.
    assert out.finished
    assert not out.stopped_reason  # bounded loop finishes cleanly, not via a safety trip


# --- agent-loop safety in the graph -----------------------------------------


def test_kill_switch_stops_a_run(agent_runner):
    runner, _log, _store = agent_runner
    budget = LoopBudget()
    budget.kill()  # pre-tripped: the very first node check stops the run
    out = runner.run(
        "how do I set up the vpn", user="u@x", groups=["everyone"], thread_id="k3", budget=budget
    )
    assert out.finished
    assert out.stopped_reason == "kill_switch"
    assert "stopped" in out.answer.lower()


def test_iteration_cap_stops_a_run(agent_runner):
    runner, _log, _store = agent_runner
    budget = LoopBudget(max_iterations=1)  # trips after the supervisor's first tick
    out = runner.run(
        "how do I set up the vpn", user="u@x", groups=["everyone"], thread_id="k4", budget=budget
    )
    assert out.finished
    assert out.stopped_reason == "max_iterations"


# --- HITL: resume, idempotency, denial, role-routing, timeout ----------------


def test_hitl_approve_executes_the_tool(agent_runner):
    runner, log, _store = agent_runner
    runner.run("send email to ceo@acme.test", user="u@x", groups=["eng"], thread_id="h1")
    out = runner.resume("h1", approved=True, approver_roles=["admin"])
    assert out.finished
    assert "email sent to ceo@acme.test" in out.answer
    assert len(log.performed) == 1


def test_hitl_resume_is_idempotent(agent_runner):
    """Resuming an already-resolved run must not execute the side effect twice."""
    runner, log, _store = agent_runner
    runner.run("send email to ceo@acme.test", user="u@x", groups=["eng"], thread_id="h2")
    first = runner.resume("h2", approved=True, approver_roles=["admin"])
    second = runner.resume("h2", approved=True, approver_roles=["admin"])  # duplicate
    assert first.answer == second.answer
    assert len(log.performed) == 1  # exactly one send


def test_hitl_denial_does_not_execute(agent_runner):
    runner, log, _store = agent_runner
    runner.run("delete records where status=stale", user="u@x", groups=["eng"], thread_id="h3")
    out = runner.resume("h3", approved=False, approver_roles=["admin"])
    assert out.finished
    assert "not approved" in out.answer.lower()
    assert len(log.performed) == 0  # nothing deleted


def test_hitl_requires_an_approver_role(agent_runner):
    """A resumer without an approver role cannot approve — treated as denied."""
    runner, log, _store = agent_runner
    runner.run("send email to ceo@acme.test", user="u@x", groups=["eng"], thread_id="h4")
    out = runner.resume("h4", approved=True, approver_roles=["viewer"])  # not an approver
    assert out.finished
    assert "not approved" in out.answer.lower()
    assert len(log.performed) == 0


def test_hitl_approval_can_expire(agent_runner):
    """A stale approval (older than the TTL) expires rather than executing."""
    runner, log, _store = agent_runner
    runner.run("send email to ceo@acme.test", user="u@x", groups=["eng"], thread_id="h5")
    # Force an ancient requested_at so the TTL check (default 3600s) fails.
    out = runner.resume("h5", approved=True, approver_roles=["admin"], requested_at=0.0)
    assert out.finished
    assert "expired" in out.answer.lower()
    assert len(log.performed) == 0


def test_pending_approval_is_inspectable(agent_runner):
    runner, _log, _store = agent_runner
    runner.run("send email to ceo@acme.test", user="u@x", groups=["eng"], thread_id="h6")
    pending = runner.pending_approval("h6")
    assert pending is not None
    assert pending["tool"] == "send_email"


# --- tool failure handled inside the graph ----------------------------------


def test_failing_tool_does_not_crash_the_graph(retriever):
    """An approved-but-failing sensitive tool surfaces an error, not an exception."""
    from langgraph.checkpoint.memory import MemorySaver

    from eaip.orchestration import AgentNodes, AgentRunner
    from eaip.orchestration.tools import Tool, ToolError
    from eaip.providers.stub import StubProvider
    from eaip.retrieval.answerer import GroundedAnswerer

    def always_fails(_args):
        raise ToolError("smtp unavailable")

    broken_tools = {
        "send_email": Tool("send_email", "", {"type": "object"}, always_fails, sensitive=True)
    }
    answerer = GroundedAnswerer(StubProvider(), min_score=0.0)
    nodes = AgentNodes(StubProvider(), retriever, answerer, max_revisions=2)
    runner = AgentRunner(nodes, broken_tools, MemorySaver(), max_revisions=2)

    runner.run("send email to ceo@acme.test", user="u@x", groups=["eng"], thread_id="f1")
    out = runner.resume("f1", approved=True, approver_roles=["admin"])
    assert out.finished  # the run completed despite the tool failing
    assert "tool failed" in out.answer or "smtp unavailable" in out.answer


# --- episodic memory --------------------------------------------------------


def test_completed_runs_are_recorded_as_episodes(agent_runner):
    runner, _log, store = agent_runner
    runner.run("how do I set up the vpn", user="alice@x", groups=["everyone"], thread_id="e1")
    episodes = store.recent_episodes(user="alice@x")
    assert any(e.thread_id == "e1" and e.route == "knowledge" for e in episodes)
