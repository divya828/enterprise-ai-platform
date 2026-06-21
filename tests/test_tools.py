"""Tests for the tool catalog: resilient execution and idempotency."""

from __future__ import annotations

from eaip.orchestration.tools import Tool, ToolError, build_default_tools, run_tool


def _noop_sleep(_seconds: float) -> None:
    pass


def test_successful_tool_runs_once():
    calls = []
    tool = Tool("ok", "", {}, handler=lambda a: calls.append(1) or "done")
    result = run_tool(tool, {}, sleep=_noop_sleep)
    assert result.ok and result.output == "done" and result.attempts == 1
    assert len(calls) == 1


def test_transient_failure_is_retried_then_succeeds():
    attempts = {"n": 0}

    def flaky(_args):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ToolError("transient")
        return "recovered"

    result = run_tool(Tool("flaky", "", {}, flaky), {}, max_attempts=3, sleep=_noop_sleep)
    assert result.ok
    assert result.output == "recovered"
    assert result.attempts == 3


def test_persistent_failure_returns_error_not_exception():
    def broken(_args):
        raise ToolError("nope")

    result = run_tool(Tool("broken", "", {}, broken), {}, max_attempts=2, sleep=_noop_sleep)
    assert not result.ok
    assert "nope" in result.error
    assert result.attempts == 2  # the agent can reason about this, not crash


def test_unexpected_exception_is_caught():
    def buggy(_args):
        raise ValueError("boom")  # not a ToolError

    result = run_tool(Tool("buggy", "", {}, buggy), {}, max_attempts=1, sleep=_noop_sleep)
    assert not result.ok
    assert "unexpected error" in result.error


def test_slow_tool_is_treated_as_timeout():
    import time

    def slow(_args):
        time.sleep(0.02)
        return "too slow"

    result = run_tool(
        Tool("slow", "", {}, slow), {}, timeout_s=0.001, max_attempts=1, sleep=_noop_sleep
    )
    assert not result.ok
    assert "timeout" in result.error


def test_sensitive_tools_are_flagged_and_idempotent():
    tools, log = build_default_tools()
    assert tools["send_email"].sensitive
    assert tools["delete_records"].sensitive

    args = {"to": "x@acme.test", "idempotency_key": "k1"}
    r1 = run_tool(tools["send_email"], args, sleep=_noop_sleep)
    r2 = run_tool(tools["send_email"], args, sleep=_noop_sleep)  # same key -> replay
    assert r1.ok and r2.ok
    assert "idempotent replay" in r2.output
    assert len(log.performed) == 1  # the side effect happened exactly once
