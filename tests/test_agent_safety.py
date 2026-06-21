"""Tests for agent-loop safety limits (the 'stop a misbehaving agent' deliverable)."""

from __future__ import annotations

from eaip.orchestration.safety import LoopBudget


def test_max_iterations_stops_the_run():
    b = LoopBudget(max_iterations=3)
    for _ in range(2):
        b.tick()
        assert b.check() is None
    b.tick()  # third tick reaches the cap
    assert b.check() == "max_iterations"


def test_token_budget_stops_the_run():
    b = LoopBudget(max_iterations=100, token_budget=1000)
    b.tick(tokens=600)
    assert b.check() is None
    b.tick(tokens=600)  # cumulative 1200 >= 1000
    assert b.check() == "token_budget_exceeded"


def test_time_budget_stops_the_run(monkeypatch):
    """A fake clock makes the wall-clock limit deterministic (no real sleeping)."""
    import eaip.orchestration.safety as safety_mod

    now = {"t": 1000.0}
    monkeypatch.setattr(safety_mod.time, "perf_counter", lambda: now["t"])

    b = LoopBudget(max_iterations=100, time_budget_s=15.0)
    b._started_at = now["t"]  # type: ignore[attr-defined]  # anchor to the fake clock
    b.tick()
    assert b.check() is None  # no time has passed yet
    now["t"] = 1020.0  # 20s elapsed > 15s budget
    assert b.check() == "time_budget_exceeded"


def test_loop_detection_trips_on_repeated_signature():
    b = LoopBudget(max_iterations=100, loop_detection_threshold=3)
    for _ in range(2):
        b.tick(signature="same_step")
        assert b.check() is None
    b.tick(signature="same_step")  # third identical step
    assert b.check() == "loop_detected"


def test_distinct_signatures_do_not_trip_loop_detection():
    b = LoopBudget(max_iterations=100, loop_detection_threshold=3)
    for step in ["a", "b", "c", "a", "b"]:
        b.tick(signature=step)
    assert b.check() is None  # no single signature hit the threshold


def test_kill_switch_takes_priority():
    b = LoopBudget()
    b.tick()
    assert b.check() is None
    b.kill()
    assert b.check() == "kill_switch"
