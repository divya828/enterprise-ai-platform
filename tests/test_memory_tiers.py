"""Tests for the four memory tiers wired into orchestration.

* in-context  — the AgentState carried through a run (exercised by every graph test)
* episodic    — past runs recorded in the store (test_orchestration covers recording)
* semantic    — the RAG corpus (Phase 2 retrieval; used by the knowledge path)
* procedural  — durable learned rules

This file focuses on episodic + procedural persistence semantics across both
store backends, since those are the durable tiers introduced in Phase 3.
"""

from __future__ import annotations

import pytest

from eaip.storage import Episode, InMemoryStateStore, SqliteStateStore


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        return InMemoryStateStore()
    return SqliteStateStore(tmp_path / "mem.db")


def test_procedural_rules_roundtrip(store):
    assert store.get_rule("tone") is None
    store.set_rule("tone", "concise and direct")
    assert store.get_rule("tone") == "concise and direct"
    store.set_rule("tone", "warm")  # update
    assert store.get_rule("tone") == "warm"
    assert store.all_rules() == {"tone": "warm"}


def test_episodes_recorded_and_ordered_newest_first(store):
    store.record_episode(
        Episode("t1", "u@x", "q1", "knowledge", "completed", "2026-01-01T00:00:00+00:00")
    )
    store.record_episode(
        Episode("t2", "u@x", "q2", "action", "completed", "2026-01-03T00:00:00+00:00")
    )
    store.record_episode(
        Episode("t3", "v@x", "q3", "knowledge", "completed", "2026-01-02T00:00:00+00:00")
    )

    all_eps = store.recent_episodes(limit=10)
    assert [e.thread_id for e in all_eps] == ["t2", "t3", "t1"]  # newest first


def test_episodes_filtered_by_user(store):
    store.record_episode(
        Episode("t1", "alice@x", "q", "knowledge", "completed", "2026-01-01T00:00:00+00:00")
    )
    store.record_episode(
        Episode("t2", "bob@x", "q", "action", "completed", "2026-01-02T00:00:00+00:00")
    )
    alice = store.recent_episodes(user="alice@x")
    assert {e.thread_id for e in alice} == {"t1"}


def test_recording_same_thread_replaces_not_duplicates(store):
    store.record_episode(
        Episode("t1", "u@x", "q", "knowledge", "in_progress", "2026-01-01T00:00:00+00:00")
    )
    store.record_episode(
        Episode("t1", "u@x", "q", "knowledge", "completed", "2026-01-01T00:01:00+00:00")
    )
    eps = store.recent_episodes()
    assert len(eps) == 1
    assert eps[0].outcome == "completed"  # the resume updated, not appended
