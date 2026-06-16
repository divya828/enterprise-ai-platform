"""Tests for the scripted stub provider — the offline, deterministic default.

These prove the stub can do everything later phases will lean on: echo with no
script, replay scripted text, simulate a tool call, record inputs, and flag
unconsumed responses.
"""

from eaip.providers import Completion, Message, Role
from eaip.providers.base import ProviderError
from eaip.providers.stub import StubProvider


def test_echo_fallback_is_deterministic():
    provider = StubProvider()
    out = provider.complete([Message(Role.USER, "hello world")])
    assert out.text == "[stub] received: hello world"
    assert out.model == "stub"
    # Usage is estimated but stable, so observability code has numbers to work with.
    assert out.usage.total_tokens > 0


def test_scripted_text_replayed_in_order():
    provider = StubProvider([Completion(text="first"), Completion(text="second")])
    assert provider.complete([Message(Role.USER, "x")]).text == "first"
    assert provider.complete([Message(Role.USER, "y")]).text == "second"
    # Queue drained → falls back to echo.
    assert provider.complete([Message(Role.USER, "z")]).text == "[stub] received: z"


def test_scripted_tool_call_is_simulated():
    provider = StubProvider()
    provider.queue_tool_call("search", {"q": "vpn setup"})
    out = provider.complete([Message(Role.USER, "how do I set up the vpn?")])
    assert out.wants_tools
    assert out.tool_calls[0].name == "search"
    assert out.tool_calls[0].arguments == {"q": "vpn setup"}


def test_records_calls_for_assertions():
    provider = StubProvider()
    provider.queue_text("ok")
    provider.complete([Message(Role.SYSTEM, "be terse"), Message(Role.USER, "hi")])
    assert len(provider.calls) == 1
    assert provider.calls[0][1].content == "hi"


def test_assert_drained_flags_unused_responses():
    provider = StubProvider([Completion(text="unused")])
    try:
        provider.assert_drained()
    except ProviderError as exc:
        assert "never consumed" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ProviderError for unconsumed response")
