"""The scripted stub provider — the offline, deterministic default.

Why this exists: a platform must be testable without a live model, and CI must
never call a paid/non-deterministic API. The stub makes every LLM-dependent
test reproducible by replaying a *preloaded queue* of responses. Because the
queue can contain either plain text or simulated tool calls, we can exercise
the full orchestration loop (Phase 3: plan -> call tool -> observe -> answer)
deterministically, with no model installed.

Two ways to use it:

* **Scripted** (recommended for tests): hand it a list of :class:`Completion`
  objects; each ``complete`` call pops the next one in order.
* **Echo fallback** (zero-config default, e.g. the Phase 0 hello smoke test):
  with an empty queue it returns a canned, deterministic acknowledgement of the
  last user message — enough to prove the abstraction is wired end to end.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from eaip.providers.base import LLMProvider, ProviderError
from eaip.providers.types import Completion, Message, Role, ToolCall, Usage


class StubProvider(LLMProvider):
    """A deterministic provider that replays a scripted response queue.

    Example (scripted)::

        provider = StubProvider([
            Completion(tool_calls=(ToolCall("search", {"q": "vpn"}),)),
            Completion(text="The VPN guide is at ..."),
        ])

    Example (echo default)::

        StubProvider().complete([Message(Role.USER, "hello")]).text
        # -> "[stub] received: hello"
    """

    name = "stub"

    def __init__(self, responses: Iterable[Completion] | None = None) -> None:
        self._queue: deque[Completion] = deque(responses or ())
        # Recorded inputs let tests assert on what the agent actually sent.
        self.calls: list[list[Message]] = []

    def queue(self, *completions: Completion) -> None:
        """Append more scripted responses (handy mid-test)."""
        self._queue.extend(completions)

    def queue_text(self, *texts: str) -> None:
        """Convenience: queue plain-text completions."""
        self._queue.extend(Completion(text=t, model=self.name) for t in texts)

    def queue_tool_call(self, name: str, arguments: dict | None = None) -> None:
        """Convenience: queue a single simulated tool call."""
        self._queue.append(Completion(tool_calls=(ToolCall(name=name, arguments=arguments or {}),)))

    def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion:
        self.calls.append(list(messages))

        if self._queue:
            scripted = self._queue.popleft()
            # Stamp the model name through so observability sees a consistent value.
            return Completion(
                text=scripted.text,
                tool_calls=scripted.tool_calls,
                model=scripted.model or self.name,
                usage=scripted.usage or _estimate_usage(messages, scripted.text),
            )

        # Echo fallback: deterministic acknowledgement of the last user turn.
        last_user = next(
            (m.content for m in reversed(messages) if m.role == Role.USER),
            "",
        )
        text = f"[stub] received: {last_user}"
        return Completion(text=text, model=self.name, usage=_estimate_usage(messages, text))

    def assert_drained(self) -> None:
        """Raise if scripted responses remain unused.

        Useful in tests to catch a model that did fewer steps than expected
        (e.g. an agent loop that exited early).
        """
        if self._queue:
            raise ProviderError(f"{len(self._queue)} scripted stub response(s) were never consumed")


def _estimate_usage(messages: list[Message], output: str) -> Usage:
    """Rough token estimate (~4 chars/token) so usage is non-zero and stable.

    A real provider returns exact counts; the stub approximates so that
    cost/observability code in later phases has plausible numbers to work with.
    """
    prompt_chars = sum(len(m.content) for m in messages)
    return Usage(prompt_tokens=prompt_chars // 4, completion_tokens=len(output) // 4)
