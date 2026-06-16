"""Provider-neutral message and response types.

These types are the *contract* between the platform and any LLM backend. They
deliberately mirror the lowest common denominator across Anthropic / OpenAI /
Ollama (a list of role-tagged messages in, a single completion out) plus a
minimal tool-call representation we will need from Phase 3 onward.

Keeping these vendor-neutral is the whole point of the provider abstraction:
swapping ``EAIP_LLM_PROVIDER`` must not require touching orchestration code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Role(StrEnum):
    """Conversation roles, normalized across providers."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class Message:
    """A single conversation turn."""

    role: Role
    content: str


@dataclass(frozen=True)
class ToolCall:
    """A model's request to invoke a tool.

    Present so the stub can simulate tool-using agents deterministically in
    Phase 3 without a live model. ``arguments`` is the parsed JSON object.
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = "call_0"


@dataclass(frozen=True)
class Usage:
    """Token accounting for one completion (cost/observability in Phase 5)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class Completion:
    """A single model response.

    Either ``text`` is set (plain answer) or ``tool_calls`` is non-empty (the
    model wants to call tools). Both may coexist for providers that interleave.
    """

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    model: str = ""
    usage: Usage = field(default_factory=Usage)

    @property
    def wants_tools(self) -> bool:
        return len(self.tool_calls) > 0
