"""The LLM provider contract.

Every backend implements :class:`LLMProvider`. The platform depends only on
this protocol, never on a concrete vendor SDK — that indirection is the
"provider strategy" concept (avoiding lock-in, enabling an offline default and
deterministic tests).

We use ``typing.Protocol`` (structural typing) rather than an ABC so backends
are not forced to inherit from us; anything with a matching ``complete`` method
satisfies the contract. This keeps thin SDK wrappers free of boilerplate.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from eaip.providers.types import Completion, Message


@runtime_checkable
class LLMProvider(Protocol):
    """A minimal, synchronous chat-completion interface.

    Phase 0 only needs blocking completion. Streaming / async can be layered on
    later without changing callers, because callers depend on this method shape.
    """

    name: str

    def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion:
        """Return a single completion for ``messages``.

        ``model`` overrides the provider's configured default when given.
        ``temperature`` defaults to 0.0 because a platform favors reproducible
        behavior; callers that want sampling opt in explicitly.
        """
        ...


class ProviderError(RuntimeError):
    """Raised when a provider is misconfigured or a backend call fails.

    A dedicated error type lets the runtime distinguish provider problems
    (missing API key, backend down) from application bugs and surface a
    meaningful message instead of a raw SDK exception.
    """
