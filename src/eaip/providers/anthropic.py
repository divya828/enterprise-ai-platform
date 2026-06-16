"""Anthropic (Claude) backend.

Lazy-imports the ``anthropic`` SDK so Phase 0 doesn't require it installed.
Install with the ``llm`` extra (``uv sync --extra llm``) and set
``ANTHROPIC_API_KEY`` + ``EAIP_LLM_PROVIDER=anthropic``.

Modeled on the current SDK (``client.messages.create``). One important quirk:
the Opus 4.x / Fable family rejects ``temperature``/``top_p`` (400), so we only
forward ``temperature`` to models that still accept it. This is exactly the kind
of vendor-specific detail the provider abstraction exists to hide from callers.
"""

from __future__ import annotations

from eaip.providers.base import LLMProvider, ProviderError
from eaip.providers.types import Completion, Message, Role, Usage

# Model id prefixes that reject sampling parameters (adaptive-thinking only).
_NO_SAMPLING_PREFIXES = ("claude-opus-4", "claude-fable", "claude-mythos")


class AnthropicProvider(LLMProvider):
    """Chat completions via the Anthropic Messages API."""

    name = "anthropic"

    def __init__(self, api_key: str, model: str) -> None:
        try:
            import anthropic  # noqa: PLC0415 (intentional lazy import)
        except ImportError as exc:  # pragma: no cover - exercised only with extra missing
            raise ProviderError(
                "The 'anthropic' package is not installed. Run `uv sync --extra llm`."
            ) from exc
        if not api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not set but provider=anthropic.")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion:
        target = model or self._model
        # Anthropic takes the system prompt as a separate top-level argument,
        # not as a message in the list — normalize here.
        system = "\n\n".join(m.content for m in messages if m.role == Role.SYSTEM)
        chat = [
            {"role": m.role.value, "content": m.content}
            for m in messages
            if m.role in (Role.USER, Role.ASSISTANT)
        ]
        kwargs: dict = {"model": target, "max_tokens": max_tokens, "messages": chat}
        if system:
            kwargs["system"] = system
        if not target.startswith(_NO_SAMPLING_PREFIXES):
            kwargs["temperature"] = temperature

        try:
            resp = self._client.messages.create(**kwargs)
        except Exception as exc:  # SDK raises typed errors; wrap for a uniform surface
            raise ProviderError(f"Anthropic request failed: {exc}") from exc

        text = "".join(block.text for block in resp.content if block.type == "text")
        usage = Usage(
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
        )
        return Completion(text=text, model=target, usage=usage)
