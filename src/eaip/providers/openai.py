"""OpenAI backend.

Lazy-imports the ``openai`` SDK. Install with the ``llm`` extra and set
``OPENAI_API_KEY`` + ``EAIP_LLM_PROVIDER=openai``. Uses the Chat Completions
API, whose role-tagged message shape maps directly onto our neutral types.
"""

from __future__ import annotations

from eaip.providers.base import LLMProvider, ProviderError
from eaip.providers.types import Completion, Message, Usage


class OpenAIProvider(LLMProvider):
    """Chat completions via the OpenAI Chat Completions API."""

    name = "openai"

    def __init__(self, api_key: str, model: str) -> None:
        try:
            import openai  # noqa: PLC0415 (intentional lazy import)
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(
                "The 'openai' package is not installed. Run `uv sync --extra llm`."
            ) from exc
        if not api_key:
            raise ProviderError("OPENAI_API_KEY is not set but provider=openai.")
        self._client = openai.OpenAI(api_key=api_key)
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
        chat = [{"role": m.role.value, "content": m.content} for m in messages]
        try:
            resp = self._client.chat.completions.create(
                model=target,
                messages=chat,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            raise ProviderError(f"OpenAI request failed: {exc}") from exc

        choice = resp.choices[0]
        usage = Usage(
            prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
        )
        return Completion(text=choice.message.content or "", model=target, usage=usage)
