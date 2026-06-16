"""Ollama backend — local models over HTTP, no API key required.

Uses Ollama's ``/api/chat`` endpoint via httpx (already a Phase 0 dep), so no
extra SDK is needed. Free and offline once a model is pulled, but note: small
local models handle multi-step tool use poorly, so Phase 3+ agent flows behave
much better on a hosted model. See README.
"""

from __future__ import annotations

import httpx

from eaip.providers.base import LLMProvider, ProviderError
from eaip.providers.types import Completion, Message, Usage


class OllamaProvider(LLMProvider):
    """Chat completions against a local Ollama server."""

    name = "ollama"

    def __init__(self, base_url: str, model: str, timeout: float = 120.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion:
        payload = {
            "model": model or self._model,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        try:
            resp = httpx.post(f"{self._base_url}/api/chat", json=payload, timeout=self._timeout)
            resp.raise_for_status()
        except httpx.HTTPError as exc:  # connection refused, timeout, 4xx/5xx
            raise ProviderError(
                f"Ollama request failed ({self._base_url}). Is `ollama serve` running "
                f"and the model pulled? Original error: {exc}"
            ) from exc

        data = resp.json()
        text = data.get("message", {}).get("content", "")
        usage = Usage(
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
        )
        return Completion(text=text, model=payload["model"], usage=usage)
