"""Provider factory — turns config into a concrete :class:`LLMProvider`.

This is the single place that knows about all backends. Callers ask for a
provider via :func:`get_provider` and receive something satisfying the
``LLMProvider`` protocol; they never import a concrete backend or an SDK. Adding
a new backend means adding one branch here and nothing else changes.
"""

from __future__ import annotations

from eaip.config.settings import LLMProvider as ProviderName
from eaip.config.settings import Settings, get_settings
from eaip.providers.base import LLMProvider, ProviderError


def get_provider(settings: Settings | None = None) -> LLMProvider:
    """Construct the configured provider.

    The stub backend is constructed eagerly (cheap, offline). The real backends
    lazy-import their SDKs inside their constructors, so importing this module
    never pulls in ``anthropic``/``openai``.
    """
    settings = settings or get_settings()
    provider = settings.llm_provider

    if provider is ProviderName.STUB:
        from eaip.providers.stub import StubProvider  # noqa: PLC0415

        return StubProvider()

    if provider is ProviderName.OLLAMA:
        from eaip.providers.ollama import OllamaProvider  # noqa: PLC0415

        return OllamaProvider(base_url=settings.ollama_base_url, model=settings.llm_model)

    if provider is ProviderName.ANTHROPIC:
        from eaip.providers.anthropic import AnthropicProvider  # noqa: PLC0415

        return AnthropicProvider(api_key=settings.anthropic_api_key or "", model=settings.llm_model)

    if provider is ProviderName.OPENAI:
        from eaip.providers.openai import OpenAIProvider  # noqa: PLC0415

        return OpenAIProvider(api_key=settings.openai_api_key or "", model=settings.llm_model)

    raise ProviderError(f"Unknown provider: {provider!r}")  # pragma: no cover
