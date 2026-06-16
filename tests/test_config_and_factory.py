"""Tests for settings parsing and provider selection."""

import pytest

from eaip.config.settings import LLMProvider, Settings, get_settings
from eaip.providers.base import ProviderError
from eaip.providers.factory import get_provider
from eaip.providers.stub import StubProvider


def test_defaults_are_offline_and_keyless():
    settings = Settings()
    assert settings.llm_provider is LLMProvider.STUB
    assert settings.anthropic_api_key is None
    assert settings.openai_api_key is None


def test_env_overrides_provider(monkeypatch):
    monkeypatch.setenv("EAIP_LLM_PROVIDER", "ollama")
    settings = Settings()
    assert settings.llm_provider is LLMProvider.OLLAMA


def test_factory_returns_stub_by_default():
    provider = get_provider(Settings())
    assert isinstance(provider, StubProvider)
    assert provider.name == "stub"


def test_anthropic_without_key_raises_clear_error():
    settings = Settings(llm_provider=LLMProvider.ANTHROPIC, anthropic_api_key=None)
    # Either the SDK is absent or the key is missing — both are ProviderError,
    # both carry an actionable message. We assert the failure is graceful.
    with pytest.raises(ProviderError):
        get_provider(settings)


def test_get_settings_is_cached():
    a = get_settings()
    b = get_settings()
    assert a is b
