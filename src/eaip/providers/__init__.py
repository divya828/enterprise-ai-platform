"""LLM provider abstraction.

Public surface: the :class:`LLMProvider` protocol, the neutral message/response
types, and the :func:`get_provider` factory. Concrete backends are intentionally
not re-exported here so importing this package stays free of SDK dependencies.
"""

from eaip.providers.base import LLMProvider, ProviderError
from eaip.providers.factory import get_provider
from eaip.providers.types import Completion, Message, Role, ToolCall, Usage

__all__ = [
    "LLMProvider",
    "ProviderError",
    "get_provider",
    "Completion",
    "Message",
    "Role",
    "ToolCall",
    "Usage",
]
