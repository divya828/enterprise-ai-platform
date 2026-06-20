"""Reranker factory — turns config into a concrete :class:`Reranker`."""

from __future__ import annotations

from eaip.config.settings import RerankerName, Settings, get_settings
from eaip.providers.base import ProviderError
from eaip.retrieval.reranker import LexicalReranker, Reranker


def get_reranker(settings: Settings | None = None) -> Reranker:
    """Construct the configured reranker (lexical default; BGE cross-encoder opt-in)."""
    settings = settings or get_settings()

    if settings.reranker is RerankerName.LEXICAL:
        return LexicalReranker()

    if settings.reranker is RerankerName.BGE:
        from eaip.retrieval.reranker import CrossEncoderReranker  # noqa: PLC0415

        return CrossEncoderReranker(model_name=settings.bge_reranker_model)

    raise ProviderError(f"Unknown reranker: {settings.reranker!r}")  # pragma: no cover
