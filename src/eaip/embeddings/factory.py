"""Embedder factory — turns config into a concrete :class:`Embedder`."""

from __future__ import annotations

from eaip.config.settings import EmbedderName, Settings, get_settings
from eaip.embeddings.base import Embedder
from eaip.providers.base import ProviderError


def get_embedder(settings: Settings | None = None) -> Embedder:
    """Construct the configured embedder (hashing default; BGE opt-in)."""
    settings = settings or get_settings()

    if settings.embedder is EmbedderName.HASHING:
        from eaip.embeddings.hashing import HashingEmbedder  # noqa: PLC0415

        return HashingEmbedder(dim=settings.embedding_dim)

    if settings.embedder is EmbedderName.BGE:
        from eaip.embeddings.bge import BGEEmbedder  # noqa: PLC0415

        return BGEEmbedder(model_name=settings.bge_model)

    raise ProviderError(f"Unknown embedder: {settings.embedder!r}")  # pragma: no cover
