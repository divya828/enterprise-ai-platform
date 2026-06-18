"""Tests for the hashing embedder (the offline default) and its factory."""

import math

from eaip.config.settings import EmbedderName, Settings
from eaip.embeddings import get_embedder
from eaip.embeddings.hashing import HashingEmbedder


def _cos(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_deterministic_and_correct_dim():
    e = HashingEmbedder(dim=64)
    v1 = e.embed_query("set up the vpn")
    v2 = e.embed_query("set up the vpn")
    assert v1 == v2
    assert len(v1) == 64
    # L2-normalized -> unit length.
    assert math.isclose(math.sqrt(_cos(v1, v1)), 1.0, rel_tol=1e-6)


def test_shared_words_are_more_similar_than_unrelated():
    e = HashingEmbedder(dim=512)
    base = e.embed_query("vpn client setup troubleshooting")
    similar = e.embed_query("vpn client connection setup")
    unrelated = e.embed_query("quarterly revenue forecast finance")
    assert _cos(base, similar) > _cos(base, unrelated)


def test_empty_text_is_safe():
    e = HashingEmbedder(dim=16)
    v = e.embed_query("   ")
    assert len(v) == 16
    assert all(x == 0.0 for x in v)


def test_factory_returns_hashing_by_default():
    e = get_embedder(Settings())
    assert e.name == "hashing"
    assert isinstance(e, HashingEmbedder)


def test_factory_honors_embedding_dim():
    e = get_embedder(Settings(embedder=EmbedderName.HASHING, embedding_dim=128))
    assert e.dim == 128
