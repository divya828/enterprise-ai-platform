"""A deterministic, offline hashing embedder — the default for tests and CI.

This is the embedding analogue of the stub LLM provider: it produces stable
vectors with no model download, so the whole ingestion + retrieval pipeline runs
in CI in milliseconds. It is *not* semantically smart — it is the "hashing trick"
(a.k.a. feature hashing): each token is hashed to a bucket and contributes to
that dimension, then the vector is L2-normalized so cosine similarity behaves.

What it gives us: documents that share words land near each other, which is
enough to exercise and test the retrieval *plumbing* (dense search, RRF, ACL
filtering, dedup) deterministically. What it does NOT give us: true semantic
similarity (synonyms, paraphrase). For real answer quality, switch to the BGE
embedder via ``EAIP_EMBEDDER=bge``. The hashing embedder exists so correctness of
the machinery can be tested without conflating it with model quality.
"""

from __future__ import annotations

import hashlib
import math
import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class HashingEmbedder:
    """Feature-hashing embedder. Deterministic, offline, dependency-free."""

    name = "hashing"

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _tokenize(text):
            # Stable hash -> bucket and sign. blake2b keeps this reproducible
            # across processes (Python's built-in hash() is salted per run).
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            h = int.from_bytes(digest, "big")
            bucket = h % self.dim
            sign = 1.0 if (h >> 1) & 1 else -1.0
            vec[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            # Empty / punctuation-only text: return a valid unit-ish zero vector.
            return vec
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)
