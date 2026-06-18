"""The embedder contract.

An *embedder* maps text to a fixed-length dense vector such that semantically
similar texts land near each other (cosine distance). This is the heart of dense
retrieval. As with the LLM provider, the platform depends only on this protocol —
the offline hashing embedder and the real sentence-transformers embedder are
interchangeable, so tests run with no model download while production uses a real
one.

The contract distinguishes documents from queries because some real embedding
models are *asymmetric* (they prepend different instructions to passages vs.
queries — BGE is one). Exposing both methods keeps that capability available even
though the hashing embedder treats them identically.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Maps text to dense vectors. All vectors from one embedder share a dim."""

    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages for indexing."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query for search."""
        ...
