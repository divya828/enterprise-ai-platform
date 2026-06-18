"""Dense embedding abstraction (hashing default, BGE opt-in)."""

from eaip.embeddings.base import Embedder
from eaip.embeddings.factory import get_embedder

__all__ = ["Embedder", "get_embedder"]
