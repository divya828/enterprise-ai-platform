"""Real dense embeddings via sentence-transformers (BGE-small by default).

Opt-in (``EAIP_EMBEDDER=bge``). Lazy-imports ``sentence_transformers`` so the
default install/test path never loads torch. On first use the model (~130MB)
downloads from HuggingFace and is cached locally; subsequent runs are offline.

BGE is an *asymmetric* model: it recommends prefixing queries with a short
instruction ("Represent this sentence for searching relevant passages:") while
passages are embedded as-is. We honor that here — it's a concrete example of why
the embedder contract separates ``embed_query`` from ``embed_documents``.
"""

from __future__ import annotations

from eaip.providers.base import ProviderError  # reuse the same error surface

_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class BGEEmbedder:
    """sentence-transformers embedder, defaulting to BAAI/bge-small-en-v1.5."""

    name = "bge"

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - only without the rag extra
            raise ProviderError(
                "sentence-transformers is not installed. Run `uv sync --extra rag`."
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(_QUERY_INSTRUCTION + text, normalize_embeddings=True)
        return vector.tolist()
