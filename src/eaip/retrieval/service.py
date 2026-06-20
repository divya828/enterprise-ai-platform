"""Assembly of the retrieval + answer stack from settings.

A small composition root so callers (the FastAPI app, the demo script) get a
ready-to-use object without wiring up six components by hand. It opens the index,
constructs the embedder/reranker/provider from config, and exposes a single
``ask`` method: retrieve, then ground an answer.
"""

from __future__ import annotations

from dataclasses import dataclass

from eaip.config import Settings, get_settings
from eaip.embeddings import get_embedder
from eaip.index.store import ChunkIndex
from eaip.providers import get_provider
from eaip.retrieval.answerer import Answer, GroundedAnswerer
from eaip.retrieval.dense import DenseRetriever
from eaip.retrieval.pipeline import HybridRetriever, Principal, RetrievalResult
from eaip.retrieval.reranker_factory import get_reranker
from eaip.retrieval.sparse import SparseRetriever


@dataclass(frozen=True)
class AskResult:
    """The answer plus the retrieval diagnostics that produced it."""

    answer: Answer
    retrieval: RetrievalResult


class RetrievalService:
    """Composition root: hybrid retrieval + grounded answering, configured."""

    def __init__(self, index: ChunkIndex, retriever: HybridRetriever, answerer: GroundedAnswerer):
        self._index = index
        self._retriever = retriever
        self._answerer = answerer

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> RetrievalService:
        settings = settings or get_settings()
        embedder = get_embedder(settings)
        index = ChunkIndex.open(
            path=str(settings.qdrant_path),
            collection=settings.qdrant_collection,
            dim=embedder.dim,
        )
        retriever = HybridRetriever(
            dense=DenseRetriever(index, embedder),
            sparse=SparseRetriever(index),
            reranker=get_reranker(settings),
            shortlist=settings.retrieval_shortlist,
            top_k=settings.retrieval_top_k,
        )
        answerer = GroundedAnswerer(get_provider(settings), min_score=settings.answer_min_score)
        return cls(index, retriever, answerer)

    def retrieve(self, query: str, principal: Principal) -> RetrievalResult:
        return self._retriever.retrieve(query, principal)

    def ask(self, query: str, principal: Principal) -> AskResult:
        result = self._retriever.retrieve(query, principal)
        answer = self._answerer.answer(result)
        return AskResult(answer=answer, retrieval=result)
