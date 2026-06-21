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
from eaip.index.resolver import TenantIndexResolver
from eaip.providers import get_provider
from eaip.retrieval.answerer import Answer, GroundedAnswerer
from eaip.retrieval.pipeline import HybridRetriever, Principal, RetrievalResult
from eaip.retrieval.reranker_factory import get_reranker


@dataclass(frozen=True)
class AskResult:
    """The answer plus the retrieval diagnostics that produced it."""

    answer: Answer
    retrieval: RetrievalResult


class RetrievalService:
    """Composition root: tenant-aware hybrid retrieval + grounded answering."""

    def __init__(self, retriever: HybridRetriever, answerer: GroundedAnswerer):
        self._retriever = retriever
        self._answerer = answerer

    @property
    def retriever(self) -> HybridRetriever:
        """The assembled hybrid retriever (reused by the orchestration layer)."""
        return self._retriever

    @property
    def answerer(self) -> GroundedAnswerer:
        """The grounded answerer (reused by the orchestration layer)."""
        return self._answerer

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> RetrievalService:
        settings = settings or get_settings()
        embedder = get_embedder(settings)
        resolver = TenantIndexResolver(
            path=str(settings.qdrant_path),
            collection_prefix=settings.qdrant_collection,
            dim=embedder.dim,
        )
        retriever = HybridRetriever(
            resolver,
            embedder,
            get_reranker(settings),
            shortlist=settings.retrieval_shortlist,
            top_k=settings.retrieval_top_k,
        )
        answerer = GroundedAnswerer(get_provider(settings), min_score=settings.answer_min_score)
        return cls(retriever, answerer)

    def retrieve(self, query: str, principal: Principal) -> RetrievalResult:
        return self._retriever.retrieve(query, principal)

    def ask(self, query: str, principal: Principal) -> AskResult:
        result = self._retriever.retrieve(query, principal)
        answer = self._answerer.answer(result)
        return AskResult(answer=answer, retrieval=result)
