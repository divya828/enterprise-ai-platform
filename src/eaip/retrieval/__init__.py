"""Retrieval (the RAG core): hybrid dense+sparse, RRF fusion, rerank, grounding."""

from eaip.retrieval.answerer import Answer, Citation, GroundedAnswerer
from eaip.retrieval.dense import DenseRetriever
from eaip.retrieval.fusion import FusedChunk, reciprocal_rank_fusion
from eaip.retrieval.pipeline import HybridRetriever, Principal, RetrievalResult
from eaip.retrieval.reranker import Reranker
from eaip.retrieval.reranker_factory import get_reranker
from eaip.retrieval.sparse import SparseRetriever

__all__ = [
    "Answer",
    "Citation",
    "GroundedAnswerer",
    "DenseRetriever",
    "SparseRetriever",
    "FusedChunk",
    "reciprocal_rank_fusion",
    "HybridRetriever",
    "Principal",
    "RetrievalResult",
    "Reranker",
    "get_reranker",
]
