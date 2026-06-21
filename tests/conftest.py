"""Shared pytest fixtures for ingestion + retrieval tests.

Everything here is offline and in-memory: the hashing embedder (no download) and
an in-memory Qdrant index (``:memory:``), so the whole suite runs fast and
hermetically in CI.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from eaip.embeddings.hashing import HashingEmbedder
from eaip.index import ChunkIndex, IngestionPipeline
from eaip.ingestion import ACL, CorpusConnector, Document, SourceType, connectors_from_corpus

CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "corpus" / "documents.json"

_DIM = 128


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=_DIM)


@pytest.fixture
def index(embedder: HashingEmbedder) -> ChunkIndex:
    """A fresh in-memory chunk index per test."""
    return ChunkIndex.open(path=":memory:", collection="test_chunks", dim=embedder.dim)


@pytest.fixture
def pipeline(index: ChunkIndex, embedder: HashingEmbedder) -> IngestionPipeline:
    return IngestionPipeline(index, embedder)


@pytest.fixture
def corpus_connectors() -> dict[SourceType, CorpusConnector]:
    """Connectors over the real synthetic corpus, one per source."""
    return connectors_from_corpus(CORPUS_PATH)


@pytest.fixture
def ingested_index(index: ChunkIndex, embedder: HashingEmbedder) -> ChunkIndex:
    """An in-memory index with the full synthetic corpus ingested."""
    pipe = IngestionPipeline(index, embedder)
    pipe.sync_all(connectors_from_corpus(CORPUS_PATH))
    return index


@pytest.fixture
def retriever(ingested_index: ChunkIndex, embedder: HashingEmbedder):
    """A HybridRetriever over the ingested corpus with the offline reranker.

    Uses a SingleIndexResolver so every tenant resolves to the one in-memory
    index — keeps single-tenant retrieval tests simple while exercising the
    resolver seam.
    """
    from eaip.index.resolver import SingleIndexResolver
    from eaip.retrieval import HybridRetriever
    from eaip.retrieval.reranker import LexicalReranker

    return HybridRetriever(
        SingleIndexResolver(ingested_index),
        embedder,
        LexicalReranker(),
        shortlist=20,
        top_k=5,
    )


@pytest.fixture
def agent_runner(retriever):
    """An AgentRunner over the ingested corpus with an in-memory checkpointer.

    Returns ``(runner, side_effect_log, episodic_store)`` so tests can assert on
    answers, side effects (idempotency), and recorded episodes.
    """
    from langgraph.checkpoint.memory import MemorySaver

    from eaip.orchestration import AgentNodes, AgentRunner, build_default_tools
    from eaip.providers.stub import StubProvider
    from eaip.retrieval.answerer import GroundedAnswerer
    from eaip.storage import InMemoryStateStore

    answerer = GroundedAnswerer(StubProvider(), min_score=0.0)
    nodes = AgentNodes(StubProvider(), retriever, answerer, max_revisions=2)
    tools, log = build_default_tools()
    store = InMemoryStateStore()
    runner = AgentRunner(nodes, tools, MemorySaver(), max_revisions=2, episodic=store)
    return runner, log, store


def make_doc(
    doc_id: str,
    *,
    source: SourceType = SourceType.CONFLUENCE,
    title: str = "Test Doc",
    text: str = "Some indexable body text about the platform.",
    day: int = 1,
    groups: list[str] | None = None,
    users: list[str] | None = None,
) -> Document:
    """Build a Document with sensible defaults for tests."""
    return Document(
        doc_id=doc_id,
        source=source,
        title=title,
        text=text,
        last_modified=datetime(2026, 1, day, 12, 0, tzinfo=UTC),
        acl=ACL.of(groups=groups or ["everyone"], users=users),
    )
