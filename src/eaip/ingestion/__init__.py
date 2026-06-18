"""Ingestion: connectors, domain models, and the structure-aware chunker."""

from eaip.ingestion.chunker import ChunkConfig, chunk_document
from eaip.ingestion.connectors import (
    CorpusConnector,
    connectors_from_corpus,
    load_corpus,
)
from eaip.ingestion.models import ACL, Chunk, Document, SourceType

__all__ = [
    "ACL",
    "Chunk",
    "Document",
    "SourceType",
    "ChunkConfig",
    "chunk_document",
    "CorpusConnector",
    "connectors_from_corpus",
    "load_corpus",
]
