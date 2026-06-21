"""Multi-tenancy isolation — the Phase 4 'no data bleed across tenants' deliverable.

Two tenants are ingested into their own collections via the TenantIndexResolver,
then we prove a retrieval scoped to one tenant cannot surface the other tenant's
documents — even with an identical query and identical (everyone) ACLs.
"""

from __future__ import annotations

import pytest

from eaip.embeddings.hashing import HashingEmbedder
from eaip.index import IngestionPipeline
from eaip.index.resolver import TenantIndexResolver
from eaip.ingestion import ACL, CorpusConnector, Document, SourceType
from eaip.platform.tenancy import collection_for_tenant, validate_tenant_id
from eaip.retrieval import HybridRetriever
from eaip.retrieval.reranker import LexicalReranker
from eaip.security import Principal

_DIM = 128


def _doc(doc_id: str, text: str) -> Document:
    from datetime import UTC, datetime

    return Document(
        doc_id=doc_id,
        source=SourceType.CONFLUENCE,
        title=doc_id,
        text=text,
        last_modified=datetime(2026, 1, 1, tzinfo=UTC),
        acl=ACL.of(groups=["everyone"]),
    )


def _ingest(resolver, tenant, docs):
    pipe = IngestionPipeline(resolver.for_tenant(tenant), HashingEmbedder(dim=_DIM))
    pipe.sync(CorpusConnector(SourceType.CONFLUENCE, docs))


def test_collection_naming():
    assert collection_for_tenant("eaip_chunks", "acme") == "eaip_chunks__acme"


def test_invalid_tenant_id_rejected():
    validate_tenant_id("acme-1")  # ok
    for bad in ["Has Space", "UPPER", "a/b", ""]:
        with pytest.raises(ValueError):
            validate_tenant_id(bad)


def test_no_data_bleed_across_tenants():
    embedder = HashingEmbedder(dim=_DIM)
    resolver = TenantIndexResolver(path=":memory:", collection_prefix="eaip_chunks", dim=_DIM)

    # Each tenant ingests a distinctively-worded secret doc.
    _ingest(resolver, "acme", [_doc("ACME-1", "acme rocket fuel formula alpha")])
    _ingest(resolver, "globex", [_doc("GLOBEX-1", "globex widget assembly secret beta")])

    retriever = HybridRetriever(resolver, embedder, LexicalReranker(), shortlist=20, top_k=10)

    # Tenant acme searches for globex's content — must find nothing of globex's.
    acme = Principal.of("u@acme.test", ["everyone"], tenant="acme")
    hits = retriever.retrieve("globex widget assembly secret beta", acme)
    found = {h.chunk.doc_id for h in hits.chunks}
    assert "GLOBEX-1" not in found
    assert found <= {"ACME-1"}  # only its own tenant's docs, if any

    # And globex cannot see acme's content.
    globex = Principal.of("u@globex.test", ["everyone"], tenant="globex")
    hits2 = retriever.retrieve("acme rocket fuel formula alpha", globex)
    assert "ACME-1" not in {h.chunk.doc_id for h in hits2.chunks}


def test_each_tenant_sees_its_own_data():
    embedder = HashingEmbedder(dim=_DIM)
    resolver = TenantIndexResolver(path=":memory:", collection_prefix="eaip_chunks", dim=_DIM)
    _ingest(resolver, "acme", [_doc("ACME-1", "acme onboarding vpn setup guide")])

    retriever = HybridRetriever(resolver, embedder, LexicalReranker(), shortlist=20, top_k=10)
    acme = Principal.of("u@acme.test", ["everyone"], tenant="acme")
    hits = retriever.retrieve("acme onboarding vpn setup guide", acme)
    assert "ACME-1" in {h.chunk.doc_id for h in hits.chunks}
