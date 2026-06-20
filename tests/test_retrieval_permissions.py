"""Permission-aware retrieval — the Phase 2 security deliverables.

Proves, against the real synthetic corpus:
  1. A user cannot retrieve a document their ACL excludes (no leakage), via the
     full hybrid pipeline AND via each arm independently (dense + sparse).
  2. When a document's ACL changes (revocation), retrieval reflects it after
     re-index — no stale "who can see what".
"""

from __future__ import annotations

from eaip.index import IngestionPipeline, access_filter
from eaip.ingestion import SourceType
from eaip.retrieval import DenseRetriever, Principal, SparseRetriever
from tests.conftest import make_doc

# CONF-7 = "FY26 Revenue Forecast", ACL: groups=['finance'].
_FINANCE_DOC = "CONF-7"
_FINANCE_QUERY = "FY26 revenue forecast annual recurring revenue"


def _doc_ids(scored):
    return {sc.chunk.doc_id for sc in scored}


def test_unauthorized_user_cannot_retrieve_restricted_doc(retriever):
    """An 'everyone' user must never see a finance-only document."""
    outsider = Principal.of("intern@acme.test", ["everyone"])
    result = retriever.retrieve(_FINANCE_QUERY, outsider)
    assert _FINANCE_DOC not in _doc_ids(result.chunks)
    # And it isn't even in the pre-rerank fused shortlist (filtered before ranking).
    assert _FINANCE_DOC not in {f.scored.chunk.doc_id for f in result.fused}


def test_authorized_user_can_retrieve_restricted_doc(retriever):
    """A finance user SHOULD see the finance document (filter isn't over-broad)."""
    insider = Principal.of("cfo@acme.test", ["finance"])
    result = retriever.retrieve(_FINANCE_QUERY, insider)
    assert _FINANCE_DOC in _doc_ids(result.chunks)


def test_no_leakage_through_dense_arm(ingested_index, embedder):
    dense = DenseRetriever(ingested_index, embedder)
    acl = access_filter(user="intern@acme.test", groups=["everyone"])
    hits = dense.search(_FINANCE_QUERY, limit=30, acl_filter=acl)
    assert _FINANCE_DOC not in _doc_ids(hits)


def test_no_leakage_through_sparse_arm(ingested_index):
    """BM25 builds its corpus from permitted chunks only — forbidden doc never enters."""
    sparse = SparseRetriever(ingested_index)
    acl = access_filter(user="intern@acme.test", groups=["everyone"])
    hits = sparse.search(_FINANCE_QUERY, limit=30, acl_filter=acl)
    assert _FINANCE_DOC not in _doc_ids(hits)


def test_user_restricted_doc_visible_only_to_named_user(retriever):
    """CONF-9 (Project Falcon) is user-restricted to the CEO/CFO, no groups."""
    falcon_q = "Project Falcon acquisition price"
    ceo = Principal.of("ceo@acme.test", [])
    other = Principal.of("eng@acme.test", ["engineering", "everyone"])
    assert "CONF-9" in _doc_ids(retriever.retrieve(falcon_q, ceo).chunks)
    assert "CONF-9" not in _doc_ids(retriever.retrieve(falcon_q, other).chunks)


def test_revocation_is_reflected_after_reindex(index, embedder):
    """Changing a doc's ACL and re-indexing removes it from an ex-viewer's results.

    Note the watermark caveat this exercises: incremental re-indexing keys on
    ``last_modified``, so an ACL change is only picked up if the edit advances the
    timestamp (as a real edit does). We use a dedicated single-doc connector so
    the watermark is governed entirely by this document's own timestamps.
    """
    from eaip.ingestion import CorpusConnector
    from eaip.retrieval import DenseRetriever, HybridRetriever, SparseRetriever
    from eaip.retrieval.reranker import LexicalReranker

    text = "\n\n".join(f"sensitive quarterly figure {i} revenue" for i in range(8))

    # Start: a doc readable by 'everyone' (day 1).
    conn = CorpusConnector(
        SourceType.CONFLUENCE,
        [make_doc("REV-1", title="Quarterly Numbers", text=text, day=1, groups=["everyone"])],
    )
    pipe = IngestionPipeline(index, embedder)
    pipe.sync(conn)

    retr = HybridRetriever(
        DenseRetriever(index, embedder),
        SparseRetriever(index),
        LexicalReranker(),
        shortlist=20,
        top_k=10,
    )
    user = Principal.of("contractor@acme.test", ["everyone"])
    before = _doc_ids(retr.retrieve("sensitive quarterly revenue figure", user).chunks)
    assert "REV-1" in before  # visible while ACL includes 'everyone'

    # Revoke: restrict to finance only; the edit advances last_modified (day 2 >
    # the day-1 watermark), so the incremental sync re-processes it.
    conn.upsert(make_doc("REV-1", title="Quarterly Numbers", text=text, day=2, groups=["finance"]))
    report = pipe.sync(conn)
    assert report.upserted_docs == 1  # the ACL change WAS re-indexed

    after = _doc_ids(retr.retrieve("sensitive quarterly revenue figure", user).chunks)
    assert "REV-1" not in after  # revocation reflected immediately after re-index
    # ...but a finance user still sees it.
    fin = Principal.of("cfo@acme.test", ["finance"])
    assert "REV-1" in _doc_ids(retr.retrieve("sensitive quarterly revenue figure", fin).chunks)
