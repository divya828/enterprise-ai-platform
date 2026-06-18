"""Tests for the mock connectors and the synthetic corpus integrity."""

from __future__ import annotations

from datetime import UTC, datetime

from eaip.ingestion import SourceType, connectors_from_corpus, load_corpus
from eaip.ingestion.connectors import CorpusConnector
from tests.conftest import CORPUS_PATH, make_doc


def test_corpus_loads_with_acls_and_sources():
    docs = load_corpus(CORPUS_PATH)
    assert len(docs) == 30
    # Every source is represented.
    sources = {d.source for d in docs}
    assert sources == {SourceType.CONFLUENCE, SourceType.JIRA, SourceType.DATABASE}
    # Some docs are restricted (not visible to 'everyone') — needed for leakage tests.
    restricted = [d for d in docs if "everyone" not in d.acl.allowed_groups]
    assert len(restricted) >= 5


def test_corpus_contains_planted_injection_doc():
    docs = load_corpus(CORPUS_PATH)
    injected = [d for d in docs if d.extra.get("planted_injection") == "true"]
    assert len(injected) == 1
    assert "IGNORE ALL PREVIOUS INSTRUCTIONS" in injected[0].text


def test_user_restricted_acl_parsed():
    docs = {d.doc_id: d for d in load_corpus(CORPUS_PATH)}
    falcon = docs["CONF-9"]
    assert falcon.acl.allowed_groups == frozenset()
    assert "ceo@acme.test" in falcon.acl.allowed_users
    # ACL semantics: the CEO can read it; a random engineer cannot.
    assert falcon.acl.permits(user="ceo@acme.test", groups=frozenset())
    assert not falcon.acl.permits(user="eng@acme.test", groups={"engineering"})


def test_connectors_split_by_source():
    conns = connectors_from_corpus(CORPUS_PATH)
    assert set(conns) == {SourceType.CONFLUENCE, SourceType.JIRA, SourceType.DATABASE}
    # Each connector only sees its own source's docs.
    for source, conn in conns.items():
        assert all(d.source is source for d in conn.fetch_all())


def test_fetch_since_is_incremental():
    conn = CorpusConnector(
        SourceType.CONFLUENCE,
        [make_doc("A", day=1), make_doc("B", day=10), make_doc("C", day=20)],
    )
    watermark = datetime(2026, 1, 9, 12, 0, tzinfo=UTC)
    changed = conn.fetch_since(watermark)
    ids = {d.doc_id for d in changed}
    assert ids == {"B", "C"}  # only docs strictly after the watermark
    assert conn.fetch_since(None) == conn.fetch_all()


def test_current_ids_reflects_deletions():
    conn = CorpusConnector(SourceType.JIRA, [make_doc("J1", source=SourceType.JIRA)])
    assert conn.current_ids() == {"J1"}
    conn.delete("J1")
    assert conn.current_ids() == set()
