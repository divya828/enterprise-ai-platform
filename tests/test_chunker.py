"""Tests for the structure-aware chunker."""

import pytest

from eaip.ingestion import ChunkConfig, chunk_document
from eaip.ingestion.models import ACL, Document, SourceType
from tests.conftest import make_doc


def test_acl_and_metadata_survive_on_every_chunk():
    """The non-negotiable invariant: every chunk carries the document's ACL."""
    acl = ACL.of(groups=["finance"], users=["cfo@acme.test"])
    long_text = "\n\n".join(f"Paragraph {i} with several words to fill space." for i in range(20))
    doc = Document(
        doc_id="D1",
        source=SourceType.CONFLUENCE,
        title="Sensitive Doc",
        text=long_text,
        last_modified=make_doc("x").last_modified,
        acl=acl,
        extra={"classification": "restricted"},
    )
    chunks = chunk_document(doc, ChunkConfig(max_tokens=30, overlap_tokens=5))

    assert len(chunks) > 1  # actually split into multiple chunks
    for c in chunks:
        assert c.acl == acl
        assert c.doc_id == "D1"
        assert c.source is SourceType.CONFLUENCE
        assert c.title == "Sensitive Doc"
        assert c.extra["classification"] == "restricted"


def test_chunk_ids_are_deterministic():
    doc = make_doc("D2", text="\n\n".join(f"para {i} words words words" for i in range(10)))
    cfg = ChunkConfig(max_tokens=20, overlap_tokens=4)
    a = [c.chunk_id for c in chunk_document(doc, cfg)]
    b = [c.chunk_id for c in chunk_document(doc, cfg)]
    assert a == b
    assert a[0] == "D2::0"


def test_headings_are_attached_to_chunk_text():
    doc = make_doc(
        "D3",
        text="# Title\n\nintro text\n\n## VPN Troubleshooting\n\nrestart the client",
    )
    chunks = chunk_document(doc, ChunkConfig(max_tokens=200, overlap_tokens=0))
    joined = " ".join(c.text for c in chunks)
    assert "VPN Troubleshooting" in joined


def test_respects_max_tokens():
    doc = make_doc("D4", text=" ".join(["word"] * 500))
    chunks = chunk_document(doc, ChunkConfig(max_tokens=50, overlap_tokens=10))
    for c in chunks:
        assert len(c.text.split()) <= 50


def test_empty_document_still_yields_one_chunk():
    doc = make_doc("D5", title="Only A Title", text="   ")
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].acl == doc.acl


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        ChunkConfig(max_tokens=10, overlap_tokens=10)  # overlap must be < max
    with pytest.raises(ValueError):
        ChunkConfig(max_tokens=0)
