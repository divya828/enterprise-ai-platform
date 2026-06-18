"""Mock enterprise connectors.

A connector's job in a real system is to talk to a source of record (Confluence,
Jira, a database) and yield documents with their metadata and access-control
lists. We mock that with connectors that read from the synthetic corpus JSON, but
we keep the *interface* realistic so the ingestion pipeline is written against
the contract a real connector would satisfy.

Two methods matter:

* ``fetch_all()`` — full sync (first ingest, or a forced rebuild).
* ``fetch_since(watermark)`` — incremental sync: return only documents modified
  strictly after ``watermark``. This is the freshness edge case — without it,
  every sync would re-process the entire corpus. The pipeline records the newest
  ``last_modified`` it has seen as the watermark and asks the connector for only
  what changed.

Connectors also report *deletions*. A real connector detects that a document that
existed last sync is gone now (a "tombstone"); we model this by comparing a set
of currently-present ids against a set of previously-seen ids. The pipeline turns
those tombstones into chunk deletions so a removed source document becomes
unretrievable.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Protocol

from eaip.ingestion.models import ACL, Document, SourceType


class Connector(Protocol):
    """The contract every source connector satisfies."""

    source: SourceType

    def fetch_all(self) -> list[Document]:
        """Return all documents currently in the source."""
        ...

    def fetch_since(self, watermark: datetime | None) -> list[Document]:
        """Return documents modified strictly after ``watermark`` (all if None)."""
        ...

    def current_ids(self) -> set[str]:
        """Return the ids of all documents currently in the source.

        Used by the pipeline to detect deletions: ids that were indexed before
        but are absent now are tombstoned.
        """
        ...


def _doc_from_record(record: dict) -> Document:
    """Parse one corpus JSON record into a :class:`Document`."""
    return Document(
        doc_id=record["doc_id"],
        source=SourceType(record["source"]),
        title=record["title"],
        text=record["text"],
        last_modified=datetime.fromisoformat(record["last_modified"]),
        acl=ACL.of(
            groups=record["acl"].get("groups"),
            users=record["acl"].get("users"),
        ),
        extra=dict(record.get("extra", {})),
    )


class CorpusConnector:
    """A connector backed by the synthetic corpus JSON, filtered to one source.

    In-memory and mutable so tests can simulate edits and deletions (via
    :meth:`upsert` / :meth:`delete`) without touching disk — exactly the moves a
    real source would make between syncs.
    """

    def __init__(self, source: SourceType, documents: Iterable[Document]) -> None:
        self.source = source
        self._docs: dict[str, Document] = {d.doc_id: d for d in documents if d.source == source}

    # --- connector contract ---
    def fetch_all(self) -> list[Document]:
        return list(self._docs.values())

    def fetch_since(self, watermark: datetime | None) -> list[Document]:
        if watermark is None:
            return self.fetch_all()
        return [d for d in self._docs.values() if d.last_modified > watermark]

    def current_ids(self) -> set[str]:
        return set(self._docs)

    # --- test/demo affordances: mutate the source between syncs ---
    def upsert(self, document: Document) -> None:
        """Add or replace a document (simulates an edit in the source)."""
        if document.source != self.source:
            raise ValueError(f"{document.doc_id} is not a {self.source} document")
        self._docs[document.doc_id] = document

    def delete(self, doc_id: str) -> None:
        """Remove a document (simulates a deletion in the source)."""
        self._docs.pop(doc_id, None)


def load_corpus(path: str | Path) -> list[Document]:
    """Load every document from a corpus JSON file (all sources)."""
    records = json.loads(Path(path).read_text())
    return [_doc_from_record(r) for r in records]


def connectors_from_corpus(path: str | Path) -> dict[SourceType, CorpusConnector]:
    """Build one connector per source from a corpus JSON file.

    Returns a mapping so the pipeline can iterate sources independently — each
    source keeps its own watermark, mirroring how real syncs are per-source.
    """
    documents = load_corpus(path)
    return {source: CorpusConnector(source, documents) for source in SourceType}
