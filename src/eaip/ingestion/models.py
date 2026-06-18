"""Core domain models for ingestion: documents, ACLs, and chunks.

These types are the contract that ties ingestion to every later phase. The most
important design choice here is that the **access-control list (ACL) travels with
the data** — it is attached to the source document and then *copied onto every
chunk*. Retrieval (Phase 2) filters candidates by the requesting user's identity
against the chunk's ACL, so if the ACL didn't survive chunking, permission-aware
retrieval would be impossible. This is why an ingestion bug ("ACL dropped during
chunking") is really a security bug.

We model a deliberately small ACL: a set of allowed groups and a set of allowed
users. A user may read a document if they are in ``allowed_users`` OR belong to
any group in ``allowed_groups``. An empty ACL means "no one" (fail closed), which
is the safe default for an enterprise system.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum


class SourceType(StrEnum):
    """Where a document originated. Mirrors the mock enterprise connectors."""

    CONFLUENCE = "confluence"
    JIRA = "jira"
    DATABASE = "database"


@dataclass(frozen=True)
class ACL:
    """Who may read a document/chunk.

    Fail-closed semantics: an empty ACL grants access to no one. Membership is
    OR across users and groups — being named directly or being in any allowed
    group is sufficient.
    """

    allowed_groups: frozenset[str] = frozenset()
    allowed_users: frozenset[str] = frozenset()

    def permits(self, *, user: str, groups: frozenset[str] | set[str]) -> bool:
        """Return True if ``user`` (a member of ``groups``) may read this."""
        if user in self.allowed_users:
            return True
        return bool(self.allowed_groups & frozenset(groups))

    @classmethod
    def of(
        cls,
        groups: list[str] | None = None,
        users: list[str] | None = None,
    ) -> ACL:
        """Convenience constructor from plain lists (used by the corpus loader)."""
        return cls(
            allowed_groups=frozenset(groups or ()),
            allowed_users=frozenset(users or ()),
        )


@dataclass(frozen=True)
class Document:
    """A source document loaded from a (mock) connector.

    ``doc_id`` is stable across syncs so re-ingestion updates rather than
    duplicates. ``last_modified`` drives incremental re-indexing (the watermark
    edge case): only documents modified after the last sync are re-processed.
    """

    doc_id: str
    source: SourceType
    title: str
    text: str
    last_modified: datetime
    acl: ACL
    # Free-form extra metadata (e.g. Jira status, author) carried through to chunks.
    extra: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit derived from a document.

    Carries a copy of the document's ACL and identifying metadata so that
    retrieval can filter and cite without re-fetching the source. ``chunk_id`` is
    deterministic (``{doc_id}::{ordinal}``) so re-chunking the same document
    upserts the same point ids — this is what makes re-indexing idempotent rather
    than duplicative.
    """

    chunk_id: str
    doc_id: str
    source: SourceType
    title: str
    text: str
    ordinal: int
    last_modified: datetime
    acl: ACL
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        """Human-readable provenance string for grounded answers (Phase 2)."""
        return f"{self.source}:{self.doc_id} — {self.title} (#{self.ordinal})"

    def with_text(self, text: str) -> Chunk:
        """Return a copy with replaced text (used by the chunker)."""
        return replace(self, text=text)
