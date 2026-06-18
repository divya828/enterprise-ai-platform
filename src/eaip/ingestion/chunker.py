"""Structure-aware chunking.

Chunking turns a document into retrievable units. Two competing pressures shape
it: chunks must be small enough that an embedding captures a focused topic and
that several fit in a prompt, but large enough to carry enough context to answer
a question. We split on document *structure* first (markdown headings, then
paragraphs) so chunk boundaries fall at natural seams, then pack those pieces
into size-bounded chunks with a configurable overlap.

Why overlap: a fact can straddle a boundary. Repeating a tail of the previous
chunk at the head of the next reduces the chance that a query matches neither
chunk because the answer was split across them. The cost is some duplication;
the size/overlap knobs make the tradeoff explicit.

The non-negotiable invariant: **every chunk inherits the document's ACL and
identifying metadata.** Chunk ids are deterministic (``{doc_id}::{ordinal}``) so
re-chunking the same document produces the same ids — the basis for idempotent
re-indexing.

Note on "size": we measure size in whitespace-delimited tokens (a cheap, model-
agnostic proxy). A real system would use the embedding model's tokenizer; word
count is close enough to teach the concept without coupling the chunker to a
specific tokenizer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from eaip.ingestion.models import Chunk, Document

# A markdown heading line, e.g. "## Troubleshooting".
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class ChunkConfig:
    """Chunking knobs.

    ``max_tokens`` caps chunk size; ``overlap_tokens`` is how many trailing
    tokens of one chunk are prepended to the next. ``overlap_tokens`` must be
    smaller than ``max_tokens`` or chunking could fail to make progress.
    """

    max_tokens: int = 120
    overlap_tokens: int = 20

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0 <= self.overlap_tokens < self.max_tokens:
            raise ValueError("overlap_tokens must be in [0, max_tokens)")


@dataclass(frozen=True)
class _Section:
    """A heading-delimited block of text (the structural unit)."""

    heading: str | None
    body: str


def _split_sections(text: str) -> list[_Section]:
    """Split markdown into heading-delimited sections.

    Text before the first heading becomes a section with ``heading=None``. This
    keeps a chunk anchored to the section title it came from, which improves both
    retrieval relevance and the readability of citations.
    """
    sections: list[_Section] = []
    current_heading: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body or current_heading:
            sections.append(_Section(heading=current_heading, body=body))

    for line in text.splitlines():
        m = _HEADING_RE.match(line.strip())
        if m:
            flush()
            current_heading = m.group(2).strip()
            buffer = []
        else:
            buffer.append(line)
    flush()
    return sections


def _paragraphs(body: str) -> list[str]:
    """Split a section body into paragraphs on blank lines."""
    return [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]


def _tokens(text: str) -> list[str]:
    return text.split()


def _pack(pieces: list[str], cfg: ChunkConfig) -> list[str]:
    """Greedily pack text pieces into size-bounded windows with overlap.

    Each piece (a paragraph, prefixed with its heading) is added to the current
    window until adding the next would exceed ``max_tokens``; then the window is
    emitted and a new one is seeded with the last ``overlap_tokens`` tokens of
    the emitted window. A single piece larger than ``max_tokens`` is hard-split.
    """
    windows: list[str] = []
    current: list[str] = []  # token list for the in-progress window

    def emit() -> None:
        if current:
            windows.append(" ".join(current))

    for piece in pieces:
        for token in _tokens(piece):
            if len(current) >= cfg.max_tokens:
                emit()
                current = current[-cfg.overlap_tokens :] if cfg.overlap_tokens else []
            current.append(token)
        # Paragraph boundary: if the window is already near full, close it so the
        # next paragraph starts cleanly (keeps chunks topically coherent).
        if len(current) >= cfg.max_tokens - cfg.overlap_tokens:
            emit()
            current = current[-cfg.overlap_tokens :] if cfg.overlap_tokens else []
    emit()
    return [w for w in windows if w.strip()]


def chunk_document(document: Document, cfg: ChunkConfig | None = None) -> list[Chunk]:
    """Split ``document`` into ACL-preserving, size-bounded chunks.

    Each paragraph is prefixed with its section heading (when present) so the
    heading's keywords land in the chunk text. Chunk ids are deterministic.
    """
    cfg = cfg or ChunkConfig()
    sections = _split_sections(document.text)

    pieces: list[str] = []
    for section in sections:
        prefix = f"{section.heading}\n" if section.heading else ""
        body_paras = _paragraphs(section.body) or ([""] if section.heading else [])
        for para in body_paras:
            pieces.append(f"{prefix}{para}".strip())

    windows = _pack([p for p in pieces if p], cfg)
    # A document with no extractable text still yields one (empty-safe) chunk so
    # its existence — and ACL — is represented; defensive against odd inputs.
    if not windows:
        windows = [document.text.strip() or document.title]

    return [
        Chunk(
            chunk_id=f"{document.doc_id}::{i}",
            doc_id=document.doc_id,
            source=document.source,
            title=document.title,
            text=window,
            ordinal=i,
            last_modified=document.last_modified,
            acl=document.acl,
            extra=dict(document.extra),
        )
        for i, window in enumerate(windows)
    ]
