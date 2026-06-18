# Learnings

A study guide written for a smart engineer who hasn't built an enterprise AI
platform before. Each phase appends 5–10 lines: the core concept, the edge cases
involved, and why they matter. Design decisions with real tradeoffs are recorded
inline so the reasoning isn't lost.

---

## Phase 0 — Scaffolding

**Core concept: the provider abstraction (avoiding LLM vendor lock-in).** A
platform shouldn't hard-code one LLM vendor. We define a tiny `LLMProvider`
protocol — `complete(messages) -> Completion` over vendor-neutral `Message` and
`Completion` types — and every backend (stub, Ollama, Anthropic, OpenAI)
implements it. The rest of the codebase imports only the protocol and the neutral
types, never an SDK. Switching `EAIP_LLM_PROVIDER` swaps the backend with zero
changes elsewhere. This is the *strategy pattern* applied to a vendor boundary,
and it's a platform concept in its own right, not just plumbing.

**Why a *scripted* stub is the default (the key design decision).** Tests and CI
must be deterministic and must never call a paid, non-deterministic API. So the
default provider is an offline **stub** that replays a preloaded queue of
responses — and crucially, each queued response can be plain text *or* a
simulated tool call. That lets us exercise the full Phase 3 agent loop (plan →
call tool → observe → answer) deterministically with no model installed. With an
empty queue it falls back to a canned echo, which is enough to prove the
abstraction is wired end to end. (Decision: we changed the spec's "Ollama
default" to "stub default" because Ollama isn't installed here and a stub is more
faithful to "runs on a laptop with no paid service" — and it makes CI hermetic.
Ollama/Anthropic/OpenAI remain opt-in via env.)

**Edge cases handled in Phase 0.** (1) *Missing credentials*: selecting a real
backend without its API key raises a single, actionable `ProviderError` instead
of a raw SDK stack trace — the runtime can distinguish provider misconfiguration
from application bugs. (2) *Missing SDKs*: real backends lazy-import their SDKs
inside their constructors, so importing `eaip.providers` never requires
`anthropic`/`openai` to be installed — Phase 0's footprint stays tiny and CI is
fast. (3) *Vendor parameter quirks*: the Opus 4.x / Fable family rejects
`temperature` (HTTP 400), so the Anthropic backend drops sampling params for
those models — exactly the kind of detail the abstraction exists to hide from
callers. Why these matter: an enterprise platform is judged on how it *fails*
(clear errors, graceful degradation), not just on the happy path.

**Config as a first-class layer.** All configuration flows through one typed
`Settings` object (pydantic-settings), giving a single documented place for every
knob, validation at startup (fail fast on a bad value), and a clear precedence
(env > .env > defaults). Decision/tradeoff: we centralize config now so that in
Phase 4 the same surface can be backed by a per-tenant config table without
touching call sites.

**App factory + dependency injection.** The FastAPI app is built by a
`create_app()` factory and gets its provider via `Depends`, so tests inject a
scripted stub through `app.dependency_overrides` with no global state. Small
thing now; it's what keeps the HTTP surface testable as it grows.

**Verifying current APIs (the "don't trust memory" rule).** Library APIs drift,
so versions were checked against PyPI on **2026-06-16** before pinning. Notable
finding: **LangGraph is at 1.x (1.2.5)** and langchain-core at 1.4.x — both past
1.0, so Phase 3 will use the stable `StateGraph` + checkpointer API rather than
any pre-1.0 shape I might half-remember. Pins (with `<next-major` upper bounds):
fastapi 0.137, pydantic 2.13 / pydantic-settings 2.14, qdrant-client 1.18,
sentence-transformers 5.5, ragas 0.4, anthropic 0.109, openai 2.41. Heavy,
phase-specific deps live in optional extras so they install only when needed.

**Production note (storage swap, deferred).** Phase 0 has no persistence yet, but
the plan is SQLite-by-default behind an abstracted storage layer. In production
the swap to Postgres would change the connection/driver and migration tooling,
not the call sites — this is the same "depend on an interface, not an
implementation" principle as the provider abstraction, applied to storage.

---

## Phase 1 — Ingestion & connectors

**Core concept: the ingestion pipeline (connectors → chunker → embedder →
index).** RAG starts long before retrieval. A *connector* pulls documents from a
source of record with their metadata and access-control list; a *chunker* splits
each document into retrievable units; an *embedder* turns chunk text into a
vector; the *index* stores vectors + payload in Qdrant. The pipeline wires these
into one `sync()` call. Keeping each stage a separate, single-responsibility unit
is what makes the edge cases tractable — each one lives in exactly one place.

**The ACL must travel with the data (the security thread).** The single most
important invariant in Phase 1: a document's ACL is copied onto *every chunk* and
stored in *every Qdrant point's payload*. If it didn't survive chunking and
indexing, permission-aware retrieval (Phase 2) would be impossible — you'd have
no per-chunk basis to filter on. So an "ACL dropped during chunking" bug is
really a data-leak bug. We model a deliberately small, **fail-closed** ACL: empty
lists grant access to no one; a user reads a chunk if named directly OR in any
allowed group. Tested end to end: index a finance-only doc, confirm the ACL is
intact on the retrieved chunk.

**Edge case — incremental re-indexing (the watermark).** Re-ingesting an entire
corpus on every sync doesn't scale (imagine the spec's "15M documents"). The
pipeline keeps a per-source *watermark* = the newest `last_modified` it has
indexed, and asks the connector for only documents changed after it
(`fetch_since`). Why per-source: real syncs run independently per source, each
with its own cadence; one source's sync must not reset another's progress
(tested). Why it matters: the second `scripts/ingest.py` run upserts **0**
documents — proven across *processes* because `SyncState` is persisted to JSON.

**Edge case — re-index updates, never duplicates (idempotency).** Chunk ids are
deterministic (`{doc_id}::{ordinal}`), and the Qdrant point id is a stable
`uuid5` of the chunk id. So re-processing a document *overwrites* its points
rather than appending new ones — the chunk count stays flat. A subtle sub-case:
if an edited document is now *shorter* (fewer chunks), the old extra chunks would
linger. We handle it by deleting the document's chunks before re-upserting, so no
stale chunk survives (tested: shrink a 120-token doc to 2 tokens → exactly 1
chunk remains).

**Edge case — deletions / tombstones.** Deleting a source document must make it
unretrievable, including all its derived chunks. The pipeline remembers the doc
ids it indexed per source and, each sync, diffs them against the connector's
*current* ids; any id that vanished is deleted by a `doc_id` payload filter
(removing every chunk, no orphans). Tested: a deleted doc returns zero hits.

**Decision/tradeoff — offline hashing embedder as the default.** Embedding the
corpus needs a model; BGE-small is a ~130MB download. Mirroring the stub-LLM
choice, the default embedder is a **feature-hashing** embedder: deterministic,
offline, dependency-free, so the whole ingestion+retrieval pipeline runs in CI in
milliseconds. It is *not* semantically smart (no synonyms/paraphrase) — it exists
to test the machinery (dense search, ACL filtering, dedup) without conflating it
with model quality. Real semantic quality comes from `EAIP_EMBEDDER=bge`. This
separation — test the plumbing deterministically, swap in a real model for
quality — is a recurring pattern in this build.

**Decision/tradeoff — Qdrant point ids.** Qdrant requires unsigned-int or UUID
ids, but our human-readable `chunk_id` is a string. Rather than maintain a
side-table, we derive the point id as `uuid5(fixed_namespace, chunk_id)` and keep
`chunk_id` in the payload — deterministic, stable across runs, and idempotent by
construction.

**Decision/tradeoff — structure-aware chunking.** We split on markdown structure
(headings → paragraphs) and pack into size-bounded windows with overlap, rather
than a blind fixed window. Boundaries fall at natural seams and each chunk is
prefixed with its section heading (so the heading's keywords are searchable).
Overlap repeats a tail of one chunk at the head of the next so a fact straddling
a boundary still matches — the cost is some duplication, made explicit via the
`max_tokens`/`overlap_tokens` knobs. "Size" is measured in whitespace tokens (a
model-agnostic proxy); a production system would use the embedder's tokenizer.

**Production note.** `SyncState` is persisted to JSON here; in production it would
be a row in the abstracted storage layer (SQLite/Postgres) — same shape. Qdrant
runs embedded/on-disk (no Docker) for the app and `:memory:` for tests.
