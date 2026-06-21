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

**The storage-layer abstraction (industry-standard state).** Sync state is
persisted through a `StateStore` interface (`eaip/storage/`), with two backends:
`SqliteStateStore` (the app default) and `InMemoryStateStore` (tests). The
pipeline depends only on the interface — the same payoff as the provider/embedder
abstractions, now applied to durable state. This is the spec's *"abstract the
storage layer so Postgres is a drop-in swap"*: a Postgres backend would be one
new class implementing `StateStore` (swap the driver, `%s` placeholders, a
connection pool), with no change to the pipeline or any call site. Later phases
(audit log, prompt registry, tenant config) add sibling interfaces and tables to
the *same* SQLite database/connection rather than re-inventing persistence.

**Decision/tradeoff — SQLite with normalized tables, not a JSON blob.** State
lives in real relational rows — `sync_watermark(source, watermark)` and
`indexed_doc(source, doc_id)` — so it's queryable (`SELECT doc_id FROM
indexed_doc WHERE source='jira'`) and is the shape an enterprise system actually
uses, rather than a serialized blob in a single cell (which would be barely more
than the JSON file it replaced). `save_state` rewrites both tables inside one
transaction so the stored state is always an exact mirror of memory; the state is
tiny, so a full rewrite is simpler and safer than diffing. Qdrant still runs
embedded/on-disk (no Docker) for the app and `:memory:` for tests.

---

## Phase 2 — Retrieval (the RAG core)

**Core concept: hybrid retrieval → RRF fusion → cross-encoder rerank.** This is
the pipeline that turns a question into grounded evidence. *Dense* retrieval
matches on meaning (vector similarity); *sparse* BM25 matches on exact terms
(keyword overlap weighted by rarity). They fail differently — dense misses a rare
acronym spelled out in the query; sparse misses a paraphrase that shares no words
— so fusing both is more robust than either alone. The fused shortlist is then
*reranked* by a cross-encoder and cut to top-k. Each stage is a separate module
(`dense.py`, `sparse.py`, `fusion.py`, `reranker.py`, `pipeline.py`) so the data
flow is legible and each concept is testable in isolation.

**Why RRF, written out explicitly.** Dense scores live in roughly [-1, 1]; BM25
scores are unbounded and much larger. You cannot add or average them — BM25 would
swamp cosine. Reciprocal Rank Fusion throws the *scores* away and fuses on *rank
position*: each result contributes `1 / (k + rank)` to its document, summed across
the lists it appears in (`k≈60` damps the top ranks). A document near the top of
*both* lists gets the biggest total — agreement between methods is the strongest
relevance signal. We implemented the formula by hand (`fusion.py`) because seeing
it is the point; a unit test asserts that a huge BM25 magnitude and a tiny cosine
score at the same rank contribute *equally*.

**The bi-encoder → cross-encoder shortlist pattern (the core efficiency trade).**
Dense and sparse are *bi-encoders*: query and document are embedded
independently, so retrieval is a fast nearest-neighbour lookup but the model never
sees them together. A *cross-encoder* scores the (query, document) pair jointly —
far more accurate, but it must run once per candidate, too slow over a whole
corpus. Resolution: retrieve a cheap top-N shortlist, rerank only that shortlist,
keep top-k. We measure and return per-stage latency (`timings_ms`) so the cost of
reranking is visible — that's the whole reason it runs on N≈20, not the full
index. A test asserts the reranker only ever sees the shortlist and that
`rerank_ms` is reported.

**Edge case — permission-aware retrieval, applied to BOTH arms before ranking.**
The single permission gate is one `access_filter(user, groups)` computed from the
principal and passed to *both* the dense search and the BM25 corpus build. Dense
filtering happens inside Qdrant's ANN (candidates constrained before scoring); for
sparse, the BM25 corpus is built from *only the permitted chunks* (pulled via
`scroll_chunks` with the same filter), so a forbidden chunk never enters the
corpus, never influences IDF, and can never be returned. This is *filter-before*,
not *filter-after* — the safe posture. Tested end to end and through each arm
independently: an `everyone` user can never retrieve a `finance`-only doc, and a
user-restricted doc (Project Falcon) is visible only to the named users.

**Edge case — permission freshness / revocation.** Because the ACL is stored on
each chunk's payload and read at query time, changing a doc's ACL takes effect as
soon as the doc is re-indexed — there's no separate "who can see what" structure
to fall out of sync. The revocation test makes a doc finance-only and confirms an
ex-viewer immediately loses access while a finance user keeps it. **A real bug
this surfaced (worth the scar):** the first version of the test set the
revocation's `last_modified` *below* the source's watermark, so the incremental
sync skipped it and the ACL never updated. The lesson is genuine, not a test
artifact: **a content watermark only re-indexes docs whose timestamp advances**,
so an ACL change must bump `last_modified` (as a real edit does) or it will be
missed. Documented in the test and noted here as a known watermark caveat.

**Edge case — low confidence → "I don't know" (abstention).** Before calling the
LLM at all, the answerer checks the top reranked score against a configurable
threshold (and that any chunk survived ACL filtering). Below it, we return a fixed
"I don't know" and never call the model — so weak/empty evidence can't be spun
into a hallucination, and the answer-vs-abstain decision is deterministic and
testable without a real LLM. A non-finance user asking a finance question retrieves
no permitted evidence → abstains, which is both the safe and the honest outcome.

**Grounding + the seed of Phase 6.** The prompt presents each chunk with a numbered
label and asks the model to cite the labels it used; we return those as structured
`Citation`s (provenance you can trace). The retrieved context is fenced in a
`CONTEXT` block and the system prompt says to treat it as *data, never as
instructions* — the baseline separation that Phase 6's prompt-injection defenses
build on. A test asserts the fencing and the instruction are present.

**Decision/tradeoff — offline lexical reranker default.** Same pattern as the
embedder and LLM provider: the default reranker is a deterministic lexical-overlap
scorer (`EAIP_RERANKER=lexical`) so the shortlist→rerank flow and latency
accounting run in CI with no model download; `EAIP_RERANKER=bge` swaps in a real
BGE cross-encoder for quality. Note: with the offline hashing embedder + lexical
reranker, ranking *quality* is modest (the architecture doc doesn't always rank
first) — that's expected and is exactly what `baseline_vs_improved.py` in Phase 5
will quantify. Phase 2 proves the machinery is correct; the real models supply the
quality.

**Decision/tradeoff — rebuild the BM25 corpus per query.** `rank_bm25` scores an
in-memory corpus, so we rebuild it from the permitted chunks on each query. At
this scale that's trivial and keeps permissions trivially correct (always exactly
the caller's permitted set). A production system would cache a per-tenant sparse
index and invalidate it on ACL/content changes — noted as the scale path.

---

## Phase 3 — Orchestration

**Core concept: a stateful graph as the agent backbone.** LangGraph models the
agent as a state machine — typed `AgentState` flows through nodes, each returning
a partial update LangGraph merges in. Making the state explicit and typed (not an
opaque dict) is half the value: you can see what flows between steps, a
*checkpointer* can persist it, and a human-in-the-loop interrupt can pause on it
and resume from it. We keep node *behavior* (`nodes.py`) separate from graph
*topology* (`graph.py`) so each reads independently. The graph:
`supervisor → {knowledge: retrieve→draft→critic→(loop)→finalize | action:
propose→hitl_gate→execute→finalize}`.

**Multi-agent collaboration built as subgraphs (no second framework).** Two
patterns, both implemented on the state machine: (1) a **supervisor** that
classifies the request and routes to a *knowledge* specialist (grounded RAG) or
an *action* specialist (sensitive tools) — role separation; and (2) a
**draft→critic loop** — a writer produces a grounded draft, a critic reviews it
and asks for a revision, the writer revises. This is the collaboration/revision
concept made concrete: two roles, a review step, a bounded loop. Building it
ourselves (rather than importing a conversational framework) means the control
flow is explicit and every edge is testable.

**Edge case — agent-loop safety (stopping a misbehaving agent).** Autonomous
loops can run forever, burn unbounded tokens, or get stuck. `LoopBudget`
centralizes four independent guardrails — **max iterations**, **token budget**,
**time budget**, **loop detection** (same node repeating ≥ threshold) — plus a
**kill switch** (operator override). Every node ticks the budget and checks it; a
trip sets `stopped_reason` and routes straight to finalize. Crucially this is
applied to the **draft→critic loop too**: the loop is bounded *twice over* — by a
revision cap (the conditional edge) AND by the budget — so a critic that always
asks for changes still terminates. The budget is a live, time-aware object kept
in a per-run registry, *not* in the checkpointed state (it isn't cleanly
serializable, and a resume shouldn't reset the clock). Tested: kill switch,
iteration cap, token/time budget (fake clock), loop detection, and a bounded
critic loop that finishes cleanly rather than via a safety trip.

**Edge case — tool failures (reason about errors, don't crash).** `run_tool`
wraps every tool call with a soft **timeout**, **retries with exponential
backoff**, and — when a tool ultimately fails — returns a *structured error
result* (`ok=False`, `error=...`) rather than raising. So a failing tool surfaces
as data the agent can reason about (and the graph completes with an error
message) instead of an exception that kills the run. Tested at the unit level
(transient-then-succeed, persistent failure, unexpected exception, timeout) and
through the graph (an approved-but-failing `send_email` finishes the run with a
"tool failed" answer).

**Edge case — HITL durability, idempotency, expiry, role routing.** The sensitive
tools (`send_email`, `delete_records`) are gated by `interrupt()`, which suspends
the graph and surfaces the proposed action for approval. Because the graph is
compiled with a **SqliteSaver** checkpointer, the paused state is *durable* — the
demo pauses in one CLI invocation and a *separate* invocation resumes from the
SQLite checkpoint and executes the tool (proven by a test that reopens the DB on
a fresh connection). **Idempotency**: each sensitive call carries an
`idempotency_key` derived from the request, and the side-effect log dedupes by
key, so resuming an already-resolved run does not double-send (tested).
**Expiry**: an approval older than the TTL is rejected as `expired`.
**Role routing**: only resumers carrying an approver role (`admin`/`approver`)
can approve; others are treated as denied (tested). Denials and expiries perform
no side effect.

**The four memory tiers, made concrete.** (1) **In-context** = the `AgentState`
carried through one run. (2) **Episodic** = past runs recorded in SQLite
(`EpisodicStore`), recalled by user. (3) **Semantic** = the Phase 2 RAG corpus,
pulled in by the `retrieve` node. (4) **Procedural** = durable learned rules
(`ProceduralStore`, key/value). The two durable tiers extend the same SQLite
store from Phase 1 — "one store, many capabilities" — and have an in-memory
backend for tests.

**Decision/tradeoff — SqliteSaver version pin (an APIs-change scar).** The
checkpointer is `langgraph-checkpoint-sqlite`. Pinning the obvious `2.x` line
*broke at runtime*: it calls `JsonPlusSerializer.dumps()`, but the installed
`langgraph-checkpoint==4.x` only exposes `dumps_typed`. The fix was the `3.x`
line of the sqlite checkpointer, which targets the new serializer. This is the
"verify current APIs before relying on them" rule paying for itself — the
incompatibility only shows up when you actually run a checkpointed graph, not at
import time.

**Why standardize on LangGraph (and where AutoGen would help vs. hurt).** The
spec asks for the argument I can now make from having built this:

* **What the supervisor + critic patterns gave us here:** explicit, inspectable
  control flow. Routing is a conditional edge I can unit-test; the critic loop is
  a bounded edge I can prove terminates; the HITL pause is a durable interrupt
  with idempotent resume. Every transition is deterministic and visible, which is
  exactly what an enterprise platform needs — auditability, safety limits applied
  uniformly, and state that survives a restart.

* **Where a dedicated conversational multi-agent framework (e.g. AutoGen) adds
  value:** open-ended, *emergent* collaboration — several agents conversing to
  explore an under-specified problem, where you *want* the dialogue to find its
  own shape (brainstorming, debate, free-form tool negotiation). AutoGen's
  group-chat abstraction makes "N agents talk until done" cheap to express, and
  the conversational paradigm is a genuinely different feel worth experiencing
  (the optional Phase 3b spike exists for exactly that).

* **Where it would add uncontrolled complexity:** the moment you need the
  properties above — a hard iteration/token/time budget, a durable
  human-approval gate with idempotent resume, deterministic routing you can test,
  per-edge audit — emergent conversation becomes a liability. "Agents chat until
  they decide they're done" has no natural place to enforce a kill switch, pin a
  checkpoint, or guarantee a sensitive tool ran exactly once. You end up
  re-imposing a state machine on top of the conversation anyway.

* **The conclusion:** standardize on LangGraph as the deterministic, stateful
  backbone (planning, tools, memory, HITL, safety), and reach for a conversational
  framework only for a *bounded* sub-task where emergent dialogue is the point —
  invoked as one node inside the graph, never as the backbone. Control at the
  edges, creativity in a sandbox.
