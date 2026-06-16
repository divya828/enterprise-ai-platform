# Project Plan

A phase-by-phase roadmap for the learning-grade enterprise AI agent platform.
Each phase ends with: tests green, app runs, a conventional commit, a push, and
an update to [LEARNINGS.md](LEARNINGS.md). We do **one phase at a time** and stop
for review before the next.

## Guiding constraints

- **Laptop-scale, free, offline-first.** No phase requires a paid service to run
  its tests. The default LLM provider is an offline deterministic stub.
- **Clarity over cleverness.** Small, single-responsibility modules; type hints;
  docstrings that explain *why*.
- **Edge cases are the point.** Each phase names the edge cases it handles and
  proves them with tests.
- **Storage is abstracted.** SQLite by default; the storage layer is written so
  Postgres is a drop-in swap (documented in LEARNINGS.md).

## Tech stack (defaults)

| Concern         | Choice                                                              |
| --------------- | ------------------------------------------------------------------ |
| Language/tooling | Python 3.12, `uv`, `ruff`, `pytest`                               |
| API             | FastAPI                                                             |
| Orchestration   | LangGraph **only** (checkpointer for durable state + HITL); supervisor + critic as subgraphs |
| Vector store    | Qdrant (local/embedded); dense + sparse; explicit RRF fusion       |
| Embeddings      | sentence-transformers, small local model (e.g. BGE-small)          |
| Reranker        | local cross-encoder (e.g. BGE-reranker-base)                       |
| State/audit/prompts | SQLite (abstracted storage layer)                              |
| LLM provider    | provider abstraction: **stub** (default), ollama, anthropic, openai |

> **Pin note (2026-06-16):** versions verified against PyPI before pinning. See
> LEARNINGS.md "Verifying current APIs" for the table and the LangGraph-1.x note.

## Phases

### Phase 0 — Scaffolding ✅ (complete)
Repo + remote, uv project, dependency pins, ruff/pytest config, `.env.example`,
`.gitignore`, folder structure, planning docs, the LLM provider abstraction with
a working offline default, and a passing smoke test.
**DoD:** `uv run pytest` passes; the app boots; the provider returns a completion
locally with no key.

### Phase 1 — Ingestion & connectors
Mock Confluence/Jira/DB connectors loading a synthetic corpus with metadata +
ACLs; a structure-aware chunker preserving metadata + ACLs per chunk; embedding
+ indexing into Qdrant.
**Edge cases (handled + tested):** incremental re-indexing via a `last_synced`
watermark (upsert, no duplicates); deletions/tombstones (deleting a source doc
removes its chunks); ACL metadata surviving onto every chunk.

### Phase 2 — Retrieval (the RAG core)
Dense + sparse/BM25 retrieval, explicit RRF fusion, cross-encoder reranking of
the fused shortlist, grounded answers with citations.
**Edge cases:** permission-aware retrieval (no leakage); permission
freshness/revocation; low-confidence → "I don't know" instead of hallucinating;
reranking only on the shortlist with latency logged.

### Phase 3 — Orchestration
LangGraph graph (plan → retrieve → answer) with explicit state + checkpointer;
typed tool schemas + small tool catalog; four memory tiers (in-context,
episodic, semantic=RAG, procedural); supervisor + draft→critic subgraphs; a
human-in-the-loop interrupt for a sensitive action.
**Edge cases:** agent-loop safety (max iterations, token/time budget, loop
detection, kill switch — applied to the critic loop too); tool failures
(timeout, retry+backoff, reasoning about errors); HITL durability (persist at
interrupt, resume, idempotent, approval timeout/expiry, role-routed approvals).

### Phase 3b — AutoGen spike (optional, isolated)
Only if requested. A standalone `spikes/autogen/` script (not wired into the
platform, not a dependency, not in CI) to feel the conversational-collaboration
paradigm, contrasted with the LangGraph supervisor/critic approach in LEARNINGS.

### Phase 4 — Platform capabilities
Multi-tenancy (tenant-scoped namespaces/filters, per-tenant config + rate limits
+ token budgets + cost attribution); RBAC (viewer/builder/admin at API and
retrieval); append-only audit log; prompt registry with versioning
(history/pin/rollback); agent-definition abstraction + lifecycle states
(draft → test → published → deprecated).
**Edge cases:** no config/data bleed across tenants; budget throttling;
unauthorized role denied; prompt rollback works.

### Phase 5 — Observability & evaluation
Structured tracing (trace id stitching a multi-agent run; latency/token/cost per
request + per tenant); eval harness over the golden set (recall@k, LLM-as-judge
correctness, RAGAS-style faithfulness/relevancy); `baseline_vs_improved.py`
(naive RAG vs hybrid+rerank, accuracy delta); eval suite wired into CI as a
regression gate.
**Edge cases:** LLM-as-judge variance (multi-run or rubric, variance noted); a
deliberately-bad prompt proves the CI gate fails on regression.

### Phase 6 — Security hardening
Indirect prompt-injection defenses (separate instructions from retrieved data;
input/output guardrails; least-privilege tools; HITL on sensitive tools; output
filtering); text-to-SQL safety if any structured querying (read-only, row
limits, query validation).
**Edge cases:** a red-team suite of malicious docs (incl. the planted Phase 1
injection payload) asserting the agent does NOT follow embedded instructions,
does NOT exfiltrate, and does NOT call a privileged tool. The passing suite is
the deliverable.

## Final definition of done
Every phase committed and pushed; CI green; README enables install/run/test in a
few commands; LEARNINGS.md reads as a coherent walkthrough; red-team +
permission-leakage tests pass; `baseline_vs_improved.py` shows a real,
documented accuracy delta.
