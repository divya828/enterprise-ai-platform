# Architecture

Text diagrams of the platform's data and control flow. This document grows with
each phase; today it reflects **Phase 0** (the skeleton) and sketches the target
shape so the direction is visible.

## Phase 0 — what exists today

```
                         HTTP client (curl / Swagger UI / tests)
                                       |
                                       v
                          ┌──────────────────────────┐
                          │  FastAPI app (eaip.app)   │
                          │  /health        /hello    │
                          └─────────────┬─────────────┘
                                        │ Depends(_provider)
                                        v
                          ┌──────────────────────────┐
                          │  Provider factory          │   reads
                          │  (eaip.providers.factory)  │◀──────────┐
                          └─────────────┬──────────────┘            │
              selects one of            │                  ┌──────────────────┐
        ┌───────────────┬───────────────┼───────────────┐ │ Settings          │
        v               v               v               v │ (eaip.config)     │
   ┌─────────┐    ┌──────────┐    ┌────────────┐  ┌────────┐ env / .env →      │
   │  stub   │    │  ollama  │    │ anthropic  │  │ openai │ EAIP_LLM_PROVIDER  │
   │ default │    │  (http)  │    │  (SDK)     │  │ (SDK)  │ EAIP_LLM_MODEL ... │
   └─────────┘    └──────────┘    └────────────┘  └────────┘─────────────────┘
        │
        │  all implement the LLMProvider protocol:
        │     complete(messages) -> Completion
        v
   Completion(text | tool_calls, model, usage)
```

Key idea: **everything above the provider line depends only on the
`LLMProvider` protocol and the neutral `Message`/`Completion` types** — never on
a concrete SDK. Swapping `EAIP_LLM_PROVIDER` changes the backend with no code
changes elsewhere. The `stub` backend is offline + deterministic and is what CI
and all unit tests use.

## Phase 1 — ingestion flow

```
 data/corpus/documents.json  (30 synthetic docs w/ ACLs; 1 planted injection)
            │
            ▼
   connectors_from_corpus()  ── one CorpusConnector per source ──┐
            │   fetch_since(watermark)   current_ids()           │
            ▼                                                     │
   ┌──────────────────────── IngestionPipeline.sync() ───────────┘
   │  for each changed doc:                         (StateStore: per-source
   │    chunk_document(doc) ── ACL copied onto       watermark + indexed ids,
   │      every chunk, deterministic ids             persisted in SQLite —
   │           │                                     normalized tables)
   │           │
   │           ▼
   │    embedder.embed_documents(texts)   ← hashing (default) | BGE (opt-in)
   │           │
   │           ▼
   │    index.delete_document(doc_id)     ← clear old chunks (no stale/dupes)
   │    index.upsert_chunks(chunks, vecs) ← Qdrant points, ACL in payload
   │
   │  reconcile deletions: indexed_ids − current_ids → delete_document(tombstone)
   └──────────────────────────────────────────────────────────────────────────

   Result: Qdrant collection where every point carries
   {doc_id, source, title, text, ordinal, last_modified, allowed_groups,
    allowed_users, extra} — the ACL payload that Phase 2 filters on.
```

## Module layout (Phases 0–1)

```
src/eaip/
├── __init__.py            # package version
├── app.py                 # FastAPI app factory + /health, /hello
├── config/
│   └── settings.py        # typed Settings; LLMProvider + EmbedderName enums
├── providers/             # LLM provider abstraction (Phase 0)
│   ├── types.py           # Role, Message, ToolCall, Usage, Completion
│   ├── base.py            # LLMProvider Protocol + ProviderError
│   ├── factory.py         # get_provider(settings) -> LLMProvider
│   └── {stub,ollama,anthropic,openai}.py
├── ingestion/             # Phase 1: sources -> documents -> chunks
│   ├── models.py          # ACL, Document, Chunk, SourceType
│   ├── connectors.py      # Connector protocol + CorpusConnector + loaders
│   └── chunker.py         # structure-aware chunking (size/overlap)
├── embeddings/            # Phase 1: text -> dense vectors
│   ├── base.py            # Embedder Protocol
│   ├── factory.py         # get_embedder(settings)
│   ├── hashing.py         # offline deterministic default
│   └── bge.py             # sentence-transformers (opt-in, lazy import)
├── index/                 # Phase 1: vectors -> Qdrant + the pipeline
│   ├── store.py           # ChunkIndex (Qdrant wrapper), ScoredChunk
│   ├── acl_filter.py      # access_filter(user, groups) -> Qdrant Filter
│   └── pipeline.py        # IngestionPipeline, SyncReport
└── storage/               # storage-layer abstraction (SQLite default)
    ├── base.py            # StateStore Protocol + SyncState transfer object
    ├── memory.py          # InMemoryStateStore (tests)
    └── sqlite.py          # SqliteStateStore (normalized tables; app default)

├── retrieval/             # Phase 2: query -> grounded, cited answer
│   ├── dense.py           # DenseRetriever (ACL-filtered vector search)
│   ├── sparse.py          # SparseRetriever (BM25 over ACL-permitted chunks)
│   ├── fusion.py          # reciprocal_rank_fusion (explicit RRF)
│   ├── reranker.py        # Reranker Protocol + Lexical (default) / CrossEncoder
│   ├── reranker_factory.py
│   ├── pipeline.py        # HybridRetriever, Principal, RetrievalResult
│   ├── answerer.py        # GroundedAnswerer (citations + "I don't know" gate)
│   └── service.py         # RetrievalService composition root (.ask)
├── orchestration/         # Phase 3: the LangGraph agent
│   ├── state.py           # AgentState (typed graph state; in-context memory)
│   ├── safety.py          # LoopBudget (iter/token/time/loop + kill switch)
│   ├── tools.py           # Tool catalog, run_tool (timeout/retry), sensitive tools
│   ├── nodes.py           # supervisor / retrieve / draft / critic / propose / finalize
│   ├── graph.py           # StateGraph wiring + HITL interrupt + conditional edges
│   ├── runner.py          # AgentRunner (compile w/ checkpointer, run, resume)
│   └── service.py         # build_runner_cm composition root (durable checkpointer)
├── platform/              # Phase 4: governance (tenancy, RBAC, registries, limits)
│   ├── tenancy.py         # collection_for_tenant, validate_tenant_id
│   ├── rbac.py            # Role/Capability + can()/require() (viewer<builder<admin)
│   ├── registry.py        # PromptRegistry + AgentRegistry (RBAC + audit + lifecycle)
│   └── limits.py          # RateLimiter + TokenBudget (per-tenant; cost attribution)
├── security.py            # Principal (security context; leaf, no package deps)
└── storage/               # storage-layer abstraction (SQLite default)
    ├── base.py            # all store interfaces + transfer objects
    ├── memory.py          # in-memory backends (tests)
    └── sqlite.py          # SQLite backend: state + memory + audit + prompts + agents + usage

scripts/
├── generate_corpus.py     # regenerate the synthetic corpus + golden set
├── ingest.py              # tenant-scoped ingest into embedded Qdrant
├── ask.py                 # query the corpus as a principal (Phase 2 demo)
├── agent.py               # run the orchestration agent (Phase 3 demo; HITL resume)
└── governance.py          # RBAC + prompt rollback + lifecycle + tenancy + audit (Phase 4)
```

## Phase 4 — governance pipeline (on /ask)

```
   request: query + Principal(tenant, user, groups, role)
            │
            ▼
   1. RBAC: require(role, ASK)               ── insufficient role → 403
            ▼
   2. RateLimiter.check(tenant)              ── over per-minute cap → 429
      TokenBudget.check(tenant, day)         ── over daily tokens   → 429
            ▼
   3. RetrievalService.ask(query, principal) ── tenant's OWN collection + ACL
            ▼
   4. TokenBudget.record(tenant, tokens)     ── cost attribution (usage_daily)
      AuditStore.append_event(...)           ── append-only "who did what"
            ▼
   answer + citations

   Registries (builder/admin, audited, tenant-scoped):
     PromptRegistry  add_version → pin/rollback → history     (prompt_version/prompt_active)
     AgentRegistry   create(draft) → transition(test→published→deprecated)  (agent_definition)
       lifecycle transitions validated against ALLOWED_TRANSITIONS

   Isolation: each tenant → its own Qdrant collection (<prefix>__<tenant>);
              every store row keyed by tenant; every read filtered by tenant.
```

## Phase 3 — orchestration flow

```
   query + user/groups + thread_id
            │
            ▼
        supervisor  (LLM-router → "knowledge" | "action")
        │ each node ticks LoopBudget (iter/token/time/loop + kill switch);
        │ a trip sets stopped_reason and jumps to finalize
        ├──────────────────────────────┬───────────────────────────────┐
        ▼ knowledge                     ▼ action
   retrieve (Phase 2 RAG,          propose_action (parse sensitive
   permission-scoped)               tool call: send_email|delete_records)
        ▼                                │
   draft  ◀──────────┐                   ▼
        ▼            │ (revise,      hitl_gate ── interrupt() ──▶ PAUSE
   critic ──────────┘  bounded by        │   (durable via SqliteSaver checkpointer)
        │   revisions cap +              │   resume: Command(resume={approved, roles, ts})
        │   budget)                      ▼
        ▼                           execute_action
   finalize ◀────────────────────── (iff approved + approver role + not expired;
        │                            idempotent via idempotency_key)
        ▼
       answer  (+ episode recorded to EpisodicStore)

   Memory tiers: in-context = AgentState · episodic = SQLite episodes ·
                 semantic = the RAG corpus · procedural = SQLite rules
```

## Phase 2 — retrieval flow

```
   query + Principal(user, groups)
            │
            ▼
   access_filter(user, groups)  ──────────── one ACL gate, applied to BOTH arms
            │                                  (filter-before-rank: forbidden
            ├───────────────┬──────────────┐   chunks never enter either set)
            ▼               ▼              ...
   DenseRetriever      SparseRetriever
   (embed query →      (BM25 over the
    ACL-filtered ANN    ACL-permitted
    in Qdrant)          corpus only)
            │  top-N         │  top-N
            └──────┬─────────┘
                   ▼
        reciprocal_rank_fusion({dense, bm25})   ← rank-based, scale-free
                   │  fused shortlist (top-N)
                   ▼
        Reranker.rerank(query, shortlist)       ← cross-encoder cost; timed
                   │  top-k
                   ▼
        GroundedAnswerer.answer()
          ├─ top_score < threshold OR no chunks → "I don't know" (LLM not called)
          └─ else → LLM with fenced CONTEXT + numbered passages → answer + citations
```

## Target shape (phases 1–6, for orientation)

```
   Ingestion (P1)                Retrieval (P2)                 Orchestration (P3)
   ────────────                  ────────────                   ──────────────────
   connectors ─┐                 ┌─ dense (embeddings) ─┐       LangGraph state machine:
   (confluence,│  chunker        │                      │ RRF     plan → retrieve → answer
    jira, db)  ├──(metadata+ACL)─┤  sparse (BM25)       ├─fuse─► rerank ─► grounded answer
              ─┘  embed+index     └──────────────────────┘ (cross-          with citations
                     │                    ▲                  encoder)            │
                     v                    │ ACL filter by                        │ subgraphs:
                  Qdrant  ◀───────────────┘ requesting user's groups            supervisor → specialists
              (dense+sparse vectors,                                            writer → critic (revise loop)
               ACL metadata per chunk)                                          HITL interrupt (checkpointer)

   Platform capabilities (P4)        Observability + Eval (P5)     Security (P6)
   ──────────────────────────        ─────────────────────────     ─────────────
   multi-tenancy (scoped namespaces) trace ids stitch a run        instructions ⟂ retrieved data
   RBAC (viewer/builder/admin)       latency/token/cost per req     input/output guardrails
   append-only audit log             eval harness (recall@k,        least-privilege tools
   prompt registry (version/pin/     LLM-judge, RAGAS faithfulness) HITL on sensitive tools
     rollback)                       baseline_vs_improved.py        red-team suite (planted
   agent lifecycle states            CI regression gate              injection payload)

   Storage layer (cross-cutting): SQLite by default, abstracted for Postgres swap.
```

## Control-flow notes (target)

- **Retrieval is permission-aware at query time:** candidates are filtered by the
  requesting user's groups/permissions *before* ranking, so an ACL change is
  reflected immediately (no stale index of "who can see what" baked into rank).
- **Orchestration state is durable:** the LangGraph checkpointer persists state at
  each step, enabling human-in-the-loop interrupts that survive a process restart
  and resume idempotently.
- **The critic loop is bounded:** the same agent-loop safety limits (max
  iterations, token/time budget, loop detection) apply to the writer→critic
  revision loop so it can't revise forever.
- **Tracing is end-to-end:** one trace id threads through every LLM call, tool
  call, and retrieval in a (possibly multi-agent) run, so a request can be
  reconstructed for debugging and cost attribution.
