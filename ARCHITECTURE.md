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
   │  for each changed doc:                         (SyncState: per-source
   │    chunk_document(doc) ── ACL copied onto       watermark + indexed ids,
   │      every chunk, deterministic ids             persisted to JSON)
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
└── index/                 # Phase 1: vectors -> Qdrant + the pipeline
    ├── store.py           # ChunkIndex (Qdrant wrapper), ScoredChunk
    ├── acl_filter.py      # access_filter(user, groups) -> Qdrant Filter
    └── pipeline.py        # IngestionPipeline, SyncState, SyncReport

scripts/
├── generate_corpus.py     # regenerate the synthetic corpus + golden set
└── ingest.py              # run a full ingest into embedded Qdrant
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
