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

## Module layout (Phase 0)

```
src/eaip/
├── __init__.py            # package version
├── app.py                 # FastAPI app factory + /health, /hello
├── config/
│   ├── __init__.py
│   └── settings.py        # typed Settings (pydantic-settings), LLMProvider enum
└── providers/
    ├── __init__.py        # public surface (protocol, types, factory)
    ├── types.py           # Role, Message, ToolCall, Usage, Completion
    ├── base.py            # LLMProvider Protocol + ProviderError
    ├── factory.py         # get_provider(settings) -> LLMProvider
    ├── stub.py            # scripted/echo offline default
    ├── ollama.py          # local models via HTTP
    ├── anthropic.py       # Claude via SDK (lazy import)
    └── openai.py          # GPT via SDK (lazy import)
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
