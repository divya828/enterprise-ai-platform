# Enterprise AI Platform — a learning-grade reference implementation

A runnable, **laptop-scale** reference implementation of an enterprise AI agent
platform. It exists to teach the concepts and edge cases behind building one:
the platform abstraction, multi-agent orchestration, enterprise RAG, and the
platform capabilities (multi-tenancy, RBAC, audit, observability, evaluation,
security hardening).

> **This is a study artifact, not production infrastructure.** It optimizes for
> clarity, correctness, and explicit edge-case handling over scale. Mock
> connectors and a small synthetic corpus stand in for real
> Confluence/Jira/database sources. Nothing here requires a paid service to run.

If you are reading the code to learn, start with **[LEARNINGS.md](LEARNINGS.md)**
(the narrative study guide), then **[ARCHITECTURE.md](ARCHITECTURE.md)** (the
data + control flow), then **[PROJECT_PLAN.md](PROJECT_PLAN.md)** (the phase
roadmap).

## Status

Built in phases (0–6). **Phases 0–1 are complete** (scaffolding; ingestion +
connectors). See [PROJECT_PLAN.md](PROJECT_PLAN.md) for what each phase delivers.

## Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** for dependency + virtualenv management

No API key and no local model are required to run the project or its tests — the
default LLM provider is an offline, deterministic **stub** (see below).

## Install

```bash
uv sync                 # create the venv and install runtime + dev deps
cp .env.example .env     # optional: only needed to switch providers
```

Phase-specific heavy dependencies (Qdrant, sentence-transformers, LangGraph,
RAGAS) live in optional extras and are installed when their phase begins:

```bash
uv sync --extra rag             # Phase 1–2: ingestion + retrieval
uv sync --extra orchestration   # Phase 3: LangGraph
uv sync --extra llm             # real LLM backends (anthropic / openai)
uv sync --extra eval            # Phase 5: evaluation harness
```

## Run

```bash
uv run uvicorn eaip.app:app --reload
```

Then:

```bash
curl localhost:8000/health
curl -X POST localhost:8000/hello -H 'content-type: application/json' \
  -d '{"message": "hello platform"}'
# => {"reply":"[stub] received: hello platform","provider":"stub",...}
```

Interactive API docs are at `http://localhost:8000/docs`.

### Ingest the corpus (Phase 1)

The synthetic corpus (`data/corpus/documents.json`, regenerable via
`uv run python scripts/generate_corpus.py`) is ingested into an embedded Qdrant
index — no Docker, no API key, no model download (the default embedder is an
offline hashing embedder):

```bash
uv sync --extra rag
uv run python scripts/ingest.py
# [confluence] upserted 16 docs (16 chunks), deleted 0 docs
# ... Total indexed chunks: 30
```

Run it again — it upserts **0** documents (the per-source watermark is persisted)
and the chunk total stays flat (re-index updates rather than duplicates). To use
real semantic embeddings instead: `EAIP_EMBEDDER=bge uv run python scripts/ingest.py`
(downloads BGE-small on first run).

## Test

```bash
uv run pytest        # all tests, offline, no API key
uv run ruff check .  # lint
uv run ruff format . # format
```

CI runs the same `pytest` against the **stub** provider — CI never calls a live
LLM.

## LLM providers

The platform depends only on a small `LLMProvider` protocol; concrete backends
are selected at runtime via `EAIP_LLM_PROVIDER`. This is the *provider strategy*
concept (avoiding vendor lock-in, enabling an offline default and deterministic
tests).

| `EAIP_LLM_PROVIDER` | Needs            | Notes                                                                 |
| ------------------- | ---------------- | --------------------------------------------------------------------- |
| `stub` (default)    | nothing          | Offline, deterministic. Used by all tests + CI. Scriptable for agent tests. |
| `ollama`            | local Ollama     | Free, offline. Small local models handle multi-step tool use poorly.  |
| `anthropic`         | `ANTHROPIC_API_KEY` | Best agent behavior. `uv sync --extra llm`. Default model `claude-opus-4-8`. |
| `openai`            | `OPENAI_API_KEY` | `uv sync --extra llm`.                                                 |

Running **real agent flows (Phase 3+)** and the **eval harness (Phase 5)**
requires setting one of `ollama` / `anthropic` / `openai`. A hosted model
(Claude or OpenAI) gives noticeably better agent behavior than a small local
Ollama model. To switch:

```bash
echo "EAIP_LLM_PROVIDER=anthropic"   >> .env
echo "ANTHROPIC_API_KEY=sk-ant-..."  >> .env
uv sync --extra llm
```

The **scripted stub** is a first-class testing tool: hand it a queue of
responses (plain text or simulated tool calls) and it replays them in order, so
orchestration and edge-case tests are fully deterministic with no model. See
`src/eaip/providers/stub.py`.

## Concept map

Each module teaches a concept. (Modules marked _(later phase)_ don't exist yet.)

| Concept                                  | Where it lives                                |
| ---------------------------------------- | --------------------------------------------- |
| Provider strategy / avoiding lock-in     | [`src/eaip/providers/`](src/eaip/providers/)  |
| Typed configuration / config layer       | [`src/eaip/config/`](src/eaip/config/)        |
| HTTP surface / app factory + DI          | [`src/eaip/app.py`](src/eaip/app.py)          |
| Connectors, ACL model, structure-aware chunking | [`src/eaip/ingestion/`](src/eaip/ingestion/) |
| Dense embeddings (hashing / BGE)         | [`src/eaip/embeddings/`](src/eaip/embeddings/) |
| Qdrant index, ACL filter, ingestion pipeline | [`src/eaip/index/`](src/eaip/index/)      |
| Synthetic corpus + golden set            | [`scripts/generate_corpus.py`](scripts/generate_corpus.py), [`data/corpus/`](data/corpus/) |
| Hybrid retrieval, RRF, reranking, citations | `src/eaip/retrieval/` _(Phase 2)_          |
| LangGraph orchestration, supervisor + critic, HITL | `src/eaip/orchestration/` _(Phase 3)_ |
| Multi-tenancy, RBAC, audit, prompt registry | `src/eaip/platform/` _(Phase 4)_           |
| Tracing + evaluation harness             | `src/eaip/observability/`, `evals/` _(Phase 5)_ |
| Prompt-injection defenses, red-team suite | `src/eaip/security/`, `tests/redteam/` _(Phase 6)_ |

## License

For learning use.
