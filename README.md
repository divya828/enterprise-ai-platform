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

Built in phases (0–6). **Phases 0–3 are complete** (scaffolding; ingestion +
connectors; retrieval — the RAG core; orchestration — LangGraph agent with
supervisor/critic, memory tiers, safety limits, and durable HITL). See
[PROJECT_PLAN.md](PROJECT_PLAN.md) for what each phase delivers.

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

Run it again — it upserts **0** documents (the per-source watermark is persisted
in SQLite at `data/eaip.db`) and the chunk total stays flat (re-index updates
rather than duplicates). To use real semantic embeddings instead:
`EAIP_EMBEDDER=bge uv run python scripts/ingest.py` (downloads BGE-small on first
run).

### Ask a question — permission-aware grounded RAG (Phase 2)

After ingesting, ask the corpus a question *as a given principal*. Retrieval is
hybrid (dense + sparse, fused with RRF, reranked) and scoped to the principal's
ACL, so the answer can only cite documents that user may see:

```bash
uv run python scripts/ask.py "how do I set up the vpn" --groups everyone

# Permission isolation in action — same question, different principals:
uv run python scripts/ask.py "what is the FY26 revenue forecast" \
  --user intern@acme.test --groups everyone     # cannot cite the finance doc
uv run python scripts/ask.py "what is the FY26 revenue forecast" \
  --user cfo@acme.test --groups finance         # reaches CONF-7
```

Or over HTTP (`POST /ask`): `{"query": "...", "user": "...", "groups": [...]}`
returns the grounded answer, citations, an `abstained` flag (true when evidence
is too weak → "I don't know"), and per-stage retrieval `timings_ms` (note the
reranking cost). The defaults run offline; `EAIP_RERANKER=bge` and a real LLM
provider give production-grade quality.

### Run the orchestration agent (Phase 3)

```bash
uv sync --extra rag --extra orchestration
```

The agent is a LangGraph state machine: a supervisor routes each request to a
**knowledge** specialist (grounded RAG) or an **action** specialist (sensitive
tools), with loop-safety budgets, a draft→critic revision loop, four memory tiers,
and a durable human-in-the-loop gate on irreversible actions.

```bash
# Knowledge path: plan → retrieve → draft → critic → finalize
uv run python scripts/agent.py "how do I set up the vpn" --groups everyone

# Action path: pauses for human approval of a sensitive tool...
uv run python scripts/agent.py "send email to ceo@acme.test" --thread act1 --groups engineering
# ...then a SEPARATE invocation resumes from the durable SQLite checkpoint:
uv run python scripts/agent.py "send email to ceo@acme.test" --thread act1 --approve --role admin
```

The pause/resume survives across invocations because graph state is checkpointed
to SQLite (`data/checkpoints.sqlite`). Approvals are idempotent (resuming twice
won't double-send), expire after a TTL, and are role-routed (only `admin`/
`approver` roles may approve). Safety limits (max iterations, token/time budget,
loop detection, kill switch) stop a misbehaving run, and tool failures are
retried then surfaced as reasoned errors rather than crashes.

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
| Storage abstraction (SQLite default, Postgres-swappable) | [`src/eaip/storage/`](src/eaip/storage/) |
| Synthetic corpus + golden set            | [`scripts/generate_corpus.py`](scripts/generate_corpus.py), [`data/corpus/`](data/corpus/) |
| Hybrid retrieval, RRF, cross-encoder rerank, grounded citations, "I don't know" | [`src/eaip/retrieval/`](src/eaip/retrieval/) |
| LangGraph agent: supervisor + critic, memory tiers, loop safety, durable HITL | [`src/eaip/orchestration/`](src/eaip/orchestration/) |
| Memory tiers (episodic + procedural) in the SQLite store | [`src/eaip/storage/`](src/eaip/storage/) |
| Multi-tenancy, RBAC, audit, prompt registry | `src/eaip/platform/` _(Phase 4)_           |
| Tracing + evaluation harness             | `src/eaip/observability/`, `evals/` _(Phase 5)_ |
| Prompt-injection defenses, red-team suite | `src/eaip/security/`, `tests/redteam/` _(Phase 6)_ |

## License

For learning use.
