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
