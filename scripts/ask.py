"""Ask the corpus a question, as a given principal — the Phase 2 demo.

Run with: ``uv run python scripts/ask.py "how do I set up the vpn?"``
Optional:  ``--user cfo@acme.test --groups finance``

Shows permission-aware hybrid retrieval end to end: the reranked top-k with
scores, per-stage latency (note the reranking cost), and the grounded answer with
citations — or "I don't know" when the best evidence is too weak or the principal
isn't permitted to see the relevant documents.

Requires an ingested index (run ``scripts/ingest.py`` first). Uses the configured
provider/embedder/reranker; defaults are offline (stub LLM, hashing embedder,
lexical reranker) so it runs with no key/model — switch via env for real quality.
"""

from __future__ import annotations

import argparse

from eaip.retrieval import Principal
from eaip.retrieval.service import RetrievalService


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the corpus a question.")
    parser.add_argument("query", help="The question to ask.")
    parser.add_argument("--user", default="newhire@acme.test", help="Requesting user id.")
    parser.add_argument(
        "--groups",
        default="everyone",
        help="Comma-separated groups the user belongs to (e.g. 'engineering,everyone').",
    )
    args = parser.parse_args()

    principal = Principal.of(args.user, [g.strip() for g in args.groups.split(",") if g.strip()])
    service = RetrievalService.from_settings()
    result = service.ask(args.query, principal)

    print(f"Q: {args.query}")
    print(f"   (as {principal.user}, groups={sorted(principal.groups)})\n")

    r = result.retrieval
    print("Reranked top-k:")
    for i, sc in enumerate(r.chunks, start=1):
        print(f"  {i}. [{sc.score:.3f}] {sc.chunk.doc_id} — {sc.chunk.title}")
    timings = " ".join(f"{k}={v:.1f}ms" for k, v in sorted(r.timings_ms.items()))
    print(f"\nLatency: {timings}\n")

    ans = result.answer
    if ans.abstained:
        print(f"Answer: {ans.text}  (abstained; top_score={ans.top_score:.3f})")
    else:
        print(f"Answer: {ans.text}")
        print("Citations: " + ", ".join(f"[{c.label}] {c.doc_id}" for c in ans.citations))


if __name__ == "__main__":
    main()
