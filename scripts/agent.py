"""Run the orchestration agent on a query — the Phase 3 demo.

Examples:
  uv run python scripts/agent.py "how do I set up the vpn"            # knowledge path
  uv run python scripts/agent.py "send email to ceo@acme.test"        # action path (HITL)
  uv run python scripts/agent.py "delete records where status=stale" --approve --role admin

Shows the supervisor routing a request to the knowledge or action specialist. A
knowledge request runs plan→retrieve→draft→critic→finalize and prints the answer.
An action request pauses at the human-in-the-loop gate; pass ``--approve`` (and a
``--role`` that may approve) to resume and execute the sensitive tool, or omit it
to see the pending approval request.

Requires an ingested index (run scripts/ingest.py first). Uses the durable SQLite
checkpointer, so a paused approval persists across invocations by thread id.
Defaults are offline (stub LLM, hashing embedder).
"""

from __future__ import annotations

import argparse

from eaip.orchestration.service import build_runner_cm


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the orchestration agent.")
    parser.add_argument("query")
    parser.add_argument("--user", default="newhire@acme.test")
    parser.add_argument("--groups", default="everyone")
    parser.add_argument("--thread", default="demo-1", help="Run/thread id (for resume).")
    parser.add_argument("--approve", action="store_true", help="Approve a pending action.")
    parser.add_argument("--role", default="admin", help="Approver role (for --approve).")
    args = parser.parse_args()

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]

    with build_runner_cm() as runner:
        # If approving, resume an existing paused run; otherwise start a new one.
        if args.approve and runner.pending_approval(args.thread):
            out = runner.resume(args.thread, approved=True, approver_roles=[args.role])
        else:
            out = runner.run(args.query, user=args.user, groups=groups, thread_id=args.thread)

        print(f"Q: {args.query}  (as {args.user}, groups={groups})")
        print(f"route: {out.route or '?'}")
        if out.awaiting_approval:
            req = out.interrupt or {}
            print("\n⏸  Awaiting human approval for a sensitive action:")
            print(f"    tool: {req.get('tool')}")
            print(f"    args: {req.get('args')}")
            print(
                f"\nResume with: uv run python scripts/agent.py '{args.query}' "
                f"--thread {args.thread} --approve --role admin"
            )
        else:
            if out.stopped_reason:
                print(f"stopped: {out.stopped_reason}")
            print(f"\nAnswer: {out.answer}")


if __name__ == "__main__":
    main()
