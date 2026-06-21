"""The tool catalog: typed schemas, robust execution, and sensitivity flags.

Agents act through *tools*. Each tool here carries a typed JSON-schema for its
arguments (so a model — or a test — knows how to call it), a flag marking whether
it is *sensitive* (irreversible / outward-facing, requiring human approval), and a
handler. Execution is wrapped with the resilience an agent needs in the real
world:

* **timeout** — a tool that hangs must not hang the whole run.
* **retry with exponential backoff** — transient failures are retried a few times
  before giving up.
* **error as data, not a crash** — when a tool ultimately fails, we return a
  structured error result the agent can *reason about* (and try a different
  approach) instead of letting the exception kill the graph.

The sensitive tools (``send_email``, ``delete_records``) are deliberately mock and
*idempotency-keyed*: each call carries an ``idempotency_key`` and the handler
records performed keys, so re-executing the same call (e.g. after a HITL resume)
does not perform the side effect twice. That property is what the HITL idempotency
test relies on.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolResult:
    """The outcome of a tool call — success or a structured, reasoned-about error."""

    ok: bool
    output: str = ""
    error: str = ""
    attempts: int = 1


@dataclass(frozen=True)
class Tool:
    """A callable tool with a typed schema and a sensitivity flag."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema for the arguments
    handler: Callable[[dict[str, Any]], str]
    sensitive: bool = False  # irreversible / outward-facing -> needs HITL approval


class ToolError(RuntimeError):
    """Raised by a handler to signal a (possibly transient) failure."""


def run_tool(
    tool: Tool,
    args: dict[str, Any],
    *,
    timeout_s: float = 5.0,
    max_attempts: int = 3,
    base_backoff_s: float = 0.05,
    sleep: Callable[[float], None] = time.sleep,
) -> ToolResult:
    """Execute ``tool`` with timeout, bounded retries, and backoff.

    Returns a :class:`ToolResult`; a final failure is reported as ``ok=False`` with
    an ``error`` message rather than raised, so the agent can reason about it. The
    ``sleep`` parameter is injectable so tests don't actually wait between retries.

    Timeout note: Python can't safely abort an arbitrary running function, so we
    enforce a *soft* timeout — the handler reports how long it took and we treat an
    over-budget call as a failure. A production system would run tools in a
    process/thread with a hard kill; the contract (over-time == failure) is the
    same and is what we test against.
    """
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        start = time.perf_counter()
        try:
            output = tool.handler(args)
            elapsed = time.perf_counter() - start
            if elapsed > timeout_s:
                last_error = f"timeout after {elapsed:.2f}s (budget {timeout_s}s)"
            else:
                return ToolResult(ok=True, output=output, attempts=attempt)
        except ToolError as exc:
            last_error = str(exc)
        except Exception as exc:  # unexpected handler bug — still don't crash the graph
            last_error = f"unexpected error: {exc}"

        if attempt < max_attempts:
            sleep(base_backoff_s * (2 ** (attempt - 1)))  # exponential backoff

    return ToolResult(ok=False, error=last_error, attempts=max_attempts)


# ---------------------------------------------------------------------------
# Concrete sensitive tools (mock, idempotency-keyed)
# ---------------------------------------------------------------------------


class _SideEffectLog:
    """Records idempotency keys of performed side effects, to dedupe replays."""

    def __init__(self) -> None:
        self.performed: dict[str, str] = {}  # key -> output

    def perform(self, key: str, do: Callable[[], str]) -> str:
        if key in self.performed:
            return f"(idempotent replay) {self.performed[key]}"
        output = do()
        self.performed[key] = output
        return output


def build_default_tools() -> tuple[dict[str, Tool], _SideEffectLog]:
    """Construct the standard tool catalog and its shared side-effect log.

    Returns the tools keyed by name plus the log, so tests can inspect exactly
    what side effects were performed (and assert no double-execution).
    """
    log = _SideEffectLog()

    def send_email(args: dict[str, Any]) -> str:
        key = args["idempotency_key"]
        return log.perform(
            key, lambda: f"email sent to {args['to']} (subject: {args.get('subject', '')})"
        )

    def delete_records(args: dict[str, Any]) -> str:
        key = args["idempotency_key"]
        return log.perform(key, lambda: f"deleted records matching {args['filter']}")

    tools = {
        "send_email": Tool(
            name="send_email",
            description="Send an email to a recipient. Irreversible / outward-facing.",
            parameters={
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                    "idempotency_key": {"type": "string"},
                },
                "required": ["to", "idempotency_key"],
            },
            handler=send_email,
            sensitive=True,
        ),
        "delete_records": Tool(
            name="delete_records",
            description="Delete database records matching a filter. Irreversible.",
            parameters={
                "type": "object",
                "properties": {
                    "filter": {"type": "string"},
                    "idempotency_key": {"type": "string"},
                },
                "required": ["filter", "idempotency_key"],
            },
            handler=delete_records,
            sensitive=True,
        ),
    }
    return tools, log
