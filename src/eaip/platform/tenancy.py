"""Multi-tenancy: isolating one customer's data and config from another's.

A multi-tenant platform serves many organizations from one deployment, and the
non-negotiable property is **isolation** — tenant A must never see tenant B's
documents, config, usage, or cost. We isolate at two layers:

* **Vector data** — each tenant gets its own Qdrant *collection*
  (``<prefix>__<tenant_id>``). This is *physical* isolation: a query against one
  tenant's collection cannot return another tenant's chunks even if the ACL
  filter had a bug, because the other tenant's vectors aren't in the collection
  at all. (The alternative — one collection with a ``tenant_id`` payload filter —
  is lighter to operate but relies on every query remembering the filter; we
  chose collection-per-tenant for defense in depth. See LEARNINGS for the trade.)

* **State / config / usage** — every row the platform stores (audit events,
  prompt versions, agent definitions, usage counters) is keyed by ``tenant_id``,
  and every read filters by it.

A ``tenant_id`` must be safe to embed in a collection name, so we validate it.
"""

from __future__ import annotations

import re

_TENANT_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")


def validate_tenant_id(tenant_id: str) -> str:
    """Return ``tenant_id`` if it's a safe identifier, else raise.

    Restricting the charset keeps tenant ids usable as collection-name suffixes
    and as SQL parameter values without surprises.
    """
    if not _TENANT_RE.match(tenant_id):
        raise ValueError(f"invalid tenant_id {tenant_id!r}: must match {_TENANT_RE.pattern}")
    return tenant_id


def collection_for_tenant(prefix: str, tenant_id: str) -> str:
    """Return the per-tenant Qdrant collection name."""
    return f"{prefix}__{validate_tenant_id(tenant_id)}"
