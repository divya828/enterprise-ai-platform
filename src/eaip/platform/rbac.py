"""Role-based access control (RBAC).

RBAC answers "is this principal *allowed* to do this?" — distinct from the ACL
checks in retrieval, which answer "may this principal *see* this document?". A
viewer can ask questions; a builder can additionally manage agents and prompts;
an admin can additionally change tenant config, approve sensitive actions, and
read the audit log.

We model three ordered roles and a capability per action. A role is permitted a
capability if its rank is at or above the capability's required rank. Enforcement
happens at two layers (defense in depth):

* **API boundary** — an endpoint declares the minimum capability it needs;
  :func:`require` raises :class:`PermissionDenied` (→ HTTP 403) otherwise.
* **Retrieval** — even an authorized role is still bound by the principal's
  tenant and per-document ACL, so RBAC never *widens* data access; it only gates
  *operations*.

Keeping roles ordered (rather than free-form permission sets) is enough to teach
the concept; the note in LEARNINGS explains when you'd graduate to a
permission-set model.
"""

from __future__ import annotations

from enum import StrEnum


class Role(StrEnum):
    """Ordered platform roles. Higher in this list = more privilege."""

    VIEWER = "viewer"
    BUILDER = "builder"
    ADMIN = "admin"


# Rank used for "at least this role" comparisons.
_RANK = {Role.VIEWER: 0, Role.BUILDER: 1, Role.ADMIN: 2}


class Capability(StrEnum):
    """Discrete things a principal might be allowed to do."""

    ASK = "ask"  # query the platform (RAG / agent)
    MANAGE_PROMPTS = "manage_prompts"  # create/pin/rollback prompt versions
    MANAGE_AGENTS = "manage_agents"  # create/transition agent definitions
    APPROVE_ACTION = "approve_action"  # approve a sensitive HITL action
    VIEW_AUDIT = "view_audit"  # read the audit log
    MANAGE_TENANT = "manage_tenant"  # change tenant config / limits


# Minimum role required for each capability.
_REQUIRED: dict[Capability, Role] = {
    Capability.ASK: Role.VIEWER,
    Capability.MANAGE_PROMPTS: Role.BUILDER,
    Capability.MANAGE_AGENTS: Role.BUILDER,
    Capability.APPROVE_ACTION: Role.ADMIN,
    Capability.VIEW_AUDIT: Role.ADMIN,
    Capability.MANAGE_TENANT: Role.ADMIN,
}


class PermissionDenied(PermissionError):
    """Raised when a principal's role lacks a required capability (→ HTTP 403)."""


def can(role: Role | str, capability: Capability) -> bool:
    """Return True if ``role`` is permitted ``capability``.

    An unrecognized role is treated as having no privileges (fail closed) rather
    than raising — so a bad role at the API boundary becomes a clean 403, never a
    500.
    """
    try:
        role = Role(role)
    except ValueError:
        return False
    return _RANK[role] >= _RANK[_REQUIRED[capability]]


def require(role: Role | str, capability: Capability) -> None:
    """Raise :class:`PermissionDenied` unless ``role`` is permitted ``capability``."""
    if not can(role, capability):
        raise PermissionDenied(
            f"role '{role}' is not permitted to '{capability}' "
            f"(requires at least '{_REQUIRED[capability]}')"
        )
