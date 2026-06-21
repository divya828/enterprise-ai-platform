"""The Principal — the security context a request runs under.

This is a foundational, dependency-free type used across layers (retrieval scopes
to it; the platform governance layer reads its tenant + role). It lives in its own
leaf module so both ``retrieval`` and ``platform`` can depend on it without
creating an import cycle between those two packages.

A principal bundles the four things a request's authority depends on:

* ``user``   — the identity (also used for user-level ACL matches),
* ``groups`` — drive document-level ACL filtering,
* ``tenant`` — scopes which tenant's data/config the request may touch,
* ``role``   — drives RBAC (what operations are permitted).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Principal:
    """The full security context a request is made on behalf of.

    Defaults keep earlier-phase call sites working: a lone ``user`` lands in the
    default tenant with the lowest-privilege role.
    """

    user: str
    groups: frozenset[str] = frozenset()
    tenant: str = "acme"
    role: str = "viewer"

    @classmethod
    def of(
        cls,
        user: str,
        groups: list[str] | None = None,
        *,
        tenant: str = "acme",
        role: str = "viewer",
    ) -> Principal:
        return cls(user=user, groups=frozenset(groups or ()), tenant=tenant, role=role)
