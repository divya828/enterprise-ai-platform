"""Tests for the RBAC model (viewer/builder/admin)."""

from __future__ import annotations

import pytest

from eaip.platform.rbac import Capability, PermissionDenied, Role, can, require


def test_role_ordering_for_capabilities():
    # viewer can ask but not manage prompts/agents or approve.
    assert can(Role.VIEWER, Capability.ASK)
    assert not can(Role.VIEWER, Capability.MANAGE_PROMPTS)
    assert not can(Role.VIEWER, Capability.APPROVE_ACTION)

    # builder can manage prompts/agents but not approve or view audit.
    assert can(Role.BUILDER, Capability.MANAGE_PROMPTS)
    assert can(Role.BUILDER, Capability.MANAGE_AGENTS)
    assert not can(Role.BUILDER, Capability.APPROVE_ACTION)
    assert not can(Role.BUILDER, Capability.VIEW_AUDIT)

    # admin can do everything.
    for cap in Capability:
        assert can(Role.ADMIN, cap)


def test_require_raises_for_insufficient_role():
    require(Role.ADMIN, Capability.MANAGE_TENANT)  # ok, no raise
    with pytest.raises(PermissionDenied):
        require(Role.VIEWER, Capability.MANAGE_PROMPTS)
    with pytest.raises(PermissionDenied):
        require(Role.BUILDER, Capability.VIEW_AUDIT)


def test_can_accepts_string_roles():
    assert can("admin", Capability.VIEW_AUDIT)
    assert not can("viewer", Capability.MANAGE_AGENTS)
