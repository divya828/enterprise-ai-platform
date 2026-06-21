"""Platform capabilities (Phase 4): tenancy, RBAC, governed registries, limits."""

from eaip.platform.limits import Decision, RateLimiter, TokenBudget
from eaip.platform.rbac import Capability, PermissionDenied, Role, can, require
from eaip.platform.registry import AgentRegistry, LifecycleError, PromptRegistry
from eaip.platform.tenancy import collection_for_tenant, validate_tenant_id

__all__ = [
    "Role",
    "Capability",
    "PermissionDenied",
    "can",
    "require",
    "PromptRegistry",
    "AgentRegistry",
    "LifecycleError",
    "RateLimiter",
    "TokenBudget",
    "Decision",
    "collection_for_tenant",
    "validate_tenant_id",
]
