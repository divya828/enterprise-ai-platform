"""Index resolution — mapping a tenant to its (isolated) chunk index.

With collection-per-tenant isolation, "the index" is no longer a single object —
it's one per tenant. An :class:`IndexResolver` hides that: given a ``tenant_id``
it returns that tenant's :class:`ChunkIndex`. The retrieval pipeline depends on a
resolver, so it stays tenant-agnostic in its logic while being tenant-isolated in
its data.

Two implementations:

* :class:`TenantIndexResolver` — the production path: one shared embedded Qdrant
  client, one collection per tenant (created lazily), cached by tenant id.
* :class:`SingleIndexResolver` — a test/dev convenience that returns the same
  index for every tenant (so existing single-tenant tests work unchanged).
"""

from __future__ import annotations

from typing import Protocol

from qdrant_client import QdrantClient

from eaip.index.store import ChunkIndex
from eaip.platform.tenancy import collection_for_tenant


class IndexResolver(Protocol):
    """Resolves a tenant id to its chunk index."""

    def for_tenant(self, tenant_id: str) -> ChunkIndex:
        """Return the :class:`ChunkIndex` for ``tenant_id`` (creating it if needed)."""
        ...


class SingleIndexResolver:
    """Returns one fixed index for every tenant (tests / single-tenant dev)."""

    def __init__(self, index: ChunkIndex) -> None:
        self._index = index

    def for_tenant(self, tenant_id: str) -> ChunkIndex:
        return self._index


class TenantIndexResolver:
    """One Qdrant client, one collection per tenant, cached by tenant id.

    A single embedded Qdrant store can hold many collections, so we keep one
    client and lazily open/create ``<prefix>__<tenant_id>`` per tenant. Each
    tenant's vectors live in a physically separate collection — the isolation
    boundary.
    """

    def __init__(self, *, path: str, collection_prefix: str, dim: int) -> None:
        self._path = path
        self._prefix = collection_prefix
        self._dim = dim
        if path == ":memory:":
            self._client = QdrantClient(location=":memory:")
        else:
            self._client = QdrantClient(path=path)
        self._cache: dict[str, ChunkIndex] = {}

    def for_tenant(self, tenant_id: str) -> ChunkIndex:
        if tenant_id not in self._cache:
            collection = collection_for_tenant(self._prefix, tenant_id)
            # Reuse the shared client so all tenants live in one embedded store.
            self._cache[tenant_id] = ChunkIndex(self._client, collection, self._dim)
        return self._cache[tenant_id]
