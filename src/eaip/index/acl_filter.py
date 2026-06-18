"""Build a Qdrant filter that enforces a requesting principal's read access.

A principal may read a chunk if their user id is in the chunk's ``allowed_users``
OR any of their groups is in the chunk's ``allowed_groups``. In Qdrant terms that
is a ``should`` (logical OR) over two match conditions. Applying this filter at
query time — rather than filtering results in Python after the fact — means the
vector store never even returns chunks the user can't see, which is both faster
and the correct security posture (the candidate set is constrained before
ranking).

Because the filter reads the ACL stored on each point, an ACL change takes effect
as soon as the chunk is re-indexed with the new ACL — there is no separate
"who can see what" index to fall out of sync. (Phase 2 tests this revocation.)
"""

from __future__ import annotations

from collections.abc import Iterable

from qdrant_client import models as qm


def access_filter(*, user: str, groups: Iterable[str]) -> qm.Filter:
    """Return a Qdrant filter matching only chunks this principal may read.

    Fail-closed: a chunk with empty ACL lists matches neither condition, so it is
    invisible to everyone — consistent with :meth:`ACL.permits`.
    """
    should: list[qm.Condition] = [
        qm.FieldCondition(key="allowed_users", match=qm.MatchAny(any=[user])),
    ]
    group_list = list(groups)
    if group_list:
        should.append(qm.FieldCondition(key="allowed_groups", match=qm.MatchAny(any=group_list)))
    # `should` is OR; min_should defaults to 1 — at least one condition must hold.
    return qm.Filter(should=should)
