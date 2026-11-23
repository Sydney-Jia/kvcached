# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""In-process residency gate for elastic (non-capacity) serving.

Elastic mode must never fence the engine between decode steps: growth maps the
missing pages and retirement waits for an idle scheduling point. The gate is a
process-wide handle because a tensor-parallel-1 engine runs the scheduler and
the worker in one process, and the scheduler is the only place that can evict
the prefix blocks whose pages retirement is allowed to release.
"""

import json
import os

_arena = None
_owners = None


def install(arena, owners):
    global _arena, _owners
    if owners is None:
        # A worker registers before reshape publishes the owner table would
        # keep a permanent None here, and every idle reclaim would return 0
        # while the RPC still reported success. Fail loudly instead.
        raise ValueError('Native gate requires a validated owner table')
    _arena = arena
    _owners = owners
    return _arena


def clear():
    global _arena, _owners
    _arena = None
    _owners = None


def installed():
    return _arena is not None and _owners is not None


def pressure(threshold=None):
    """Whether the shared ledger is mapped close to its sampled budget."""
    if _arena is None:
        return False
    if threshold is None:
        try:
            threshold = float(os.getenv(
                'KVCACHED_NATIVE_IDLE_RECLAIM_UTILIZATION', '0.8'))
        except ValueError:
            threshold = 0.8
    if threshold <= 0:
        return True
    try:
        state = _arena.operations.physical_budget_state()
        budget = int(state.get('budget_bytes', 0) or 0)
        charged = int(state.get('charged_bytes', 0) or 0)
    except Exception:
        return True
    return budget <= 0 or charged >= budget * threshold


def reclaim_idle():
    """Fence and retire unowned pages at an idle scheduling point.

    Only the scheduler's empty step may call this: the physical transition
    waits for in-flight work, which is free when nothing was scheduled.
    """
    if _arena is None or _owners is None:
        return 0
    try:
        reclaimed = int(
            _arena.transition(_owners.required_pages())['reclaimed_bytes'])
    except Exception as error:
        print('KVCACHED_NATIVE_IDLE_RECLAIM_FAILED ' + repr(error)[:200],
              flush=True)
        return 0
    if reclaimed:
        # Without this line a dead retirement path is indistinguishable from a
        # ledger that simply had nothing to retire.
        print('KVCACHED_NATIVE_GATE_RECLAIM ' + json.dumps(
            dict(reclaimed=reclaimed)), flush=True)
    return reclaimed
