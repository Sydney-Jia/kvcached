# SPDX-License-Identifier: Apache-2.0
"""Bounded allocator handoff after a rolled-back physical capacity rejection.

The serving adapter supplies the allocator callback. This module knows neither
model layouts nor torch, and never evicts live KV or changes the capacity limit.
"""


def is_capacity_rejection(error):
    # Existing VMM backend contract. Do not retry arbitrary driver failures,
    # failed synchronization, quarantined mappings, or Python MemoryError.
    return isinstance(error, RuntimeError) and str(error).startswith('capacity_exhausted:')


def ensure_with_allocator_relief(residency, desired, release_unused, snapshot, report):
    try:
        return residency.ensure(desired)
    except RuntimeError as error:
        if residency.poisoned or not is_capacity_rejection(error):
            raise
        rejection = str(error)
    # ensure() rolled back only its newly added pages. Fence device execution
    # before giving unused allocator arenas back to the same driver.
    before = snapshot()
    try:
        residency.synchronize()
        release_unused()
    except Exception:
        residency.poisoned = True
        raise
    after = snapshot()
    report({'reason': rejection, 'before': before, 'after': after})
    # Exactly one retry. Continued pressure remains a rejection, not a busy
    # loop or permission to discard content / lower the safety margin.
    return residency.ensure(desired)
