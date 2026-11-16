# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Device-independent contracts for the native adapter's physical KV budget.

Logical capacity is immutable. Limits count mapped VMM bytes (rounded pages),
not tokens, torch allocator reservations, weights, or whole-device free memory.
Limits are soft for existing live state and hard for additional physical growth.
Only the framework may invalidate a reusable prefix; this layer never does so.
"""

from dataclasses import dataclass


def optional_nonnegative(value, name):
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError(f"{name} must be a nonnegative integer or None")
    return value


@dataclass(frozen=True)
class NativeBudget:
    revision: int = 0
    limit_bytes: int | None = None
    prefix_max_blocks: int | None = None

    def __post_init__(self):
        optional_nonnegative(self.revision, "revision")
        if self.revision is None:
            raise ValueError("revision must be an integer")
        optional_nonnegative(self.limit_bytes, "limit_bytes")
        optional_nonnegative(self.prefix_max_blocks, "prefix_max_blocks")

    def as_dict(self):
        return dict(
            revision=self.revision,
            limit_bytes=self.limit_bytes,
            prefix_max_blocks=self.prefix_max_blocks,
        )


def page_bytes(storages, pages):
    if pages.keys() != storages.keys():
        raise ValueError("Capacity requires a complete storage-page snapshot")
    return sum(len(offsets) * storages[key].page_bytes for key, offsets in pages.items())


class NativeBudgetState:
    """Revisioned per-worker policy; publication is distinct from convergence."""

    def __init__(self, budget=None):
        self.budget = budget or NativeBudget()
        self.rejections = 0

    def apply(self, budget):
        if budget.revision < self.budget.revision:
            raise ValueError("Stale native budget revision")
        if budget.revision == self.budget.revision and budget != self.budget:
            raise ValueError("Conflicting native budget at the same revision")
        self.budget = budget

    def permits(self, storages, mapped, required):
        size = page_bytes(storages, required)
        limit = self.budget.limit_bytes
        # A reduced limit must not discard live state or stop an already-backed
        # decode. It may converge when owners retire. New pages are not granted
        # above the target even if some other old pages could be discarded.
        already_backed = all(required[key] <= mapped[key] for key in required)
        allowed = limit is None or size <= limit or already_backed
        if not allowed:
            self.rejections += 1
        return allowed

    def status(self, storages, mapped, required):
        actual = page_bytes(storages, mapped)
        live = page_bytes(storages, required)
        limit = self.budget.limit_bytes
        return dict(
            **self.budget.as_dict(),
            mapped_bytes=actual,
            required_bytes=live,
            reclaimable_bytes=sum(
                len(mapped[key] - required[key]) * storages[key].page_bytes for key in mapped
            ),
            convergence="converged" if limit is None or actual <= limit else "pending",
            pending_reason=None if limit is None or actual <= limit else "retained_state",
            capacity_rejections=self.rejections,
        )


def all_rank_success(rows, expected_ranks):
    """Do not mistake a single rank reply or duplicated rank for a TP grant."""
    expected = set(expected_ranks)
    if (
        not isinstance(rows, list)
        or len(rows) != len(expected)
        or any(not isinstance(row, dict) for row in rows)
    ):
        raise RuntimeError("Incomplete native capacity response")
    ranks = [row.get("rank") for row in rows]
    if set(ranks) != expected or len(set(ranks)) != len(ranks):
        raise RuntimeError("Native capacity response rank mismatch")
    if any(type(row.get("ok")) is not bool for row in rows):
        raise RuntimeError("Native capacity response lacks explicit success")
    return all(row["ok"] for row in rows)
