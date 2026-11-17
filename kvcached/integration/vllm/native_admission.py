# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Synchronous native-block admission, independent of model/cache-spec classes.

Native managers still choose slots and implement preemption. Before they mutate
the allocation, all workers must back the candidate slots. Candidates are kept
in *every* group because native copy/zero operations may use every component.
This conservative first implementation prioritizes safety over packing density.
"""

import inspect
import os
from copy import deepcopy
from dataclasses import replace
from functools import wraps
from types import SimpleNamespace

from kvcached.native_capacity import NativeBudget, all_rank_success
from kvcached.integration.vllm.native_residency import snapshot_ownership


class _AdmissionMiss(Exception):
    """Return native allocate_slots(None) before it allocates any new slots."""


class NativeAdmission:
    def __init__(self, manager, decode_group, rpc, ranks, budget):
        self.manager = manager
        self.coordinator = manager.coordinator
        self.pool = self.coordinator.block_pool
        self.decode_group = decode_group
        self.rpc = rpc
        self.ranks = tuple(ranks)
        self.budget = budget
        self.ticket = 0
        self.epoch = 0
        self.step_candidates = set()
        self.step_base = {}
        self.in_schedule = False
        self.context = None
        self.inside_get_new = False
        self.last_rows = []
        self.last_failed_prepare = None
        self.denied_allocations = 0
        self.evicted_prefix_blocks = 0
        self.spare_enabled = int(os.getenv('KVCACHED_NATIVE_SPARE_BYTES', '0')) > 0
        self._install()

    def _snapshot(self):
        return snapshot_ownership(
            self.coordinator,
            SimpleNamespace(new_block_ids_to_zero=(), kv_cache_block_copies=()),
            epoch=max(1, self.epoch),
            decode_group=self.decode_group,
        )

    def _protected_snapshot(self):
        snapshot = self._snapshot()
        # A native allocation can stage a copy and release its source before
        # SchedulerOutput publishes copy endpoints. Preserve the start-of-step
        # non-prefix owners until that step finishes. Prefix hits are separately
        # protected as admission candidates before their reference counts change.
        owners = {
            key: tuple(sorted(set(value) | self.step_base.get(key, set())))
            for key, value in snapshot.owners.items()
        }
        return replace(snapshot, owners=owners)

    def _prepare(self, candidates, *, release_spares=False):
        self.ticket += 1
        options = {'kwargs': {'release_spares': True}} if release_spares and self.spare_enabled else {}
        rows = self.rpc(
            "kvcached_native_prepare",
            args=(
                self._protected_snapshot(),
                tuple(sorted(candidates)),
                self.budget.as_dict(),
                self.ticket,
            ),
            **options,
        )
        success = all_rank_success(rows, self.ranks)
        self.last_rows = rows
        if any(row.get("fatal") for row in rows):
            raise RuntimeError(f"Native capacity worker requires restart: {rows}")
        if any(
            row.get("ticket") != self.ticket
            or row.get("budget", {}).get("revision") != self.budget.revision
            for row in rows
        ):
            raise RuntimeError("Stale native capacity response")
        if not success:
            # Rollback/reconcile replaces last_rows with a successful current
            # observation. Retain the bounded failure record separately so a
            # caller can distinguish per-rank physical rejection from quota
            # rejection, without treating old rows as the current grant.
            self.last_failed_prepare = dict(
                ticket=self.ticket,
                revision=self.budget.revision,
                candidate_count=len(candidates),
                ranks=deepcopy(rows),
            )
        return success

    def _eligible_prefixes(self, exclude=()):
        exclude = set(exclude)
        aliases = self.pool.cached_block_hashes_by_block
        return [
            block
            for block in self.pool.free_block_queue.get_all_free_blocks()
            if block.block_id not in exclude
            and not block.is_null
            and block.ref_cnt == 0
            and block.block_id not in self.pool._reuse_watchers
            and (block.block_hash is not None or aliases.get(block.block_id))
        ]

    def evict_prefixes(self, *, pressure=False, exclude=()):
        eligible = self._eligible_prefixes(exclude)
        cap = self.budget.prefix_max_blocks
        count = len(eligible) if pressure else (0 if cap is None else max(0, len(eligible) - cap))
        # Use the engine's eviction API so all hash aliases, metrics and events
        # change together. Never discard native reuse watchers or active blocks.
        selected = {block.block_id for block in eligible[:count]}
        if selected:
            self.pool.evict_blocks(selected)
            self.evicted_prefix_blocks += len(selected)
        return len(selected)

    def begin_step(self, epoch):
        if self.in_schedule:
            raise RuntimeError("Native admission does not support queued scheduling")
        self.epoch = epoch
        self.step_candidates.clear()
        self.step_base = {}
        self.evict_prefixes()
        snapshot = self._snapshot()
        self.step_base = {
            key: set(value) for key, value in snapshot.owners.items() if key != "prefix"
        }
        self.in_schedule = True

    def end_schedule(self):
        self.in_schedule = False
        self.context = None

    def try_admit(self, candidates, prefix_hits=()):
        previous = set(self.step_candidates)
        wanted = previous | set(candidates) | set(prefix_hits)
        if self._prepare(wanted):
            self.step_candidates = wanted
            return True
        # Idle prefixes can otherwise keep all reclaimable pages mapped. A
        # pressure retry invalidates only cold entries, in native LRU order.
        # This initial policy evicts all eligible cold entries on a hard miss;
        # it deliberately does not claim an optimal hit-rate tradeoff.
        if self.evict_prefixes(pressure=True, exclude=wanted):
            if self._prepare(wanted):
                self.step_candidates = wanted
                return True
        # A TP grant is not atomic. Drop speculative pages on *every* rank
        # before reporting no capacity to the native scheduler.
        if not self._prepare(previous, release_spares=True):
            raise RuntimeError("Unable to preserve existing state after admission rejection")
        self.denied_allocations += 1
        return False

    def reconcile(self, *, pressure=False):
        if self.in_schedule or self.context is not None:
            raise RuntimeError("Cannot reconcile during native allocation")
        self.step_candidates.clear()
        self.step_base = {}
        self.evict_prefixes(pressure=pressure)
        if not self._prepare(set()):
            raise RuntimeError("Existing native owners lost their backing")
        return self.status()

    def set_budget(self, *, limit_bytes, prefix_max_blocks, expected_revision):
        if type(expected_revision) is not int or expected_revision != self.budget.revision:
            raise ValueError("Native budget revision conflict")
        candidate = NativeBudget(self.budget.revision + 1, limit_bytes, prefix_max_blocks)
        previous_limit = self.budget.limit_bytes
        self.budget = candidate
        # A smaller target is acknowledged independently of convergence: live
        # state is retained, while idle prefixes may be invalidated immediately.
        shrinking = limit_bytes is not None and (
            previous_limit is None or limit_bytes < previous_limit
        )
        return self.reconcile(pressure=shrinking)

    def status(self):
        converged = bool(self.last_rows) and all(
            row.get("budget", {}).get("convergence") == "converged"
            and row.get("budget", {}).get("revision") == self.budget.revision
            for row in self.last_rows
        )
        return dict(
            **self.budget.as_dict(),
            convergence="converged" if converged else "pending",
            denied_allocations=self.denied_allocations,
            evicted_prefix_blocks=self.evicted_prefix_blocks,
            expected_ranks=list(self.ranks),
            observed_ticket=self.ticket,
            ranks=self.last_rows,
            last_failed_prepare=deepcopy(self.last_failed_prepare),
        )

    def _install(self):
        pool, manager, coordinator = self.pool, self.manager, self.coordinator
        for name in ("evict_blocks", "get_new_blocks", "get_num_free_blocks"):
            if not callable(getattr(pool, name, None)):
                raise ValueError(f"Native capacity requires BlockPool.{name}")
        self.original_free = pool.get_num_free_blocks
        original_new = pool.get_new_blocks
        original_demand = coordinator.get_num_blocks_to_allocate
        original_allocate = manager.allocate_slots
        signature = inspect.signature(original_allocate)
        if (
            not {"new_computed_blocks", "reserved_blocks", "has_scheduled_reqs"}
            <= signature.parameters.keys()
        ):
            raise ValueError("Native allocate_slots contract changed")

        @wraps(original_demand)
        def demand(*args, **kwargs):
            count = original_demand(*args, **kwargs)
            if type(count) is not int or count < 0:
                raise ValueError("Native block demand must be a nonnegative integer")
            if self.context is not None:
                self.context["demand"] = count
            return count

        @wraps(self.original_free)
        def available():
            logical = self.original_free()
            context = self.context
            if context is None or self.inside_get_new:
                return logical
            if context["demand"] is None:
                raise RuntimeError("Native allocation queried capacity before computing demand")
            needed = context["demand"] + context["headroom"]
            if needed > logical:
                return logical
            hits = context["hits"]
            free = pool.free_block_queue.get_all_free_blocks()
            # Prefix touch removes its blocks before get_new_blocks. Exclude
            # those IDs when predicting the free queue after touch. Demand may
            # include those hits; retaining a few extra candidates is safe.
            candidates = [block.block_id for block in free if block.block_id not in hits][:needed]
            if self.try_admit(candidates, hits | context["sources"]):
                return logical
            # Zero demand can still touch a cached block in a different native
            # component. Returning zero would incorrectly pass a 0 <= 0 check.
            raise _AdmissionMiss

        @wraps(original_new)
        def get_new(count):
            if self.context is not None:
                actual = [
                    block.block_id for block in pool.free_block_queue.get_all_free_blocks()[:count]
                ]
                if not set(actual) <= self.step_candidates:
                    # Fail before mutating the pool. A changed native allocation
                    # order needs an adapter update, not an unsafe late mapping.
                    raise RuntimeError("Native allocator selected unprepared slots")
            else:
                raise RuntimeError("Native allocation bypassed allocate_slots admission")
            self.inside_get_new = True
            try:
                return original_new(count)
            finally:
                self.inside_get_new = False

        @wraps(original_allocate)
        def allocate(*args, **kwargs):
            if self.context is not None or not self.in_schedule:
                raise RuntimeError("Native capacity requires serialized scheduler allocation")
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            values = bound.arguments
            computed = values["new_computed_blocks"]
            hits = (
                set()
                if computed is None
                else {
                    block.block_id
                    for group in computed.blocks
                    for block in group
                    if not block.is_null
                }
            )
            # Native CoW can move a running request's partial tail into a new
            # slot, and its copy kernel addresses all groups. The current
            # protocol has no typed "possible copy sources" preview, so retain
            # this request's existing IDs in every group before allocation.
            # No model/spec class is used to guess which manager may copy.
            request_id = values["request"].request_id
            sources = {
                block.block_id
                for native in coordinator.single_type_managers
                for block in native.req_to_blocks.get(request_id, ())
                if not block.is_null
            }
            # Conservative headroom: the native manager applies its watermark
            # only to some admissions; reserving it here may under-admit but
            # cannot expose an unbacked slot.
            headroom = values["reserved_blocks"]
            if values["has_scheduled_reqs"]:
                headroom += manager.watermark_blocks
            self.context = dict(demand=None, hits=hits, sources=sources, headroom=headroom)
            try:
                return original_allocate(*args, **kwargs)
            except _AdmissionMiss:
                return None
            finally:
                self.context = None

        coordinator.get_num_blocks_to_allocate = demand
        pool.get_num_free_blocks = available
        pool.get_new_blocks = get_new
        manager.allocate_slots = allocate
