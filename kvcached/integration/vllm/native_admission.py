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
import threading
import time
from copy import deepcopy
from dataclasses import replace
from functools import wraps
from itertools import islice
from types import SimpleNamespace

from kvcached.native_capacity import NativeBudget, all_rank_success
from kvcached.integration.vllm.native_residency import snapshot_ownership


class _AdmissionMiss(Exception):
    """Return native allocate_slots(None) before it allocates any new slots."""


class NativeAdmission:
    def __init__(self, manager, decode_group, rpc, ranks, budget, *, in_process=False):
        self.manager = manager
        self.coordinator = manager.coordinator
        self.pool = self.coordinator.block_pool
        self.decode_group = decode_group
        self.rpc = rpc
        self.ranks = tuple(ranks)
        self.budget = budget
        self.in_process = bool(in_process)
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
        self.step_grants_enabled = os.getenv('KVCACHED_NATIVE_STEP_GRANTS') == '1'
        self.profile_enabled = os.getenv('KVCACHED_NATIVE_PROFILE') == '1'
        self.profile = {}
        self.counters = dict(rpc_calls=0, grant_hits=0, batch_attempts=0,
                             batch_fallbacks=0, snapshots=0, snapshot_hits=0,
                             prefix_scans=0, candidate_queue_scans=0)
        # The scheduler asks for capacity several times per step, and every
        # admission call needs the protected owner snapshot. The owner set only
        # changes when blocks are allocated or prefixes are evicted, so one
        # snapshot per such event is enough.
        self._snapshot_cache = None
        self._protected_cache = None
        self._grant = None
        self._step_warm = set()
        self._warm_attempted = False
        # Cross-step prepare pipeline: map the free-queue head for the next
        # step on a background thread while the engine executes the current
        # one. Map-only, so the worker never fences or retires here.
        self.pipeline_prepare_enabled = (
            os.getenv('KVCACHED_NATIVE_PIPELINE_PREPARE') == '1'
            and self.in_process)
        try:
            self.pipeline_blocks = max(0, int(os.getenv(
                'KVCACHED_NATIVE_PREFETCH_BLOCKS', '8')))
        except ValueError:
            self.pipeline_blocks = 8
        self.counters.setdefault('prefetch_calls', 0)
        self.counters.setdefault('prefetch_errors', 0)
        self._prefetch_ticket = 0
        self._prefetch_thread = None
        self._prefetch_stop = threading.Event()
        self._prefetch_wake = threading.Event()
        self._prefetch_lock = threading.Lock()
        self._prefetch_pending = None
        self._install()

    def _timed(self, name, function, *args, **kwargs):
        if not self.profile_enabled:
            return function(*args, **kwargs)
        start = time.perf_counter_ns()
        try:
            return function(*args, **kwargs)
        finally:
            row = self.profile.setdefault(name, dict(calls=0, host_ns=0))
            row['calls'] += 1
            row['host_ns'] += time.perf_counter_ns() - start

    def _invalidate_snapshot(self):
        """Drop the per-step snapshot after any ownership-visible mutation."""
        self._snapshot_cache = None
        self._protected_cache = None

    def _snapshot(self):
        epoch = max(1, self.epoch)
        cached = self._snapshot_cache
        if cached is not None and cached.epoch == epoch:
            self.counters['snapshot_hits'] += 1
            return cached
        self.counters['snapshots'] += 1
        snapshot = self._timed('snapshot', snapshot_ownership,
            self.coordinator,
            SimpleNamespace(new_block_ids_to_zero=(), kv_cache_block_copies=()),
            epoch=epoch,
            decode_group=self.decode_group,
        )
        self._snapshot_cache = snapshot
        return snapshot

    def _protected_snapshot(self):
        cached = self._protected_cache
        if cached is not None and cached.epoch == max(1, self.epoch):
            return cached
        snapshot = self._snapshot()
        # A native allocation can stage a copy and release its source before
        # SchedulerOutput publishes copy endpoints. Preserve the start-of-step
        # non-prefix owners until that step finishes. Prefix hits are separately
        # protected as admission candidates before their reference counts change.
        owners = {
            key: tuple(sorted(set(value) | self.step_base.get(key, set())))
            for key, value in snapshot.owners.items()
        }
        protected = replace(snapshot, owners=owners)
        self._protected_cache = protected
        return protected

    @staticmethod
    def _coverage(snapshot, candidates):
        # This is a physical-readiness proof, not native prefix validity.
        # Every candidate is prepared in every group by the worker contract.
        owners = set().union(*snapshot.owners.values())
        owners.update((group, block) for group in range(snapshot.num_groups)
                      for block in candidates)
        return frozenset(owners)

    def _prepare(self, candidates, *, release_spares=False, snapshot=None):
        return self._timed('prepare', self._prepare_impl, candidates,
                           release_spares=release_spares, snapshot=snapshot)

    def _prepare_impl(self, candidates, *, release_spares=False, snapshot=None):
        if snapshot is None:
            snapshot = self._protected_snapshot()
        scope = (snapshot.version, snapshot.epoch, snapshot.num_groups,
                 snapshot.num_blocks, self.budget)
        coverage = None
        if self.step_grants_enabled and self.in_schedule and not release_spares:
            coverage = self._coverage(snapshot, candidates)
            if (self._grant is not None and self._grant[0] == scope
                    and coverage <= self._grant[1]):
                self.counters['grant_hits'] += 1
                return True
        # Any worker call can retire pages or partially fail. Never keep the
        # old proof across it, even when response validation raises.
        self._grant = None
        self.ticket += 1
        options = {'kwargs': {'release_spares': True}} if release_spares and self.spare_enabled else {}
        self.counters['rpc_calls'] += 1
        rows = self._timed('rpc', self.rpc,
            "kvcached_native_prepare",
            args=(
                snapshot,
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
        elif coverage is not None:
            # Valid only while this serialized scheduler round owns control.
            # No forward, dummy execution or external maintenance can run
            # between these allocations; end_schedule invalidates the proof.
            self._grant = (scope, coverage)
        return success

    def _eligible_prefixes(self, exclude=()):
        self.counters['prefix_scans'] += 1
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
        cap = self.budget.prefix_max_blocks
        if not pressure and cap is None:
            return 0
        eligible = self._timed('prefix_scan', self._eligible_prefixes, exclude)
        count = len(eligible) if pressure else (0 if cap is None else max(0, len(eligible) - cap))
        # Use the engine's eviction API so all hash aliases, metrics and events
        # change together. Never discard native reuse watchers or active blocks.
        selected = {block.block_id for block in eligible[:count]}
        if selected:
            self.pool.evict_blocks(selected)
            self.evicted_prefix_blocks += len(selected)
            # Prefix hashes are owners; the cached snapshot no longer matches.
            self._invalidate_snapshot()
        return len(selected)

    def begin_step(self, epoch):
        if self.in_schedule:
            raise RuntimeError("Native admission does not support queued scheduling")
        self.epoch = epoch
        self._invalidate_snapshot()
        self._grant = None
        self._step_warm.clear()
        self._warm_attempted = False
        self.step_candidates.clear()
        self.step_base = {}
        self.evict_prefixes()
        snapshot = self._snapshot()
        self.step_base = {
            key: set(value) for key, value in snapshot.owners.items() if key != "prefix"
        }
        self.in_schedule = True

    def end_schedule(self):
        self._grant = None
        self._step_warm.clear()
        self.in_schedule = False
        self.context = None

    def try_admit(self, candidates, prefix_hits=()):
        previous = set(self.step_candidates)
        wanted = previous | set(candidates) | set(prefix_hits)
        snapshot = self._protected_snapshot()
        if self.step_grants_enabled and not self._warm_attempted:
            self._warm_attempted = True
            # Batch only existing running-request copy-source candidates.
            # No guessed future slots and no change to native allocation.
            warm = {block for _, block in snapshot.owners['requests']}
            if warm - wanted:
                self.counters['batch_attempts'] += 1
                if self._prepare(wanted | warm, snapshot=snapshot):
                    self._step_warm = warm
                    self.step_candidates = wanted
                    return True
                self.counters['batch_fallbacks'] += 1
                # An oversized batch must not prevent a smaller request from
                # running. Roll back all ranks before the original admission.
                if not self._prepare(previous, release_spares=True, snapshot=snapshot):
                    raise RuntimeError('Unable to roll back native step preparation')
        if self._prepare(wanted | self._step_warm, snapshot=snapshot):
            self.step_candidates = wanted
            return True
        if self._step_warm:
            self.counters['batch_fallbacks'] += 1
            self._step_warm.clear()
            if not self._prepare(previous, release_spares=True, snapshot=snapshot):
                raise RuntimeError('Unable to drop native step preparation')
            if self._prepare(wanted, snapshot=snapshot):
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
        self._grant = None
        self._step_warm.clear()
        self.step_base = {}
        self._invalidate_snapshot()
        self.evict_prefixes(pressure=pressure)
        if not self._prepare(set()):
            raise RuntimeError("Existing native owners lost their backing")
        return self.status()

    def set_budget(self, *, limit_bytes, prefix_max_blocks, expected_revision):
        if self.in_schedule or self.context is not None:
            raise RuntimeError('Cannot change native budget during allocation')
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
            step_grants_enabled=self.step_grants_enabled,
            admission_counters=dict(self.counters),
            admission_profile=deepcopy(self.profile),
        )

    def pipeline_prepare(self):
        """Arm a map-only prepare for the next step and return immediately."""
        if not self.pipeline_prepare_enabled or self.pipeline_blocks <= 0:
            return 0
        try:
            free = self.pool.free_block_queue.get_all_free_blocks()
        except Exception:  # a queue change during a step is not an error
            return 0
        candidates = []
        for block in free:
            if block.is_null or block.ref_cnt:
                continue
            candidates.append(block.block_id)
            if len(candidates) >= self.pipeline_blocks:
                break
        if not candidates:
            return 0
        self._prefetch_ticket += 1
        job = (self._prefetch_ticket, tuple(candidates), self.budget.as_dict())
        with self._prefetch_lock:
            # Latest step wins; a stale prediction is worthless.
            self._prefetch_pending = job
        if self._prefetch_thread is None:
            self._prefetch_thread = threading.Thread(
                target=self._prefetch_loop, name='native-prepare-pipeline',
                daemon=True)
            self._prefetch_thread.start()
        self._prefetch_wake.set()
        return len(candidates)

    def _prefetch_loop(self):
        while not self._prefetch_stop.is_set():
            self._prefetch_wake.wait(0.5)
            self._prefetch_wake.clear()
            with self._prefetch_lock:
                job = self._prefetch_pending
                self._prefetch_pending = None
            if job is None:
                continue
            ticket, candidates, budget = job
            try:
                self.counters['prefetch_calls'] += 1
                self.rpc('kvcached_native_prefetch', args=(candidates, budget, ticket))
            except Exception:  # speculative path never fails the engine
                self.counters['prefetch_errors'] += 1

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
            # Prefix touch removes its blocks before get_new_blocks. Exclude
            # those IDs when predicting the free queue after touch. Demand may
            # include those hits; retaining a few extra candidates is safe.
            candidates = []
            if needed:
                self.counters['candidate_queue_scans'] += 1
                free = pool.free_block_queue.get_all_free_blocks()
                candidates = list(islice(
                    (block.block_id for block in free if block.block_id not in hits), needed))
            if self.try_admit(candidates, hits | context["sources"]):
                return logical
            # Zero demand can still touch a cached block in a different native
            # component. Returning zero would incorrectly pass a 0 <= 0 check.
            raise _AdmissionMiss

        @wraps(original_new)
        def get_new(count):
            if self.context is not None:
                actual = [] if count == 0 else [
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
                # Allocated blocks change request ownership and refcounts.
                self._invalidate_snapshot()

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
