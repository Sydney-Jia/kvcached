# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Native vLLM scheduler with an explicit, versioned residency side channel."""

import os
import time

from vllm.v1.core.kv_cache_utils import get_group_id
from vllm.v1.core.sched.scheduler import Scheduler

from kvcached.integration.vllm import native_gate
from kvcached.integration.vllm.native_residency import snapshot_ownership


class NativeResidencyScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kvcached_epoch = 0
        self._native_admission = None
        # Gate retirement runs in the worker processes. When those are the same
        # process (in-process TP1 executor) the scheduler calls the arena
        # directly; otherwise it asks every rank once per idle/pressure point.
        self._gate_rpc = None
        self._gate_pressure = False
        try:
            # Busy-mode probe period in scheduling epochs. It must not depend
            # on the previous sample, or pressure that rises mid-decode is
            # never re-examined until the next empty step.
            self._gate_probe_epochs = max(1, int(os.getenv(
                'KVCACHED_NATIVE_GATE_PROBE_EPOCHS', '16')))
        except ValueError:
            self._gate_probe_epochs = 16
        if (os.getenv('KVCACHED_NATIVE_GATE') == '1'
                and os.getenv('KVCACHED_NATIVE_CAPACITY') != '1'):
            from kvcached.integration.vllm.native_executor import executor_for
            try:
                executor = executor_for(self.vllm_config)
            except ValueError:
                executor = None
            if executor is not None and not getattr(executor, 'in_process', False):
                self._gate_rpc = executor.collective_rpc
            elif executor is None and self.vllm_config.parallel_config.world_size > 1:
                # Without a native executor the scheduler cannot reach the
                # worker arenas, and the gate would silently stop retiring.
                raise ValueError(
                    "KVCACHED_NATIVE_GATE with TP>1 requires the native capacity executor")
        if os.getenv('KVCACHED_NATIVE_CAPACITY') == '1':
            from kvcached.native_capacity import NativeBudget
            from kvcached.integration.vllm.native_admission import NativeAdmission
            from kvcached.integration.vllm.native_executor import executor_for

            executor = executor_for(self.vllm_config)
            limit = os.getenv('KVCACHED_NATIVE_LIMIT_BYTES')
            prefix = os.getenv('KVCACHED_NATIVE_PREFIX_BLOCKS')
            budget = NativeBudget(0, None if limit is None else int(limit),
                                  None if prefix is None else int(prefix))
            self._native_admission = NativeAdmission(
                self.kv_cache_manager, get_group_id, executor.collective_rpc,
                range(executor.world_size), budget,
                in_process=getattr(executor, 'in_process', False))
            executor.native_admission = self._native_admission
            self._native_admission.reconcile()

    def schedule(self, *args, **kwargs):
        admission = self._native_admission
        denied_before = admission.denied_allocations if admission is not None else 0
        if admission is not None:
            admission.begin_step(self._kvcached_epoch + 1)
        try:
            output = super().schedule(*args, **kwargs)
        finally:
            if admission is not None:
                admission.end_schedule()
        if admission is not None:
            # Predicted map-only work for the next step runs on the admission's
            # prefetch thread while this step executes; it never blocks here.
            admission.pipeline_prepare()
        if admission is None:
            idle = not output.total_num_scheduled_tokens
            if self._gate_rpc is not None:
                # Multi-process ranks. An empty step retires unowned pages,
                # which fences; a busy step only samples pressure, because a
                # fence mid-decode is the cost the gate exists to avoid.
                if idle:
                    self._gate_idle()
                    if self._gate_pressure:
                        # Evict only after a rank actually reports pressure;
                        # retirement above may already have brought it down.
                        self._release_cold_prefix()
                elif self._kvcached_epoch % self._gate_probe_epochs == 0:
                    self._gate_status()
                    if self._gate_pressure:
                        self._release_cold_prefix()
            elif native_gate.installed():
                if idle:
                    # Gated elastic mode maps only what a step requires, so
                    # retirement waits for an empty scheduling point where its
                    # fence costs nothing. Never evict on the assumption of
                    # pressure: retiring unowned pages may be enough, and each
                    # cold prefix eviction costs a recompute later.
                    if native_gate.pressure():
                        native_gate.reclaim_idle()
                    if native_gate.pressure():
                        # Cold prefixes still own their pages at this point.
                        self._release_cold_prefix()
                elif native_gate.pressure(0.9):
                    # Near the wall with no idle point: evicting cold prefixes
                    # is fence-free, and the worker's rejection path reclaims.
                    self._release_cold_prefix()
        self._kvcached_epoch += 1
        # The pinned multiprocess executor pickles SchedulerOutput, including
        # this side channel. Workers reject a missing/stale side channel rather
        # than continuing with incomplete ownership after transport changes.
        output.kvcached_ownership = snapshot_ownership(
            self.kv_cache_manager.coordinator, output,
            epoch=self._kvcached_epoch, decode_group=get_group_id)
        if (admission is not None and not output.total_num_scheduled_tokens
                and admission.denied_allocations > denied_before):
            # The engine still processes cancellation/control between steps.
            # Avoid a hot RPC loop while a peer occupies the remaining memory.
            time.sleep(0.01)
        return output

    def _gate_rows(self, action):
        """One collective gate call; an unreachable rank is reported, not fatal."""
        try:
            rows = self._gate_rpc('kvcached_native_gate',
                                  kwargs={'action': action})
        except Exception as error:
            print('KVCACHED_NATIVE_GATE_FAILED ' + repr(error)[:160], flush=True)
            return []
        return [row for row in (rows or []) if isinstance(row, dict)]

    def _gate_status(self):
        """Sample per-rank pressure only; never fences and never retires."""
        self._gate_pressure = any(
            row.get('pressure') for row in self._gate_rows('status'))

    def _gate_idle(self):
        """Ask every rank to retire unowned pages if its ledger is tight."""
        rows = self._gate_rows('idle')
        self._gate_pressure = any(row.get('pressure') for row in rows)
        return sum(int(row.get('reclaimed') or 0) for row in rows)

    def _release_cold_prefix(self):
        """Evict the oldest idle prefix blocks while the ledger is tight.

        Only called at an empty scheduling point; the pages become unowned, so
        the idle retirement that follows can release them. Never touches active
        blocks, reuse watchers or hash aliases.
        """
        try:
            limit = int(os.getenv('KVCACHED_NATIVE_IDLE_PREFIX_EVICT', '8'))
        except ValueError:
            return 0
        if limit <= 0:
            return 0
        pool = self.kv_cache_manager.coordinator.block_pool
        aliases = pool.cached_block_hashes_by_block
        queue = pool.free_block_queue
        head = getattr(queue, 'fake_free_list_head', None)
        tail = getattr(queue, 'fake_free_list_tail', None)
        if head is None or tail is None:
            nodes = queue.get_all_free_blocks()
        else:
            nodes = []
            node = head.next_free_block
            while node is not None and node is not tail:
                nodes.append(node)
                node = node.next_free_block
        selected = set()
        for node in nodes:
            if (not node.is_null and node.ref_cnt == 0
                    and node.block_id not in pool._reuse_watchers
                    and (node.block_hash is not None
                         or aliases.get(node.block_id))):
                selected.add(node.block_id)
                if len(selected) >= limit:
                    break
        if not selected:
            return 0
        pool.evict_blocks(selected)
        return len(selected)

    def update_from_output(self, *args, **kwargs):
        result = super().update_from_output(*args, **kwargs)
        if self._native_admission is not None:
            # Includes the last request: idle engines must return cold capacity
            # without needing another inference request to trigger maintenance.
            self._native_admission.reconcile()
        return result

    def finish_requests(self, *args, **kwargs):
        result = super().finish_requests(*args, **kwargs)
        admission = getattr(self, '_native_admission', None)
        if admission is not None:
            admission.reconcile()
        return result
