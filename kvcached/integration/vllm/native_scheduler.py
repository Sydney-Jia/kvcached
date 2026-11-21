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
        if (admission is None and native_gate.installed()
                and not output.total_num_scheduled_tokens
                and native_gate.pressure()):
            # Gated elastic mode maps only what a step requires, so retirement
            # waits for an empty scheduling point where its fence costs nothing.
            # Cold prefixes go first: their pages are owners until evicted.
            self._release_cold_prefix()
            native_gate.reclaim_idle()
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
