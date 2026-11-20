# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Native vLLM scheduler with an explicit, versioned residency side channel."""

import os
import time

from vllm.v1.core.kv_cache_utils import get_group_id
from vllm.v1.core.sched.scheduler import Scheduler

from kvcached.integration.vllm.native_residency import (
    NativeOwnership,
    ownership_signature,
    snapshot_ownership,
)


class NativeResidencyScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kvcached_epoch = 0
        self._kvcached_last_signature = None
        self._kvcached_last_owners = None
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
        self._kvcached_epoch += 1
        # The pinned multiprocess executor pickles SchedulerOutput, including
        # this side channel. Workers reject a missing/stale side channel rather
        # than continuing with incomplete ownership after transport changes.
        coordinator = self.kv_cache_manager.coordinator
        signature = ownership_signature(coordinator, output)
        if (signature == self._kvcached_last_signature
                and self._kvcached_last_owners is not None):
            # Two decode steps with an identical pool and request map: reuse
            # the immutable owner tuples, only the epoch advances.
            output.kvcached_ownership = NativeOwnership(
                version=1, epoch=self._kvcached_epoch,
                num_blocks=len(coordinator.block_pool.blocks),
                num_groups=len(coordinator.single_type_managers),
                owners=self._kvcached_last_owners)
        else:
            ownership = snapshot_ownership(
                coordinator, output, epoch=self._kvcached_epoch,
                decode_group=get_group_id)
            self._kvcached_last_signature = signature
            self._kvcached_last_owners = ownership.owners
            output.kvcached_ownership = ownership
        if (admission is not None and not output.total_num_scheduled_tokens
                and admission.denied_allocations > denied_before):
            # The engine still processes cancellation/control between steps.
            # Avoid a hot RPC loop while a peer occupies the remaining memory.
            time.sleep(0.01)
        return output

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
