# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Native vLLM scheduler with an explicit, versioned residency side channel."""

import time

from vllm.v1.core.kv_cache_utils import get_group_id
from vllm.v1.core.sched.scheduler import Scheduler

from kvcached.integration.vllm import native_profile
from kvcached.integration.vllm.native_residency import (
    NativeOwnership,
    ownership_signature,
    snapshot_ownership,
)
from kvcached.integration.vllm.native_admission import (
    install_scheduler_context,
    note_progress,
    patch_allocate_slots,
    patch_block_pool_admission,
)


class NativeResidencyScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kvcached_epoch = 0
        self._kvcached_last_signature = None
        self._kvcached_last_owners = None
        # The worker installs the physical gate during engine initialization;
        # allocate_slots consults it before committing a request.
        patch_allocate_slots()
        patch_block_pool_admission()
        install_scheduler_context(self.kv_cache_manager.coordinator, self)

    def schedule(self, *args, **kwargs):
        schedule_started = time.perf_counter()
        output = super().schedule(*args, **kwargs)
        native_profile.record(
            "scheduler.super_schedule", time.perf_counter() - schedule_started)
        # Idle prefix release runs on the gate's dedicated idle thread only:
        # doing driver release work here blocks the engine loop behind the
        # physical lock and turns a slow unmap into a stalled scheduler.
        self._kvcached_epoch += 1
        # The pinned multiprocess executor pickles SchedulerOutput, including
        # this side channel. Workers reject a missing/stale side channel rather
        # than continuing with incomplete ownership after transport changes.
        started = time.perf_counter()
        coordinator = self.kv_cache_manager.coordinator
        signature = ownership_signature(coordinator, output)
        if (self._kvcached_last_signature == signature
                and self._kvcached_last_owners is not None):
            managers = tuple(coordinator.single_type_managers)
            ownership = NativeOwnership(
                version=1, epoch=self._kvcached_epoch,
                num_blocks=len(coordinator.block_pool.blocks),
                num_groups=len(managers),
                owners=self._kvcached_last_owners)
            native_profile.record(
                "scheduler.snapshot_reuse", time.perf_counter() - started,
                last_n=sum(len(v) for v in ownership.owners.values()))
        else:
            ownership = snapshot_ownership(
                coordinator, output,
                epoch=self._kvcached_epoch, decode_group=get_group_id)
            self._kvcached_last_signature = signature
            self._kvcached_last_owners = ownership.owners
            native_profile.record(
                "scheduler.snapshot", time.perf_counter() - started,
                last_n=sum(len(v) for v in ownership.owners.values()))
        output.kvcached_ownership = ownership
        native_profile.record(
            "scheduler.schedule_total", time.perf_counter() - schedule_started)
        note_progress()
        return output
