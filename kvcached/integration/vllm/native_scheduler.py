# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Native vLLM scheduler with an explicit, versioned residency side channel."""

from vllm.v1.core.kv_cache_utils import get_group_id
from vllm.v1.core.sched.scheduler import Scheduler

from kvcached.integration.vllm.native_residency import snapshot_ownership


class NativeResidencyScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kvcached_epoch = 0

    def schedule(self, *args, **kwargs):
        output = super().schedule(*args, **kwargs)
        self._kvcached_epoch += 1
        # The pinned multiprocess executor pickles SchedulerOutput, including
        # this side channel. Workers reject a missing/stale side channel rather
        # than continuing with incomplete ownership after transport changes.
        output.kvcached_ownership = snapshot_ownership(
            self.kv_cache_manager.coordinator, output,
            epoch=self._kvcached_epoch, decode_group=get_group_id)
        return output
