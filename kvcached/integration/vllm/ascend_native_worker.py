# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Opt-in native-layout V2 worker, initially restricted to DeepSeek's pool ABI.

Selected with --worker-cls; the ordinary Ascend integration is unchanged.
The caller must disable generic kvcached autopatching for this experimental
path. Prefix retention, graphs, connectors and asynchronous scheduling are
rejected until their ownership contracts have separate acceptance evidence.
"""

import json
import os
import time
from functools import wraps

import torch
from vllm_ascend.worker.worker import NPUWorker

from kvcached.integration.vllm.ascend_native_geometry import NativeLiveBlocks
from kvcached.integration.vllm.ascend_native_pool import NativeKVPool


class NativeElasticWorker(NPUWorker):
    def init_device(self):
        if any(os.getenv(name, '0').lower() in ('1', 'true')
               for name in ('ENABLE_KVCACHED', 'KVCACHED_AUTOPATCH')):
            raise ValueError("Disable generic kvcached autopatching for the native worker")
        super().init_device()
        config = self.vllm_config
        if not self.use_v2_model_runner or not config.model_config.enforce_eager:
            raise ValueError("Native elastic KV requires the eager V2 runner")
        if config.model_config.enable_sleep_mode or config.cache_config.enable_prefix_caching:
            raise ValueError("Disable sleep mode and prefix caching for native elastic KV")
        if config.scheduler_config.async_scheduling or config.kv_transfer_config is not None:
            raise ValueError("Async scheduling and KV connectors are not accepted yet")
        parallel = config.parallel_config
        for axis in ('pipeline_parallel_size', 'prefill_context_parallel_size', 'decode_context_parallel_size'):
            if getattr(parallel, axis, 1) != 1:
                raise ValueError(f"Native elastic KV does not yet support {axis}")
        if (parallel.tensor_parallel_size, parallel.data_parallel_size) != (2, 2):
            raise ValueError("This experimental worker targets TP2/DP2/EP4")
        if not parallel.enable_expert_parallel:
            raise ValueError("The four-card native cache recipe requires expert parallelism")
        if config.speculative_config is not None:
            raise ValueError("Speculative KV lifetimes require separate acceptance")
        self.native_kv_pool = None
        self.native_live_blocks = None

    def initialize_from_config(self, kv_cache_config):
        from vllm_ascend.models.deepseek_v41.cache_config import is_deepseek_v41_cache
        from vllm_ascend.worker.v2 import attn_utils

        from kvcached import vmm_ops

        original = attn_utils._allocate_kv_cache

        @wraps(original)
        def allocate_native(config, shared_layers, device):
            if shared_layers:
                raise ValueError("Separate KV-sharing consumers need an explicit owner contract")
            if not is_deepseek_v41_cache(config.kv_cache_groups):
                raise ValueError("Expected DeepSeek V4.1 native cache descriptors")
            if self.native_kv_pool is not None:
                raise RuntimeError("Persistent native cache was initialized twice")
            self.native_kv_pool = NativeKVPool(
                config, device, vmm_ops, torch.npu.synchronize,
                instance_name=f'kvcached_native_{os.getpid()}',
            )
            self.native_live_blocks = NativeLiveBlocks(len(config.kv_cache_groups), config.num_blocks)
            # Match the native initialized backing before warm-up kernels run.
            # The first explicit 'elastic' control retires unused physical pages.
            self.native_kv_pool.ensure(full=True)
            torch.npu.synchronize()
            return self.native_kv_pool.views_by_layer()

        # Only substitute byte allocation for this persistent initialization.
        # The original reshape and bind paths construct every specialized view.
        attn_utils._allocate_kv_cache = allocate_native
        try:
            super().initialize_from_config(kv_cache_config)
        finally:
            attn_utils._allocate_kv_cache = original
        if self.native_kv_pool is None:
            raise RuntimeError("Native allocation hook was not invoked")

    def execute_model(self, scheduler_output):
        pool = self.native_kv_pool
        if pool is not None:
            required = self.native_live_blocks.update(scheduler_output)
            pool.ensure(required)
            if pool.elastic:
                # Include transient zero/copy endpoints until this iteration's
                # kernels finish. Unscheduled request state remains protected.
                pool.reclaim(required)
        return super().execute_model(scheduler_output)

    def kvcached_native_control(self, action='status'):
        pool = self.native_kv_pool
        if pool is None:
            raise RuntimeError("Native elastic pool is not initialized")
        started = time.monotonic()
        reclaimed = 0
        if action == 'elastic':
            reclaimed = pool.reclaim(self.native_live_blocks.live_blocks())
            pool.elastic = True
        elif action == 'held':
            # Grow only free backing; a failed grow rolls back its new pages.
            pool.ensure(full=True)
            pool.elastic = False
        elif action != 'status':
            raise ValueError(f"Unknown native KV control action: {action}")
        result = dict(pool.status(), action=action, reclaimed_bytes=reclaimed,
                    live_requests=len(self.native_live_blocks.requests),
                    live_blocks=len(self.native_live_blocks.live_blocks()),
                    base_addresses=[t.data_ptr() for t in pool.raw.values()],
                    device=str(self.device), rank=self.rank, pid=os.getpid(),
                    dp_rank=self.vllm_config.parallel_config.data_parallel_rank,
                    elapsed_ms=(time.monotonic() - started) * 1000)
        # vLLM's DP load-balancer fans utility RPCs out to every engine but
        # returns only the first engine's results. Retain all ranks' evidence.
        print('KVCACHED_NATIVE_CONTROL ' + json.dumps(result), flush=True)
        return result
