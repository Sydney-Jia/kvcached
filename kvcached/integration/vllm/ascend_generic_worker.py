# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Native V2 Ascend adapter for generic storage residency contracts.

Pair with NativeResidencyScheduler; legacy autopatches must be disabled.
Native layout construction and native logical/cache ownership remain intact.
This experimental entry point reports evidence, not a universal support claim.
"""

import json
import os
import time
from functools import wraps

import torch
from vllm_ascend.worker.worker import NPUWorker

from kvcached.integration.vllm.native_storage import NativeByteAllocation, NativeStorageArena
from kvcached.integration.vllm.native_residency import dummy_state_owners
from kvcached.residency import ResidencyOwners


class NativeElasticWorker(NPUWorker):
    def init_device(self):
        if any(os.getenv(key, '0').lower() in ('true', '1')
               for key in ('ENABLE_KVCACHED', 'KVCACHED_AUTOPATCH')):
            raise ValueError("Native residency must not run with legacy kvcached autopatches")
        super().init_device()
        config = self.vllm_config
        if not self.use_v2_model_runner:
            raise ValueError("This adapter requires the native V2 allocation/view contract")
        if config.model_config.enable_sleep_mode:
            raise ValueError("Only one physical storage owner is permitted; disable sleep mode")
        if config.scheduler_config.async_scheduling:
            raise ValueError("Queued execution needs an ownership-delivery protocol; use synchronous scheduling")
        if config.kv_transfer_config is not None:
            raise ValueError("Connector-owned transfers require additional completion owners")
        for axis in ('pipeline_parallel_size', 'decode_context_parallel_size', 'prefill_context_parallel_size'):
            if getattr(config.parallel_config, axis, 1) != 1:
                raise ValueError(f"{axis} requires an additional distributed lifetime contract")
        self.native_arena = None
        self.native_owners = None
        self.native_epoch = 0
        self.native_elastic = False
        self.native_steps = 0
        self.native_mapped_revision = -1
        self.native_graph_replays = 0
        self.native_graph_captures = 0
        self.native_trace_residency = os.getenv('KVCACHED_TRACE_RESIDENCY') == '1'
        if self.native_trace_residency:
            from pathlib import Path
            directory = Path(os.getenv('KVCACHED_TRACE_DIR', '/tmp/kvcached-residency'))
            directory.mkdir(parents=True, exist_ok=True)
            self.native_trace_file = (directory / (
                f'dp{config.parallel_config.data_parallel_rank}-tp{self.rank}-{os.getpid()}.txt'
            )).open('a')
        if os.getenv('KVCACHED_TRACE_GRAPHS') == '1':
            # Validation instrumentation only; count real native API calls,
            # not configuration flags. Compare before/after-request counters
            # to separate startup warmups from serving replays.
            for method, counter in (('replay', 'native_graph_replays'),
                                    ('__init__', 'native_graph_captures')):
                original = getattr(torch.npu.NPUGraph, method)

                def traced(graph, *args, _original=original, _counter=counter, **kwargs):
                    result = _original(graph, *args, **kwargs)
                    setattr(self, _counter, getattr(self, _counter) + 1)
                    return result

                setattr(torch.npu.NPUGraph, method, traced)

    def initialize_from_config(self, kv_cache_config):
        from kvcached import vmm_ops
        from vllm_ascend.worker.v2 import attn_utils

        if self.native_arena is not None:
            raise RuntimeError("Persistent elastic cache cannot be initialized twice")
        # Profiling populated PyTorch's caching allocator. Native KV tensors
        # could reuse its free arenas, whereas a separate VMM allocator cannot.
        # Return only unused allocator storage before sizing the VMM budget;
        # keep this startup transfer distinct from later KV reclamation.
        torch.npu.synchronize()
        before = dict(zip(('free', 'total'), torch.npu.mem_get_info()))
        before.update(allocated=torch.npu.memory_allocated(),
                      reserved=torch.npu.memory_reserved())
        torch.npu.empty_cache()
        after = dict(zip(('free', 'total'), torch.npu.mem_get_info()))
        after.update(allocated=torch.npu.memory_allocated(),
                     reserved=torch.npu.memory_reserved())
        print('KVCACHED_NATIVE_ALLOCATOR_HANDOFF ' + json.dumps(
            dict(rank=self.rank, before=before, after=after)), flush=True)
        arena = NativeStorageArena(self.device, vmm_ops, self._residency_fence)
        allocate = attn_utils._allocate_kv_cache
        reshape = attn_utils._reshape_kv_cache_v2
        allocations = 0

        @wraps(allocate)
        def allocate_native(*args, **kwargs):
            nonlocal allocations
            allocations += 1
            if allocations != 1:
                raise RuntimeError("Expected one persistent native allocation phase")
            with NativeByteAllocation(arena):
                return allocate(*args, **kwargs)

        @wraps(reshape)
        def reshape_native(*args, **kwargs):
            views = reshape(*args, **kwargs)
            config = kwargs.get('kv_cache_config')
            if config is None:
                raise ValueError("Native reshape did not supply its cache configuration")
            self.native_owners = ResidencyOwners(arena.layout(config, views))
            # Null/padding slots can be accessed by empty DP batches and graphs.
            self.native_owners.replace({'null': [(group, 0)
                                        for group in self.native_owners.layout.groups]})
            return views

        # Worker initialization is serialized. Limit hook lifetime to the one
        # persistent phase; profiling/capture's temporary caches remain native.
        attn_utils._allocate_kv_cache = allocate_native
        attn_utils._reshape_kv_cache_v2 = reshape_native
        try:
            super().initialize_from_config(kv_cache_config)
        finally:
            attn_utils._allocate_kv_cache = allocate
            attn_utils._reshape_kv_cache_v2 = reshape
        if allocations != 1 or not arena.raw or self.native_owners is None:
            raise RuntimeError("Native storage hook did not materialize a validated layout")
        self.native_arena = arena
        prepare_dummy = self.model_runner.prepare_dummy_attn

        @wraps(prepare_dummy)
        def prepare_owned_dummy(input_batch, *args, **kwargs):
            # Idle DP ranks also execute forwards. Native preparation may write
            # state slots before launching attention, so backing is needed here,
            # not only when a real request enters the scheduler's block table.
            self._trace_residency('dummy-prepare', num_requests=input_batch.num_reqs)
            blocks = dummy_state_owners(self.native_owners.layout.block_counts,
                                        input_batch.num_reqs)
            self.native_owners.replace({'runtime-dummy': blocks})
            self.native_arena.residency.ensure(self.native_owners.required_pages())
            self._trace_residency('dummy-backed')
            result = prepare_dummy(input_batch, *args, **kwargs)
            self._trace_residency('dummy-ready')
            return result

        self.model_runner.prepare_dummy_attn = prepare_owned_dummy
        torch.npu.synchronize()
        print('KVCACHED_GENERIC_LAYOUT ' + json.dumps(self._native_status()), flush=True)

    def execute_model(self, scheduler_output):
        return self._call_traced('execute', self._execute_with_residency, scheduler_output)

    def execute_dummy_batch(self):
        # Idle DP participation bypasses execute_model at the worker boundary.
        return self._call_traced('dummy-batch', super().execute_dummy_batch)

    def sample_tokens(self, grammar_output):
        # Native sampling may also launch speculative forwards/collectives.
        return self._call_traced('sample', super().sample_tokens, grammar_output)

    def _call_traced(self, phase, function, *args):
        # Opt-in diagnostics distinguish a CPU-side residency fence from a
        # native collective stall. They must not change the execution policy.
        self._trace_residency(phase + '-call')
        if self.native_trace_residency:
            import faulthandler
            faulthandler.dump_traceback_later(45, repeat=False, file=self.native_trace_file)
        try:
            return function(*args)
        finally:
            if self.native_trace_residency:
                faulthandler.cancel_dump_traceback_later()
            self._trace_residency(phase + '-finished')

    def _trace_residency(self, phase, **extra):
        if self.native_trace_residency:
            print('KVCACHED_RESIDENCY_TRACE ' + json.dumps(dict(
                phase=phase, timestamp=time.time(), rank=self.rank,
                dp_rank=self.vllm_config.parallel_config.data_parallel_rank,
                pid=os.getpid(), epoch=self.native_epoch, **extra)), flush=True)

    def _residency_fence(self):
        self._trace_residency('fence-enter')
        torch.npu.synchronize()
        self._trace_residency('fence-complete')

    def _execute_with_residency(self, scheduler_output):
        self._trace_residency('step-enter',
                              scheduled_tokens=scheduler_output.total_num_scheduled_tokens)
        if self.native_arena is None:
            raise RuntimeError("Native residency is not initialized")
        snapshot = getattr(scheduler_output, 'kvcached_ownership', None)
        layout = self.native_owners.layout
        if snapshot is None or snapshot.version != 1:
            raise ValueError("Missing native ownership: select NativeResidencyScheduler and check executor transport")
        if (snapshot.epoch != self.native_epoch + 1 or snapshot.num_groups != len(layout.groups)
                or set(layout.block_counts.values()) != {snapshot.num_blocks}):
            raise ValueError("Native ownership epoch or layout identity changed")
        self.native_owners.replace({**snapshot.owners, 'runtime-dummy': ()})
        if self.native_mapped_revision != self.native_owners.revision:
            required = self.native_owners.required_pages()
            self._trace_residency('ensure-enter')
            self.native_arena.residency.ensure(required)
            self._trace_residency('ensure-complete')
            if self.native_elastic:
                # Full device completion fences all streams, including graph-update
                # and state-copy streams. Optimize only after event coverage proof.
                self.native_arena.residency.reclaim(required)
                self._trace_residency('reclaim-complete')
            self.native_mapped_revision = self.native_owners.revision
        self.native_epoch = snapshot.epoch
        self.native_steps += 1
        self._trace_residency('native-execute-enter')
        result = super().execute_model(scheduler_output)
        self._trace_residency('native-execute-return')
        return result

    def _native_status(self):
        arena = self.native_arena
        layout = self.native_owners.layout
        return dict(arena.residency.status(), elastic=self.native_elastic,
                    instance_name=arena.instance_name,
                    epoch=self.native_epoch, steps=self.native_steps,
                    rank=self.rank, dp_rank=self.vllm_config.parallel_config.data_parallel_rank,
                    pid=os.getpid(), device=str(self.device), groups=len(layout.groups),
                    graph_replays=self.native_graph_replays, graph_captures=self.native_graph_captures,
                    physical_device=arena.operations.current_device_pci_bus_id(),
                    physical_budget=dict(arena.operations.physical_budget_state()),
                    group_blocks=layout.block_counts, spans=len(layout.spans),
                    base_addresses={key: raw.data_ptr() for key, raw in arena.raw.items()},
                    owners=self.native_owners.snapshot())

    def kvcached_native_control(self, action='status'):
        if self.native_arena is None:
            raise RuntimeError("Native residency is not initialized")
        started = time.monotonic()
        before, total = torch.npu.mem_get_info()
        reclaimed = 0
        if action == 'elastic':
            reclaimed = self.native_arena.residency.reclaim(self.native_owners.required_pages())
            self.native_elastic = True
        elif action == 'held':
            self.native_arena.residency.ensure(self.native_arena.residency.full())
            torch.npu.synchronize()
            self.native_elastic = False
        elif action == 'trim_allocator':
            torch.npu.synchronize()
            torch.npu.empty_cache()
        elif action != 'status':
            raise ValueError(f"Unknown residency action {action!r}")
        after, _ = torch.npu.mem_get_info()
        result = dict(self._native_status(), action=action, reclaimed_bytes=reclaimed,
                      driver_free_before=before, driver_free_after=after, driver_total=total,
                      elapsed_ms=(time.monotonic() - started) * 1000)
        print('KVCACHED_GENERIC_CONTROL ' + json.dumps(result), flush=True)
        return result
