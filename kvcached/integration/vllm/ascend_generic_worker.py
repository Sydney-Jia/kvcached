# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Native V2 Ascend adapter for generic storage residency contracts.

Pair with NativeResidencyScheduler; legacy autopatches must be disabled.
Native layout construction and native logical/cache ownership remain intact.
This experimental entry point reports evidence, not a universal support claim.
"""

import json
import os
import threading
import time
from contextlib import nullcontext
from functools import wraps

import torch
from vllm_ascend.worker.worker import NPUWorker

from kvcached.integration.vllm.native_storage import NativeByteAllocation, NativeMetadataViews, NativeStorageArena
from kvcached.integration.vllm.native_residency import dummy_state_owners
from kvcached.residency import ResidencyOwners
from kvcached.capacity_relief import is_capacity_rejection
from kvcached.native_capacity import NativeBudget, NativeBudgetState


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
        self.native_capacity_enabled = os.getenv('KVCACHED_NATIVE_CAPACITY') == '1'
        # Sparse physical mapping is a residency property, not an admission
        # property: an elastic deployment without the capacity protocol must
        # still be able to reserve the pool and map pages on demand.
        self.native_lazy_storage = os.getenv('KVCACHED_NATIVE_LAZY_STORAGE') == '1'
        self.native_elastic = self.native_capacity_enabled or self.native_lazy_storage
        limit = os.getenv('KVCACHED_NATIVE_LIMIT_BYTES')
        prefix = os.getenv('KVCACHED_NATIVE_PREFIX_BLOCKS')
        self.native_budget = NativeBudgetState(NativeBudget(
            0, None if limit is None else int(limit), None if prefix is None else int(prefix)))
        self.native_capacity_ticket = 0
        self.native_spare_bytes = int(os.getenv('KVCACHED_NATIVE_SPARE_BYTES', '0'))
        if self.native_spare_bytes < 0:
            raise ValueError('KVCACHED_NATIVE_SPARE_BYTES must be nonnegative')
        # Speculative map-only pages for the next scheduling step. They are
        # protected from retirement until they are consumed or expire, and a
        # prefetch thread may touch them while the engine executes.
        self.native_prefetch_blocks = {}
        self.native_prefetch_ticket = 0
        self.native_prefetch_lock = threading.Lock()
        try:
            self.native_prefetch_bytes = max(0, int(os.getenv(
                'KVCACHED_NATIVE_PREFETCH_BYTES', str(512 << 20))))
        except ValueError:
            self.native_prefetch_bytes = 512 << 20
        try:
            self.native_prefetch_ttl_s = float(os.getenv(
                'KVCACHED_NATIVE_PREFETCH_TTL_S', '30') or 0)
        except ValueError:
            self.native_prefetch_ttl_s = 30.0
        if self.native_capacity_enabled:
            if (config.parallel_config.data_parallel_size != 1
                    or config.max_concurrent_batches != 1):
                raise ValueError('Capacity mode requires single-DP serialized execution')
            backend = config.parallel_config.distributed_executor_backend
            explicit = isinstance(backend, str) and backend.endswith(
                ('.NativeCapacityExecutor', '.NativeCapacityUniProcExecutor'))
            if not explicit:
                raise ValueError(
                    'Select the explicit NativeCapacityExecutor or '
                    'NativeCapacityUniProcExecutor for capacity mode')
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
        arena = NativeStorageArena(self.device, vmm_ops, self._residency_fence,
                                   lazy=self.native_elastic)
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
            with NativeMetadataViews(arena) if arena.lazy else nullcontext():
                views = reshape(*args, **kwargs)
            config = kwargs.get('kv_cache_config')
            if config is None:
                raise ValueError("Native reshape did not supply its cache configuration")
            self.native_owners = ResidencyOwners(arena.layout(config, views))
            # Null/padding slots can be accessed by empty DP batches and graphs.
            self.native_owners.replace({'null': [(group, 0)
                                        for group in self.native_owners.layout.groups]})
            # Native initialization may bind/inspect null cache data after
            # reshape. Only materialize the validated null views at this point.
            arena.ensure(self.native_owners.required_pages())
            return views

        # Worker initialization is serialized. Limit hook lifetime to the one
        # persistent phase; profiling/capture's temporary caches remain native.
        self.native_arena = arena
        prepare_dummy = self.model_runner.prepare_dummy_attn

        @wraps(prepare_dummy)
        def prepare_owned_dummy(input_batch, *args, **kwargs):
            # Idle DP ranks also execute forwards. Native preparation may write
            # state slots before launching attention, so backing is needed here,
            # not only when a real request enters the scheduler's block table.
            self._trace_residency('dummy-prepare', num_requests=input_batch.num_reqs)
            if self.native_owners is None:
                raise RuntimeError('Dummy cache access preceded validated native layout')
            blocks = dummy_state_owners(self.native_owners.layout.block_counts,
                                        input_batch.num_reqs)
            self.native_owners.replace({'runtime-dummy': blocks})
            self.native_arena.ensure(self.native_owners.required_pages())
            self._trace_residency('dummy-backed')
            result = prepare_dummy(input_batch, *args, **kwargs)
            self._trace_residency('dummy-ready')
            return result

        if self.native_capacity_enabled:
            self.model_runner.prepare_dummy_attn = prepare_owned_dummy
        attn_utils._allocate_kv_cache = allocate_native
        attn_utils._reshape_kv_cache_v2 = reshape_native
        try:
            super().initialize_from_config(kv_cache_config)
        finally:
            attn_utils._allocate_kv_cache = allocate
            attn_utils._reshape_kv_cache_v2 = reshape
        if allocations != 1 or not arena.raw or self.native_owners is None:
            raise RuntimeError("Native storage hook did not materialize a validated layout")
        self.model_runner.prepare_dummy_attn = prepare_owned_dummy
        torch.npu.synchronize()
        print('KVCACHED_GENERIC_LAYOUT ' + json.dumps(self._native_status()), flush=True)

    def compile_or_warm_up_model(self):
        if not self.native_capacity_enabled:
            result = super().compile_or_warm_up_model()
            if self.native_arena is None:
                return result
            # Warmup and graph capture map a dummy batch. Capacity mode retires
            # those pages below; elastic mode must do the same or the resident
            # dummy pages starve a second engine on the same card.
            self.native_arena.transition(
                {key: set() for key in self.native_arena.residency.storages})
            print('KVCACHED_NATIVE_STARTUP_RECLAIM ' + json.dumps(dict(
                rank=self.rank,
                mapped_bytes=self.native_arena.residency.status()['mapped_bytes'])),
                flush=True)
            return result
        # Capture must not grow storage while recording a graph. The native
        # MRV2 dummy contract uses null and state slots 1..max_num_reqs; back
        # them before entering warmup/capture, independently of virtual size.
        owners = dummy_state_owners(self.native_owners.layout.block_counts,
                                    self.model_runner.max_num_reqs)
        self.native_owners.replace({'startup-dummy': owners})
        self.native_arena.ensure(self.native_owners.required_pages())
        result = super().compile_or_warm_up_model()
        self._residency_fence()
        self.native_owners.replace({'startup-dummy': (), 'runtime-dummy': ()})
        self.native_arena.transition(self.native_owners.required_pages())
        return result

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
        self.native_owners.replace({**snapshot.owners, 'runtime-dummy': (), 'admission': ()})
        if self.native_mapped_revision != self.native_owners.revision:
            required = self.native_owners.required_pages()
            self._trace_residency('ensure-enter')
            if self.native_capacity_enabled:
                # Admission has already materialized all candidate components
                # on every TP rank. Do not attempt uncoordinated late growth.
                if any(not required[key] <= self.native_arena.residency.mapped[key]
                       for key in required):
                    raise RuntimeError('Execution requires pages absent from native admission')
            if self.native_elastic:
                # Admission materialized every required page on every rank, so
                # a quiet step needs no transition at all: reclaiming here only
                # unmaps the candidates the next admission maps again. Under
                # shared-budget pressure, though, the peer engine needs those
                # pages back, so release them exactly as the base protocol did.
                self._elastic_residency(required, snapshot)
                self._trace_residency('reclaim-complete')
            else:
                self.native_arena.ensure(required)
            self._trace_residency('ensure-complete')
            self.native_mapped_revision = self.native_owners.revision
        self.native_epoch = snapshot.epoch
        self.native_steps += 1
        if scheduler_output.total_num_scheduled_tokens:
            note_activity = getattr(self.native_arena, 'note_activity', None)
            if note_activity is not None:
                note_activity()
        self._trace_residency('native-execute-enter')
        result = super().execute_model(scheduler_output)
        self._trace_residency('native-execute-return')
        return result

    def _live_prefetch(self, now=None):
        """Pages held by unexpired speculative blocks, plus their pairs."""
        registry = getattr(self, 'native_prefetch_blocks', None)
        if not registry:
            return (), {}
        now = time.monotonic() if now is None else now
        with self.native_prefetch_lock:
            expired = [block for block, deadline in registry.items()
                       if deadline <= now]
            for block in expired:
                registry.pop(block, None)
            blocks = tuple(registry)
        if not blocks:
            return (), {}
        layout = self.native_owners.layout
        pairs = tuple((group, block) for group in layout.groups for block in blocks)
        return pairs, layout.pages_for(pairs)

    def kvcached_native_prefetch(self, candidate_ids, budget, ticket):
        """Map predicted candidates ahead of the next step; never initialized.

        Runs on the admission's prefetch thread while the engine executes the
        previous step. Mapping only, so it issues no device fence and no
        retirement; failures are reported and dropped, never fatal.
        """
        response = dict(rank=self.rank, ticket=ticket, ok=False)
        if not self.native_capacity_enabled or self.native_arena is None:
            response['skipped'] = 'disabled'
            return response
        if self.native_prefetch_bytes <= 0 or self.native_prefetch_ttl_s <= 0:
            response['skipped'] = 'off'
            return response
        try:
            if type(ticket) is not int or ticket <= self.native_prefetch_ticket:
                raise ValueError('Stale native prefetch ticket')
            self.native_prefetch_ticket = ticket
            if budget['revision'] != self.native_budget.budget.revision:
                response['skipped'] = 'stale_budget'
                return response
            layout = self.native_owners.layout
            limit = min(layout.block_counts.values())
            blocks = sorted({int(block) for block in candidate_ids
                             if 0 <= int(block) < limit})
            if not blocks:
                response['skipped'] = 'empty'
                return response
            pairs = tuple((group, block) for group in layout.groups
                          for block in blocks)
            pages = layout.pages_for(pairs)
            storage = self.native_arena.residency.storages
            size = sum(len(offsets) * storage[key].page_bytes
                       for key, offsets in pages.items())
            response['bytes'] = size
            if size <= 0 or size > self.native_prefetch_bytes:
                response['skipped'] = 'bytes'
                return response
            mapped = self.native_arena.prefetch(pages)
            deadline = time.monotonic() + self.native_prefetch_ttl_s
            with self.native_prefetch_lock:
                for block in blocks:
                    self.native_prefetch_blocks[block] = deadline
            response.update(ok=True, mapped_bytes=mapped, blocks=len(blocks))
            return response
        except Exception as error:  # prefetch is best-effort
            response['error'] = repr(error)[:160]
            return response

    def _transition_native(self, required, snapshot, *, release_spares=False):
        reserve = getattr(self, 'native_spare_bytes', 0)
        _, keep = self._live_prefetch()
        if (self.native_capacity_enabled and reserve and not release_spares
                and snapshot.owners.get('requests')):
            return self.native_arena.transition(required, spare_bytes=reserve,
                limit_bytes=self.native_budget.budget.limit_bytes, keep=keep)
        # Idle engines, explicit budget changes and partial-grant rollback
        # must converge without another request or an expiry timer.
        return self.native_arena.transition(required, keep=keep)

    def _elastic_residency(self, required, snapshot):
        """Back an elastic step, surviving a peer-held shared budget.

        Without the capacity protocol nothing retries a physical rejection, so
        a peer holding the ledger would kill the engine. Reclaim our own
        unowned pages and wait briefly for the peer's release instead.
        """
        try:
            if self._budget_pressure():
                return self._transition_native(required, snapshot)
            # Quiet step: map only and leave retirement to the idle reclaimer.
            # Without admission the execution owner set is the only retirement
            # target there is, so it must be published every step; a stale
            # target lets the reclaimer unmap pages a running request still
            # uses, which faults the device. With admission, its candidate set
            # stays authoritative and execution must not shrink it.
            return self.native_arena.grow(
                required, publish=not self.native_capacity_enabled)
        except RuntimeError as error:
            if not is_capacity_rejection(error):
                raise
        budget_deadline = time.monotonic() + float(os.getenv(
            'KVCACHED_NATIVE_ELASTIC_WAIT_S', '10'))
        while time.monotonic() < budget_deadline:
            time.sleep(0.05)
            try:
                return self.native_arena.transition(required)
            except RuntimeError as error:
                if not is_capacity_rejection(error):
                    raise
        raise RuntimeError('elastic residency waited for the shared KV budget')

    def _budget_pressure(self):
        """Whether the shared ledger is tight enough to justify a fence.

        The base protocol reclaimed unowned pages on every execution step.
        That keeps a peer engine growable but fences the device each step, so
        only pay it once the mapped budget is close to exhausted.
        """
        try:
            threshold = float(os.getenv(
                'KVCACHED_NATIVE_EXEC_RECLAIM_UTILIZATION', '0.85'))
            state = self.native_arena.operations.physical_budget_state()
            budget = int(state.get('budget_bytes', 0) or 0)
            charged = int(state.get('charged_bytes', 0) or 0)
        except Exception:
            return True
        if budget <= 0:
            return True
        return charged >= budget * threshold

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
                    layout_page_cache=layout.cache_status(),
                    base_addresses={key: raw.data_ptr() for key, raw in arena.raw.items()},
                    capacity_mode=self.native_capacity_enabled,
                    spare_bytes_limit=self.native_spare_bytes,
                    spare_capacity_retries=arena.spare_retries,
                    memory_profile=arena.profile,
                    budget=self.native_budget.status(arena.residency.storages,
                        arena.residency.mapped, self.native_owners.required_pages()),
                    owners=self.native_owners.snapshot())

    def kvcached_native_prepare(self, snapshot, candidate_ids, budget, ticket, *, release_spares=False):
        """Return every TP response before the scheduler makes an admission.

        No inference is launched here. Local pages may be released before a
        clean growth miss; only unowned contents are discarded. Failed TP
        grants are reconciled by the scheduler before allocation can proceed.
        """
        if not self.native_capacity_enabled or self.native_arena is None:
            raise RuntimeError('Native capacity mode is not initialized')
        fatal = None
        rejection = None
        try:
            if type(ticket) is not int or ticket <= self.native_capacity_ticket:
                raise ValueError('Stale native capacity ticket')
            self.native_capacity_ticket = ticket
            layout = self.native_owners.layout
            if (snapshot.version != 1 or snapshot.num_groups != len(layout.groups)
                    or set(layout.block_counts.values()) != {snapshot.num_blocks}):
                raise ValueError('Native admission layout mismatch')
            if type(release_spares) is not bool:
                raise ValueError('release_spares must be boolean')
            changed = budget['revision'] != self.native_budget.budget.revision
            self.native_budget.apply(NativeBudget(**budget))
            self.native_owners.replace({**snapshot.owners, 'admission': [
                (group, block) for group in layout.groups for block in candidate_ids]})
            arena = self.native_arena
            required = self.native_owners.required_pages()
            if not self.native_budget.permits(arena.residency.storages,
                                              arena.residency.mapped, required):
                rejection = 'instance_limit'
            else:
                self._transition_native(required, snapshot,
                    release_spares=release_spares or changed)
        except Exception as error:
            if (not self.native_arena.residency.poisoned
                    and is_capacity_rejection(error)):
                rejection = str(error)
            else:
                fatal = repr(error)
        return dict(rank=self.rank, ticket=ticket, ok=fatal is None and rejection is None,
                    fatal=fatal, rejection=rejection,
                    budget=self.native_budget.status(self.native_arena.residency.storages,
                        self.native_arena.residency.mapped, self.native_owners.required_pages()))

    def kvcached_native_control(self, action='status'):
        if self.native_arena is None:
            raise RuntimeError("Native residency is not initialized")
        started = time.monotonic()
        before, total = torch.npu.mem_get_info()
        reclaimed = 0
        rejection = None
        if action == 'elastic':
            reclaimed = self.native_arena.residency.reclaim(self.native_owners.required_pages())
            self.native_elastic = True
        elif action == 'held':
            if self.native_capacity_enabled:
                rejection = 'held_disabled_in_capacity_mode: use kvcached_native_budget'
            else:
                try:
                    self.native_arena.ensure(self.native_arena.residency.full())
                except RuntimeError as error:
                    if self.native_arena.residency.poisoned or not is_capacity_rejection(error):
                        raise
                    # Return a normal RPC result on every TP worker.
                    rejection = str(error)
            if rejection is None:
                torch.npu.synchronize()
                self.native_elastic = False
        elif action == 'trim_allocator':
            torch.npu.synchronize()
            torch.npu.empty_cache()
        elif action != 'status':
            raise ValueError(f"Unknown residency action {action!r}")
        after, _ = torch.npu.mem_get_info()
        result = dict(self._native_status(), action=action, ok=rejection is None,
                      rejection=rejection, reclaimed_bytes=reclaimed,
                      capacity_relief_attempts=self.native_arena.capacity_relief_attempts,
                      last_capacity_relief=self.native_arena.last_capacity_relief,
                      driver_free_before=before, driver_free_after=after, driver_total=total,
                      elapsed_ms=(time.monotonic() - started) * 1000)
        print('KVCACHED_GENERIC_CONTROL ' + json.dumps(result), flush=True)
        return result
