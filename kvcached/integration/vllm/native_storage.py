# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Native tensor storage substitution; view construction stays in the engine.

The scoped dispatch intercepts only byte-buffer allocation by Ascend's native
KV allocator. It never changes process-wide torch allocation or rewrites model
views. Explicit metadata validation follows allocation before elastic use.
"""

import json
import os
import threading
import time
from uuid import uuid4

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from kvcached.physical_residency import PhysicalResidency
from kvcached.capacity_relief import ensure_with_allocator_relief
from kvcached.errors import MapQuarantinedError, StateConsistencyError, QuarantinedResizeError
from kvcached.residency import Layout, Storage, block_first_spans
from kvcached.residency_cache import SparePageCache


class NativeStorageArena:
    def __init__(self, device, operations, synchronize, page_bytes=2 * 1024 * 1024,
                 *, lazy=False):
        self.device = torch.device(device)
        self.operations = operations
        self.page_bytes = page_bytes
        self.lazy = lazy
        # Container PID namespaces can assign identical PIDs to independent
        # engines sharing this physical device and budget file.
        self.instance_name = f'native-residency-{os.getpid()}-{uuid4().hex}'
        self.raw = {}
        self.spare_cache = None
        self.spare_retries = 0
        self.profile_enabled = os.getenv('KVCACHED_NATIVE_PROFILE') == '1'
        self.profile = {}
        self._complete = synchronize
        self.residency = PhysicalResidency(
            operations, self._synchronize, self._initialize,
            fatal_errors=(MapQuarantinedError, StateConsistencyError, QuarantinedResizeError),
            timed=self._timed if self.profile_enabled else None)
        self._initialized = False
        self.capacity_relief_attempts = 0
        self.last_capacity_relief = None
        # A speculative prepare runs on its own thread while the engine executes
        # the previous step. Serialize every residency mutation so the ledger,
        # the VMM pool and the spare cache stay consistent.
        self._lock = threading.RLock()
        # Background retirement: growth maps only what the engine requires;
        # pages that lost their owner are fenced and unmapped by this thread,
        # so the engine loop never waits on a device-wide completion.
        self.retire_async = os.getenv('KVCACHED_NATIVE_RETIRE_ASYNC', '1') == '1'
        try:
            self.retire_interval_s = max(0.05, float(os.getenv(
                'KVCACHED_NATIVE_RETIRE_INTERVAL_S', '0.5')))
        except ValueError:
            self.retire_interval_s = 0.5
        try:
            self.retire_utilization = float(os.getenv(
                'KVCACHED_NATIVE_RETIRE_UTILIZATION', '0.8'))
        except ValueError:
            self.retire_utilization = 0.8
        try:
            self.retire_batch_bytes = max(2 * 1024 * 1024, int(os.getenv(
                'KVCACHED_NATIVE_RETIRE_BATCH_BYTES', str(4 * 1024 * 1024))))
        except ValueError:
            self.retire_batch_bytes = 4 * 1024 * 1024
        self._latest_required = None
        self._retire_thread = None
        self._retire_start_lock = threading.Lock()
        self._retire_stop = threading.Event()
        self.retire_chunks = 0
        self.retired_bytes_async = 0

    def _timed(self, name, function, *args):
        if not self.profile_enabled:
            return function(*args)
        start = time.perf_counter_ns()
        try:
            return function(*args)
        finally:
            row = self.profile.setdefault(name, dict(calls=0, host_ns=0))
            row['calls'] += 1
            row['host_ns'] += time.perf_counter_ns() - start

    def _synchronize(self):
        return self._timed('completion_fence', self._complete)

    def _allocator_snapshot(self):
        free, total = torch.npu.mem_get_info()
        return dict(free=free, total=total, allocated=torch.npu.memory_allocated(),
                    reserved=torch.npu.memory_reserved(),
                    physical_budget=self.operations.physical_budget_state(),
                    mapped_bytes=self.residency.status()['mapped_bytes'])

    def _release_unused_allocator(self):
        torch.npu.empty_cache()
        # Grants normally sample at most once per second. A capacity rejection
        # may have just refreshed that sample before empty_cache returned real
        # memory, so the immediate bounded retry must request a fresh sample.
        # Concurrent sampling can still win; grants retain all normal limits.
        self.operations.recalibrate_physical_budget()

    def _report_capacity_relief(self, event):
        self.capacity_relief_attempts += 1
        self.last_capacity_relief = event
        print('KVCACHED_NATIVE_CAPACITY_RELIEF ' + json.dumps(
            dict(instance=self.instance_name, device=str(self.device), **event)), flush=True)

    def ensure(self, desired):
        with self._lock:
            return self._timed('ensure_with_relief', ensure_with_allocator_relief,
                self.residency, desired, self._release_unused_allocator,
                self._allocator_snapshot, self._report_capacity_relief)

    def prefetch(self, desired):
        """Map pages only; never fenced, never initialized, never retired."""
        with self._lock:
            return self._timed('prefetch', self.residency.prefetch, desired)

    def grow(self, desired):
        """Map required pages only; no fence and no retirement on this path."""
        with self._lock:
            desired = self.residency._validate(desired)
            self._latest_required = desired
            self._start_retire_thread()
            return self._timed('grow', self.residency.ensure, desired)

    def defer_retire(self, desired):
        """Publish the current owner set for the background reclaimer."""
        with self._lock:
            self._latest_required = self.residency._validate(desired)
            self._start_retire_thread()

    def _start_retire_thread(self):
        if not self.retire_async or self._retire_thread is not None:
            return
        with self._retire_start_lock:
            if self._retire_thread is not None:
                return
            thread = threading.Thread(target=self._retire_loop,
                                      name='native-retire', daemon=True)
            self._retire_thread = thread
            thread.start()

    def _retire_loop(self):
        """Release unowned pages off the engine loop, one bounded chunk at a time."""
        while not self._retire_stop.wait(self.retire_interval_s):
            if not self._retirement_due():
                continue
            # Drain a few bounded chunks per tick, yielding between them so a
            # step that needs the lock never waits behind a long drain.
            for _ in range(8):
                try:
                    if self._retire_once() <= 0:
                        break
                except Exception as error:  # retirement retries next tick
                    print('KVCACHED_NATIVE_RETIRE_FAILED ' + repr(error)[:200],
                          flush=True)
                    break
                if self._retire_stop.wait(0.01):
                    return

    def _retirement_due(self):
        threshold = self.retire_utilization
        if threshold <= 0:
            return True
        try:
            state = self.operations.physical_budget_state()
        except Exception:
            return True
        budget = int(state.get('budget_bytes', 0) or 0)
        charged = int(state.get('charged_bytes', 0) or 0)
        return budget <= 0 or charged >= budget * threshold

    def _retire_once(self):
        with self._lock:
            target = self._latest_required
            if target is None:
                return 0
            chunk = self._select_retire_chunk(target, self.retire_batch_bytes)
            if not chunk:
                return 0
            reclaimed = self._timed('retire_async', self.residency.reclaim,
                                    target, chunk)
        self.retire_chunks += 1
        self.retired_bytes_async += int(reclaimed)
        return reclaimed

    def _select_retire_chunk(self, target, limit):
        chunk = {}
        total = 0
        for key, mapped in self.residency.mapped.items():
            extra = mapped - target[key]
            if not extra:
                continue
            page_bytes = self.residency.storages[key].page_bytes
            take = []
            for page in sorted(extra):
                take.append(page)
                total += page_bytes
                if total >= limit:
                    break
            if take:
                chunk[key] = set(take)
            if total >= limit:
                break
        return chunk

    def transition(self, desired, *, spare_bytes=0, limit_bytes=None, keep=()):
        with self._lock:
            return self._timed('transition', self._transition, desired, spare_bytes,
                               limit_bytes, keep)

    def _transition(self, desired, spare_bytes, limit_bytes, keep=()):
        # Validation and the device fence happen before retirement. Preserve
        # allocator relief for clean growth failures after unowned pages retire.
        desired = self.residency._validate(desired)
        if keep:
            keep = self.residency._validate(keep)
            # Speculatively mapped pages are not required yet, but retiring
            # them here would throw away the work the prefetcher just did.
            protected = {key: desired[key] | keep[key] for key in desired}
        else:
            protected = desired
        if spare_bytes:
            if self.spare_cache is None:
                self.spare_cache = SparePageCache()
            target = self.spare_cache.select(self.residency.storages,
                self.residency.mapped, protected, spare_bytes, limit_bytes)
        else:
            if self.spare_cache is not None:
                self.spare_cache.clear()
            target = protected
        # Serialized residency owns the mapping ledger. An unchanged target
        # needs neither a VMM operation nor another pair of full page walks.
        # Validate first so poisoned state and invalid offsets still fail.
        if all(target[key] == self.residency.mapped[key] for key in target):
            return dict(reclaimed_bytes=0, grown_bytes=0)
        # Only real spare-cache pages take the drop-spares retry path; pages
        # kept from retirement for a speculative prefetch are not spares and
        # must not require a spare cache to exist.
        has_spares = bool(spare_bytes) and self.spare_cache is not None and any(
            set(target[key]) - desired[key] for key in desired)
        reclaimed = self._timed('reclaim', self.residency.reclaim, target)
        if has_spares:
            try:
                # Do not flush the torch allocator while disposable spare
                # pages could satisfy our own growth. Drop those first.
                # Initialize only the required set: spare and prefetched pages
                # are already zeroed or intentionally left to their consumer.
                grown = self._timed('ensure_with_spares', self.residency.ensure, desired)
            except RuntimeError as error:
                if self.residency.poisoned or not str(error).startswith('capacity_exhausted:'):
                    raise
                self.spare_retries += 1
                self.spare_cache.clear()
                reclaimed += self._timed('reclaim', self.residency.reclaim, desired)
                grown = self.ensure(desired)
        else:
            grown = self.ensure(desired)
        return dict(reclaimed_bytes=reclaimed, grown_bytes=grown)

    def _initialize(self, key, pages):
        return self._timed('initialize_enqueue', self._initialize_pages, key, pages)

    def _initialize_pages(self, key, pages):
        storage = self.residency.storages[key]
        runs = []
        for offset in sorted(pages):
            if runs and runs[-1][1] == offset:
                runs[-1][1] += storage.page_bytes
            else:
                runs.append([offset, offset + storage.page_bytes])
        for start, end in runs:
            self.raw[key][start:min(end, storage.size_bytes)].zero_()

    def allocate(self, size):
        if not self._initialized:
            self.operations.init_kvcached(str(self.device), self.page_bytes, False,
                                          self.instance_name)
            self._initialized = True
        pool_id = len(self.raw)
        storage = Storage(str(pool_id), int(size), self.page_bytes)
        tensors = self.operations.create_kv_tensors(
            storage.reservation_bytes, 1, str(self.device), 1, 1,
            pool_id, True, self.page_bytes)
        if len(tensors) != 1:
            raise RuntimeError("Raw storage factory must return exactly one byte tensor")
        self.raw[storage.key] = tensors[0][:storage.size_bytes]
        self.residency.register(storage, pool_id)
        if not self.lazy:
            self.ensure(self.residency.full())
        return self.raw[storage.key]

    def find_storage(self, tensor):
        address = tensor.untyped_storage().data_ptr()
        for key, raw in self.raw.items():
            if address == raw.untyped_storage().data_ptr():
                return key, tensor.data_ptr() - raw.data_ptr()
        raise ValueError("Native KV view is not backed by the captured elastic allocation")

    def layout(self, config, views):
        spans = []
        for group_id, group in enumerate(config.kv_cache_groups):
            for name in group.layer_names:
                if name not in views:
                    raise ValueError(f"Native cache view missing for {name}")
                parts = views[name]
                if isinstance(parts, torch.Tensor):
                    parts = (parts,)
                if not isinstance(parts, (tuple, list)) or not parts:
                    raise ValueError("Native cache must expose explicit tensor components")
                for tensor in parts:
                    if not isinstance(tensor, torch.Tensor):
                        raise ValueError("Unknown native cache component contract")
                    key, offset = self.find_storage(tensor)
                    spans.extend(block_first_spans(
                        storage=key, group_id=group_id, offset=offset,
                        shape=tuple(tensor.shape), strides=tuple(tensor.stride()),
                        itemsize=tensor.element_size(), num_blocks=config.num_blocks))
        return Layout(self.residency.storages.values(), spans)


class NativeMetadataViews(TorchDispatchMode):
    """Reject data access while lazy native allocations are being reshaped.

    An unfamiliar native reshape must fail before a copy/convert kernel reads
    unbacked addresses. Tensor storage/layout validation follows these views.
    This list is an engine interface contract, never a model-name allowlist.
    """

    _metadata_ops = frozenset((
        'aten.view.default', 'aten.view.dtype', 'aten._unsafe_view.default',
        'aten.as_strided.default', 'aten.slice.Tensor', 'aten.select.int',
        'aten.transpose.int', 'aten.permute.default', 'aten.alias.default',
        'aten.detach.default', 'aten.squeeze.default', 'aten.squeeze.dim',
        'aten.squeeze.dims', 'aten.unsqueeze.default', 'aten.expand.default',
        'aten.split.Tensor', 'aten.split_with_sizes.default', 'aten.unbind.int',
    ))

    def __init__(self, arena):
        super().__init__()
        self.arena = arena

    def _uses_native_storage(self, value):
        if isinstance(value, torch.Tensor):
            if value.device != self.arena.device:
                return False
            address = value.untyped_storage().data_ptr()
            return any(address == raw.untyped_storage().data_ptr()
                       for raw in self.arena.raw.values())
        if isinstance(value, dict):
            return any(self._uses_native_storage(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return any(self._uses_native_storage(item) for item in value)
        return False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if (self.arena.lazy and str(func) not in self._metadata_ops
                and self._uses_native_storage((args, kwargs))):
            raise ValueError(f'Lazy native layout requires a metadata-only view; got {func}')
        return func(*args, **kwargs)


class NativeByteAllocation(NativeMetadataViews):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func == torch.ops.aten.zeros.default:
            device = torch.device(kwargs.get('device') or 'cpu')
            if device.type == self.arena.device.type:
                shape = args[0]
                if (kwargs.get('dtype') != torch.int8 or len(shape) != 1
                        or device != self.arena.device):
                    raise ValueError("Native KV allocator changed its raw byte-buffer contract")
                return self.arena.allocate(shape[0])
        return super().__torch_dispatch__(func, types, args, kwargs)
