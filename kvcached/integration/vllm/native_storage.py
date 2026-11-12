# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Native tensor storage substitution; view construction stays in the engine.

The scoped dispatch intercepts only byte-buffer allocation by Ascend's native
KV allocator. It never changes process-wide torch allocation or rewrites model
views. Explicit metadata validation follows allocation before elastic use.
"""

import json
import os
from uuid import uuid4

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from kvcached.physical_residency import PhysicalResidency
from kvcached.capacity_relief import ensure_with_allocator_relief
from kvcached.errors import MapQuarantinedError, StateConsistencyError, QuarantinedResizeError
from kvcached.residency import Layout, Storage, block_first_spans


class NativeStorageArena:
    def __init__(self, device, operations, synchronize, page_bytes=2 * 1024 * 1024):
        self.device = torch.device(device)
        self.operations = operations
        self.page_bytes = page_bytes
        # Container PID namespaces can assign identical PIDs to independent
        # engines sharing this physical device and budget file.
        self.instance_name = f'native-residency-{os.getpid()}-{uuid4().hex}'
        self.raw = {}
        self.residency = PhysicalResidency(
            operations, synchronize, self._initialize,
            fatal_errors=(MapQuarantinedError, StateConsistencyError, QuarantinedResizeError))
        self._initialized = False
        self.capacity_relief_attempts = 0
        self.last_capacity_relief = None

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
        return ensure_with_allocator_relief(
            self.residency, desired, self._release_unused_allocator,
            self._allocator_snapshot, self._report_capacity_relief)

    def _initialize(self, key, pages):
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


class NativeByteAllocation(TorchDispatchMode):
    def __init__(self, arena):
        super().__init__()
        self.arena = arena

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
        return func(*args, **kwargs)
