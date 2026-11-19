# SPDX-License-Identifier: Apache-2.0
"""Actual arena allocation methods with byte-addressed CPU storage, no torch."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS
from uuid import uuid4
import json
import os
import threading
import time
import unittest

from test_native_admission import load
from test_physical_residency import Memory, module as physical

ROOT = Path(__file__).parents[1]
residency = load("lazy_test_residency", "kvcached/residency.py")
relief = load("lazy_test_relief", "kvcached/capacity_relief.py")


class RawBytes:
    def __init__(self, memory, pool, size, start=0, stop=None):
        self.memory, self.pool, self.size = memory, pool, size
        self.device = "npu:0"
        self.start, self.stop = start, size if stop is None else stop

    def __getitem__(self, span):
        start = self.start + (span.start or 0)
        stop = self.stop if span.stop is None else min(self.start + span.stop, self.stop)
        return RawBytes(self.memory, self.pool, self.size, start, stop)

    def zero_(self):
        for page in range(self.start // 16 * 16, (self.stop + 15) // 16 * 16, 16):
            if page not in self.memory.data[self.pool]:
                raise RuntimeError("write to unbacked page")
            self.memory.data[self.pool][page] = 0

    def untyped_storage(self):
        return NS(data_ptr=lambda: 100000 + self.pool * 10000)


namespace = dict(
    torch=NS(device=lambda device: device, Tensor=RawBytes),
    TorchDispatchMode=object,
    os=os,
    threading=threading,
    time=time,
    json=json,
    uuid4=uuid4,
    Storage=residency.Storage,
    PhysicalResidency=physical.PhysicalResidency,
    MapQuarantinedError=type("MapQuarantinedError", (RuntimeError,), {}),
    StateConsistencyError=type("StateConsistencyError", (RuntimeError,), {}),
    QuarantinedResizeError=type("QuarantinedResizeError", (RuntimeError,), {}),
    ensure_with_allocator_relief=relief.ensure_with_allocator_relief,
)
source = ROOT / "kvcached/integration/vllm/native_storage.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NativeStorageArena")
exec(compile(ast.Module(body=[klass], type_ignores=[]), str(source), "exec"), namespace)
Arena = namespace["NativeStorageArena"]
metadata = next(
    n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NativeMetadataViews"
)
exec(compile(ast.Module(body=[metadata], type_ignores=[]), str(source), "exec"), namespace)


class Operation:
    def __init__(self, name):
        self.name, self.called = name, False

    def __str__(self):
        return self.name

    def __call__(self, *args, **kwargs):
        self.called = True
        return args[0]


class LazyStorageTests(unittest.TestCase):
    def arena(self, lazy):
        memory = Memory()
        memory.init_kvcached = lambda *args: None
        memory.create_kv_tensors = lambda size, layers, device, heads, dim, pool, *args: [
            RawBytes(memory, pool, size)
        ]
        return Arena("npu:0", memory, memory.synchronize, 16, lazy=lazy), memory

    def test_virtual_reservation_does_not_require_full_physical_capacity(self):
        arena, memory = self.arena(True)
        raw = arena.allocate(1024 * 1024)
        self.assertEqual(arena.residency.status()["reserved_bytes"], 1024 * 1024)
        self.assertEqual(arena.residency.status()["mapped_bytes"], 0)
        self.assertEqual(memory.data[0], {})
        arena.ensure({"0": {0}})
        self.assertEqual(memory.data[0], {0: 0})
        self.assertIs(arena.raw["0"], raw)

    def test_lazy_page_reuse_initializes_only_new_backing(self):
        arena, memory = self.arena(True)
        raw = arena.allocate(41)
        arena.ensure({"0": {0, 16}})
        memory.data[0][16] = "retained state"
        arena.transition({"0": {16, 32}})
        self.assertIs(arena.raw["0"], raw)
        self.assertEqual(memory.data[0], {16: "retained state", 32: 0})

    def test_default_held_initialization_is_preserved(self):
        arena, memory = self.arena(False)
        arena.allocate(41)
        self.assertEqual(memory.data[0], {0: 0, 16: 0, 32: 0})
        self.assertEqual(arena.residency.status()["mapped_bytes"], 48)

    def test_lazy_reshape_rejects_a_copy_before_reading_unbacked_storage(self):
        arena, memory = self.arena(True)
        raw = arena.allocate(41)
        guard = namespace["NativeMetadataViews"](arena)
        operation = Operation("aten.clone.default")
        with self.assertRaisesRegex(ValueError, "metadata-only"):
            guard.__torch_dispatch__(operation, (), (raw,))
        self.assertFalse(operation.called)
        self.assertEqual(memory.data[0], {})
        view = Operation("aten.view.dtype")
        self.assertIs(guard.__torch_dispatch__(view, (), (raw,)), raw)
        self.assertTrue(view.called)


    def test_grow_maps_without_retiring_unowned_pages(self):
        os.environ["KVCACHED_NATIVE_RETIRE_ASYNC"] = "0"
        try:
            arena, memory = self.arena(True)
            arena.allocate(4096)
            arena.grow({"0": {0, 16}})
            arena.grow({"0": {0}})
            self.assertEqual(set(memory.data[0]), {0, 16})
        finally:
            del os.environ["KVCACHED_NATIVE_RETIRE_ASYNC"]

    def test_background_reclaimer_retires_pages_that_lost_their_owner(self):
        os.environ["KVCACHED_NATIVE_RETIRE_ASYNC"] = "1"
        os.environ["KVCACHED_NATIVE_RETIRE_UTILIZATION"] = "0"
        os.environ["KVCACHED_NATIVE_RETIRE_INTERVAL_S"] = "0.05"
        os.environ["KVCACHED_NATIVE_RETIRE_GRACE_S"] = "0"
        try:
            arena, memory = self.arena(True)
            arena.allocate(4096)
            arena.grow({"0": {0, 16}})
            arena.defer_retire({"0": {0}})
            deadline = time.time() + 2
            while time.time() < deadline and 16 in memory.data[0]:
                time.sleep(0.02)
            self.assertEqual(set(memory.data[0]), {0})
            self.assertGreaterEqual(arena.retire_chunks, 1)
        finally:
            for name in ("KVCACHED_NATIVE_RETIRE_ASYNC",
                         "KVCACHED_NATIVE_RETIRE_UTILIZATION",
                         "KVCACHED_NATIVE_RETIRE_INTERVAL_S",
                         "KVCACHED_NATIVE_RETIRE_GRACE_S"):
                del os.environ[name]


if __name__ == "__main__":
    unittest.main()
