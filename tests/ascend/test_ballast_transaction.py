# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""CPU checks that a partially allocated test ballast is released on failure."""
import ctypes
import types
import unittest
from unittest.mock import patch

from worker_probe import PhysicalBudgetProbe

MIB = 1024**2


class Function:
    def __init__(self, invoke):
        self.invoke = invoke

    def __call__(self, *args):
        return self.invoke(*args)


class Runtime:
    def __init__(self, fail_allocation=None, fail_touch=False):
        self.allocated = []
        self.freed = []
        self.attempts = 0
        self.fail_allocation = fail_allocation
        self.fail_touch = fail_touch
        self.aclrtMalloc = Function(self.allocate)
        self.aclrtFree = Function(self.free)
        self.aclrtMemset = Function(self.touch)

    def allocate(self, pointer, size, policy):
        self.attempts += 1
        if self.attempts == self.fail_allocation:
            return 207001
        address = self.attempts * 4096
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.c_void_p(address)
        self.allocated.append((address, size))
        return 0

    def free(self, pointer):
        self.freed.append(pointer.value)
        return 0

    def touch(self, *args):
        return 1 if self.fail_touch else 0


class BallastTransaction(unittest.TestCase):
    def run_probe(self, runtime, operation):
        probe = PhysicalBudgetProbe()
        readings = iter([{"free_bytes": 200 * MIB}, {"free_bytes": 190 * MIB}])
        probe.driver_memory_snapshot = lambda: next(readings)
        torch = types.SimpleNamespace(npu=types.SimpleNamespace(synchronize=lambda: None))
        with patch.dict("os.environ", {"KVCACHED_PRESSURE_AUDIT_AUTHORIZED": "1",
                                       "KVCACHED_AUDIT_BALLAST_CHUNK_BYTES": str(4 * MIB)}), \
                patch.dict("sys.modules", {"torch": torch}), \
                patch("ctypes.CDLL", return_value=runtime):
            operation(probe)

    def test_exact_total_and_reverse_release(self):
        runtime = Runtime()

        def operation(probe):
            result = probe.audit_allocate_ballast(10 * MIB)
            self.assertEqual([size for _, size in runtime.allocated], [4 * MIB, 4 * MIB, 2 * MIB])
            self.assertEqual(result["allocation_count"], 3)
            probe.audit_release_ballast()
            self.assertEqual(runtime.freed, [12288, 8192, 4096])
            self.assertIsNone(probe._audit_ballast)
            probe.audit_release_ballast()
            self.assertEqual(len(runtime.freed), 3)

        self.run_probe(runtime, operation)

    def test_allocation_failure_rolls_back_prior_chunks(self):
        runtime = Runtime(fail_allocation=3)

        def operation(probe):
            with self.assertRaisesRegex(RuntimeError, "207001"):
                probe.audit_allocate_ballast(10 * MIB)
            self.assertEqual(runtime.freed, [8192, 4096])
            self.assertIsNone(probe._audit_ballast)

        self.run_probe(runtime, operation)

    def test_touch_failure_releases_current_chunk(self):
        runtime = Runtime(fail_touch=True)

        def operation(probe):
            with self.assertRaisesRegex(RuntimeError, "touch failed"):
                probe.audit_allocate_ballast(10 * MIB)
            self.assertEqual(runtime.freed, [4096])
            self.assertIsNone(probe._audit_ballast)

        self.run_probe(runtime, operation)


if __name__ == "__main__":
    unittest.main()
