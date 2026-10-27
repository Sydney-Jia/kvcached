# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Local validation-only worker instrumentation via named vLLM RPC."""


class PhysicalBudgetProbe:
    def audit_trace_fia(self, path):
        from fia_trace import install

        return install(path)

    def audit_allocate_ballast(self, amount):
        """Test-only physical allocation, never an artificial kvcached quota."""
        import ctypes
        import os

        import torch

        if os.getenv("KVCACHED_PRESSURE_AUDIT_AUTHORIZED") != "1":
            raise RuntimeError("Physical-pressure allocation requires explicit authorization")
        if getattr(self, "_audit_ballast", None) is not None:
            raise RuntimeError("Ballast already allocated")
        if amount <= 0 or amount % (2 * 1024 * 1024):
            raise ValueError("Ballast must be a positive multiple of 2 MiB")
        before = self.driver_memory_snapshot()
        if before["free_bytes"] < amount + 32 * 1024 * 1024:
            raise RuntimeError("Insufficient physical headroom for the fixed ballast")
        # Use a separate CANN allocation. PyTorch's expandable segments can
        # release older cache and round the new segment, changing the net HBM
        # pressure independently of the requested tensor size.
        runtime = ctypes.CDLL("libascendcl.so")
        allocate = runtime.aclrtMalloc
        allocate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int]
        allocate.restype = ctypes.c_int
        pointer = ctypes.c_void_p()
        status = allocate(ctypes.byref(pointer), amount, 0)  # ACL_MEM_MALLOC_HUGE_FIRST
        if status:
            raise RuntimeError(f"CANN ballast allocation failed: {status}")
        self._audit_ballast = (runtime, pointer)
        touch = runtime.aclrtMemset
        touch.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32, ctypes.c_size_t]
        touch.restype = ctypes.c_int
        status = touch(pointer, amount, 3, 32)
        if status:
            self.audit_release_ballast()
            raise RuntimeError(f"CANN ballast touch failed: {status}")
        torch.npu.synchronize()
        after = self.driver_memory_snapshot()
        delta = before["free_bytes"] - after["free_bytes"]
        if abs(delta - amount) > 16 * 1024 * 1024:
            self.audit_release_ballast()
            raise RuntimeError(
                f"Ballast driver delta {delta} disagrees with allocation {amount}; "
                f"before={before}; after={after}")
        return {"requested_bytes": amount, "driver_delta_bytes": delta,
                "allocator": "CANN aclrtMalloc ACL_MEM_MALLOC_HUGE_FIRST",
                "before": before, "after": after}

    def audit_release_ballast(self):
        import ctypes

        import torch

        if getattr(self, "_audit_ballast", None) is not None:
            torch.npu.synchronize()
            runtime, pointer = self._audit_ballast
            release = runtime.aclrtFree
            release.argtypes = [ctypes.c_void_p]
            release.restype = ctypes.c_int
            status = release(pointer)
            if status:
                raise RuntimeError(f"CANN ballast release failed: {status}")
            self._audit_ballast = None
        return {"released": True}

    def driver_memory_snapshot(self):
        from driver_memory import driver_memory_snapshot

        return driver_memory_snapshot()

    def kvcached_physical_snapshot(self):
        import torch

        from kvcached import _C, vmm_ops

        torch.npu.synchronize()
        return {**vmm_ops.physical_budget_state(),
                "pool_control": dict(_C.local_instance_control(0)),
                "driver": self.driver_memory_snapshot()}
