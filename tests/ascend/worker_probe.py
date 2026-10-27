# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Local validation-only worker instrumentation via named vLLM RPC."""


class PhysicalBudgetProbe:
    def audit_allocate_ballast(self, amount):
        """Test-only physical allocation, never an artificial kvcached quota."""
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
        self._audit_ballast = torch.empty(amount // 2, dtype=torch.bfloat16, device=self.device)
        # Touch the allocation without issuing an enormous fill kernel. CANN's
        # driver free-memory delta, not tensor shape, establishes commitment.
        for start in range(0, amount // 2, 128 * 1024 * 1024):
            self._audit_ballast[start:start + 32].fill_(3)
        torch.npu.synchronize()
        after = self.driver_memory_snapshot()
        delta = before["free_bytes"] - after["free_bytes"]
        if abs(delta - amount) > 16 * 1024 * 1024:
            self.audit_release_ballast()
            raise RuntimeError(f"Ballast driver delta {delta} disagrees with allocation {amount}")
        return {"requested_bytes": amount, "driver_delta_bytes": delta,
                "before": before, "after": after}

    def audit_release_ballast(self):
        import torch

        if getattr(self, "_audit_ballast", None) is not None:
            self._audit_ballast = None
            torch.npu.synchronize()
            torch.npu.empty_cache()
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
