# SPDX-License-Identifier: Apache-2.0
"""Opt-in measurement RPCs; no change to engine scheduling or attention."""
from worker_probe import PhysicalBudgetProbe


class ServiceBurstProbe(PhysicalBudgetProbe):
    def audit_trim_framework(self):
        import time
        import torch
        before = self.driver_memory_snapshot()
        start = time.monotonic()
        torch.npu.synchronize()
        torch.npu.empty_cache()
        return dict(before=before, after=self.driver_memory_snapshot(),
                    elapsed_s=time.monotonic() - start)
