# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Local validation-only worker instrumentation via named vLLM RPC."""


class PhysicalBudgetProbe:
    def kvcached_physical_snapshot(self):
        import torch

        from kvcached import _C, vmm_ops

        torch.npu.synchronize()
        return {**vmm_ops.physical_budget_state(),
                "pool_control": dict(_C.local_instance_control(0))}
