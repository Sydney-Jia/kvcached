# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Independent CANN HBM observations; never derived from kvcached accounting."""
import ctypes
import statistics
import time


def driver_memory_snapshot(samples=5):
    import torch

    torch.npu.synchronize()
    runtime = ctypes.CDLL("libascendcl.so")
    query = runtime.aclrtGetMemInfo
    query.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_size_t),
                      ctypes.POINTER(ctypes.c_size_t)]
    query.restype = ctypes.c_int
    observations = []
    for _ in range(samples):
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        # ACL_HBM_MEM == 1 in the CANN aclrtMemAttr enum.
        status = query(1, ctypes.byref(free), ctypes.byref(total))
        if status:
            raise RuntimeError(f"aclrtGetMemInfo(ACL_HBM_MEM) failed: {status}")
        observations.append({"free_bytes": free.value, "total_bytes": total.value})
        time.sleep(0.02)
    free_values = [item["free_bytes"] for item in observations]
    return {"source": "CANN aclrtGetMemInfo(ACL_HBM_MEM)",
            "scope": "device-wide; other processes can affect the result",
            "samples": observations,
            "free_bytes": int(statistics.median(free_values)),
            "sample_range_bytes": max(free_values) - min(free_values),
            "torch_allocated_bytes": torch.npu.memory_allocated(),
            "torch_reserved_bytes": torch.npu.memory_reserved()}
