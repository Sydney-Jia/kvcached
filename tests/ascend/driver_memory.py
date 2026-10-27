# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Independent CANN HBM observations; never derived from kvcached accounting."""
import ctypes
import os
import re
import statistics
import subprocess
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
    result = {"source": "CANN aclrtGetMemInfo(ACL_HBM_MEM)",
            "scope": "device-wide; other processes can affect the result",
            "samples": observations,
            "free_bytes": int(statistics.median(free_values)),
            "sample_range_bytes": max(free_values) - min(free_values),
            "torch_allocated_bytes": torch.npu.memory_allocated(),
            "torch_reserved_bytes": torch.npu.memory_reserved()}
    if os.getenv("KVCACHED_AUDIT_PHYSICAL_DEVICE") is not None:
        identify = runtime.aclrtDeviceGetBareTgid
        identify.argtypes = [ctypes.POINTER(ctypes.c_int32)]
        identify.restype = ctypes.c_int
        pid = ctypes.c_int32()
        status = identify(ctypes.byref(pid))
        if status:
            raise RuntimeError(f"aclrtDeviceGetBareTgid failed: {status}")
        readings = []
        for _ in range(2):
            output = subprocess.check_output([
                "npu-smi", "info", "-t", "proc-mem", "-i",
                os.environ["KVCACHED_AUDIT_PHYSICAL_DEVICE"]], text=True, timeout=20)
            match = re.search(r"Process id:" + str(pid.value)
                              + r"\b[^\n]*Process memory\(MB\):(\d+)", output)
            if not match:
                raise RuntimeError(f"Cannot identify test PID {pid.value} in npu-smi: {output}")
            readings.append({"reported_MB": int(match.group(1)), "raw": output})
            time.sleep(0.1)
        result["process"] = {"source": "npu-smi info -t proc-mem",
                             "host_pid": pid.value, "samples": readings,
                             "reported_MB": readings[-1]["reported_MB"],
                             "sample_range_MB": abs(readings[0]["reported_MB"]
                                                    - readings[1]["reported_MB"])}
    return result
