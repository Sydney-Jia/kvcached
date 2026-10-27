# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Bounded physical-release audit with independent driver readings and controls.

This is not a memory-pressure or serving-performance benchmark. Run only on an
available device. At most about 0.6 GiB of test data is resident at a time.
"""
import argparse
import json
import os
from pathlib import Path

from driver_memory import driver_memory_snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.getenv("ASCEND_RT_VISIBLE_DEVICES"):
        parser.error("Select an available NPU with ASCEND_RT_VISIBLE_DEVICES")
    import torch
    import torch_npu  # noqa: F401

    from kvcached import vmm_ops as vmm

    torch.npu.set_device(0)
    page, amount, tolerance = 2 << 20, 256 << 20, 8 << 20
    events = []

    def record(label):
        value = {"label": label, "driver": driver_memory_snapshot(),
                 "ledger": vmm.physical_budget_state()}
        events.append(value)
        args.output.write_text(json.dumps(events, indent=2) + "\n")
        print(label, json.dumps(value), flush=True)
        assert value["driver"]["sample_range_bytes"] <= tolerance, "Unstable driver samples"
        return value["driver"]["free_bytes"]

    def check_delta(before, after, expected):
        assert abs((after - before) - expected) <= tolerance, {
            "observed_free_delta": after - before, "expected": expected,
            "tolerance": tolerance, "note": "Device-wide noise can invalidate this measurement"}

    # Warm the fill/copy paths before measuring. Empty the PyTorch allocator
    # so a later independent recipient must acquire backing from the driver.
    warm = torch.empty(amount // 2, dtype=torch.bfloat16, device="npu:0")
    warm.fill_(3)
    assert warm[:16].cpu().tolist() == [3] * 16
    del warm
    torch.npu.synchronize()
    torch.npu.empty_cache()
    vmm.init_kvcached("npu:0", page, False,
                     instance_name=f"ascend-physical-audit-{os.getpid()}")
    try:
        # The native API takes element width, returning an integer storage
        # tensor. Match the serving integration's explicit dtype view.
        tensor = vmm.create_kv_tensors(2 * amount, 2, "npu:0", 1, 1,
                                      page_size=page)[0].view(torch.bfloat16)
        assert tensor.dtype == torch.bfloat16 and tensor.element_size() == 2
        address = tensor.data_ptr()
        assert vmm.map_to_kv_tensors([0])
        tensor[:32].fill_(17)
        offsets = list(range(page, page + amount, page))
        start, end = page // 2, (page + amount) // 2
        for iteration in range(3):
            before = record(f"{iteration}-baseline-live-guard")
            assert vmm.map_to_kv_tensors(offsets)
            tensor[start:end].fill_(23)
            mapped = record(f"{iteration}-mapped-touched")
            check_delta(before, mapped, -amount)
            transaction = f"audit-abort-{iteration}"
            assert vmm.prepare_unmap_from_kv_tensors(offsets, transaction, 0)
            prepared = record(f"{iteration}-prepared-retained-handles")
            check_delta(mapped, prepared, 0)
            assert vmm.abort_unmap_from_kv_tensors(transaction, 0)
            assert tensor[start:start + 32].cpu().tolist() == [23] * 32
            aborted = record(f"{iteration}-aborted-still-resident")
            check_delta(mapped, aborted, 0)
            assert vmm.unmap_from_kv_tensors(offsets)
            released = record(f"{iteration}-committed-release")
            check_delta(mapped, released, amount)
            assert tensor.data_ptr() == address
            assert tensor[:32].cpu().tolist() == [17] * 32
            # An independent allocator can consume the returned capacity. This
            # is not a claim about reusing the exact physical page handles.
            recipient = torch.empty(amount // 2, dtype=torch.bfloat16, device="npu:0")
            recipient.fill_(31)
            assert recipient[:32].cpu().tolist() == [31] * 32
            received = record(f"{iteration}-independent-torch-recipient")
            check_delta(released, received, -amount)
            del recipient
            torch.npu.synchronize()
            cached = record(f"{iteration}-torch-object-freed-cache-retained")
            check_delta(received, cached, 0)
            torch.npu.empty_cache()
            uncached = record(f"{iteration}-torch-cache-returned-to-driver")
            check_delta(cached, uncached, amount)
            check_delta(before, uncached, 0)
        assert vmm.unmap_from_kv_tensors([0])
        record("guard-released")
        print("PASS: independent driver deltas, retained-page controls, recipient, stable VA/live guard", flush=True)
    finally:
        torch.npu.synchronize()
        vmm.shutdown_kvcached()


if __name__ == "__main__":
    main()
