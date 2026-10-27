# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Opt-in hardware tests: select an idle device with ASCEND_RT_VISIBLE_DEVICES."""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("KVCACHED_RUN_ASCEND_TESTS") != "1",
    reason="requires an explicitly selected Ascend test device",
)


@pytest.fixture
def native():
    import torch
    import torch_npu  # noqa: F401
    from kvcached import vmm_ops

    assert torch.npu.is_available()
    torch.npu.set_device(0)
    yield torch, vmm_ops
    torch.npu.synchronize()
    vmm_ops.shutdown_kvcached()


@pytest.mark.parametrize("contiguous,buffers", [(False, 1), (False, 2), (True, 1), (True, 2)])
def test_partial_reclaim_preserves_live_pages_and_virtual_addresses(native, contiguous, buffers):
    torch, vmm = native
    page = 2 * 1024 * 1024
    vmm.init_kvcached("npu:0", page, contiguous)
    tensors = vmm.create_kv_tensors(8 * page, 2, "npu:0", 2, buffers, page_size=page)
    # Contiguous layout maps one compound page for every layer and KV plane.
    stride = page * 2 * buffers if contiguous else page
    addresses = [tensor.data_ptr() for tensor in tensors]
    print("device", vmm.current_device_pci_bus_id(), "addresses", addresses, flush=True)
    initial_charge = vmm.physical_budget_state()["instance_charged_bytes"]
    bytes_per_offset = page * 2 * buffers
    assert vmm.map_to_kv_tensors([0, stride])
    assert vmm.physical_budget_state()["instance_charged_bytes"] == initial_charge + 2 * bytes_per_offset
    # Each non-contiguous MHA tensor holds separate K and V halves.
    starts = (list(range(0, stride // 2, page // 2)) if contiguous else
              [0, 4 * page // 2] if buffers == 2 else [0])
    for tensor in tensors:
        for start in starts:
            tensor[start:start + 32].fill_(17)
            tensor[start + stride // 2:start + stride // 2 + 32].fill_(23)
    torch.npu.synchronize()
    assert vmm.unmap_from_kv_tensors([0])
    assert vmm.physical_budget_state()["instance_charged_bytes"] == initial_charge + bytes_per_offset
    for tensor in tensors:
        for start in starts:
            assert tensor[start + stride // 2:start + stride // 2 + 32].cpu().tolist() == [23] * 32
    assert vmm.map_to_kv_tensors([0])
    assert [tensor.data_ptr() for tensor in tensors] == addresses
    for tensor in tensors:
        for start in starts:
            tensor[start:start + 32].fill_(31)
            assert tensor[start:start + 32].cpu().tolist() == [31] * 32
    torch.npu.synchronize()
    assert vmm.prepare_unmap_from_kv_tensors([stride], "ascend-abort", 0)
    assert vmm.abort_unmap_from_kv_tensors("ascend-abort", 0)
    for tensor in tensors:
        for start in starts:
            assert tensor[start + stride // 2:start + stride // 2 + 32].cpu().tolist() == [23] * 32
    torch.npu.synchronize()
    assert vmm.unmap_from_kv_tensors([0, stride])
    assert vmm.physical_budget_state()["instance_charged_bytes"] == initial_charge
