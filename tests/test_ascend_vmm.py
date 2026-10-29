# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Opt-in hardware tests: select an idle device with ASCEND_RT_VISIBLE_DEVICES."""

import multiprocessing
import os
import json
import subprocess
import sys
import traceback

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("KVCACHED_RUN_ASCEND_TESTS") != "1",
    reason="requires an explicitly selected Ascend test device",
)


@pytest.mark.skipif(
    os.getenv('KVCACHED_RUN_ASCEND_MULTI_DEVICE_TESTS') != '1',
    reason='requires an explicitly authorized multi-device visibility list',
)
def test_physical_keys_are_distinct_and_visibility_stable():
    visible = os.environ['ASCEND_RT_VISIBLE_DEVICES'].split(',')
    assert len(visible) >= 2 and all(value.isdigit() for value in visible)
    script = '''
import json, torch, torch_npu
from kvcached import _C
keys = []
for device in range(torch.npu.device_count()):
    torch.npu.set_device(device)
    keys.append(_C.current_device_pci_bus_id())
print('PHYSICAL_KEYS=' + json.dumps(keys), flush=True)
'''

    def query(selection):
        env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=','.join(selection),
                   PYTHONPATH=os.pathsep.join(sys.path))
        output = subprocess.check_output([sys.executable, '-c', script], env=env,
                                         text=True, timeout=120)
        return json.loads(next(line.split('=', 1)[1] for line in output.splitlines()
                               if line.startswith('PHYSICAL_KEYS=')))

    keys = query(visible)
    assert len(keys) == len(visible) and len(set(keys)) == len(keys)
    assert query([visible[1]]) == [keys[1]]


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


def _capacity_worker(connection, name):
    """Each engine owns its virtual addresses; only the physical ledger is shared."""
    try:
        import torch
        import torch_npu  # noqa: F401

        from kvcached import vmm_ops as vmm

        page = 2 * 1024 * 1024
        torch.npu.set_device(0)
        vmm.init_kvcached("npu:0", page, False, instance_name=name)
        tensor = vmm.create_kv_tensors(8 * page, 2, "npu:0", 1, 1, page_size=page)[0]
        connection.send({"device": vmm.current_device_pci_bus_id(), "address": tensor.data_ptr()})
        while True:
            command, args = connection.recv()
            if command == "stop":
                break
            try:
                if command == "limit":
                    vmm.set_instance_budget_limit(*args)
                elif command == "map":
                    assert vmm.map_to_kv_tensors(args)
                    for offset in args:
                        tensor[offset // 2:offset // 2 + 16].fill_(71)
                    torch.npu.synchronize()
                elif command == "unmap":
                    torch.npu.synchronize()
                    assert vmm.unmap_from_kv_tensors(args)
                elif command == "read":
                    assert tensor[args[0] // 2:args[0] // 2 + 16].cpu().tolist() == [71] * 16
                connection.send({"ok": True, "state": vmm.physical_budget_state()})
            except RuntimeError as error:
                connection.send({"ok": False, "error": str(error), "state": vmm.physical_budget_state()})
        torch.npu.synchronize()
        vmm.shutdown_kvcached()
    except BaseException:
        connection.send({"fatal": traceback.format_exc()})
        raise
    finally:
        connection.close()


def test_two_processes_rebalance_capacity_without_rewriting_live_kv(tmp_path, monkeypatch):
    monkeypatch.setenv("KVCACHED_PHYSICAL_GROWTH_LOCK_DIR", str(tmp_path))
    context = multiprocessing.get_context("spawn")
    processes, connections = [], []

    def receive(connection):
        assert connection.poll(60), "NPU worker did not respond"
        result = connection.recv()
        assert "fatal" not in result, result
        return result

    def call(index, command, *args):
        connections[index].send((command, args))
        return receive(connections[index])

    try:
        for index in range(2):
            parent, child = context.Pipe()
            process = context.Process(target=_capacity_worker, args=(child, f"ascend-test-{index}"))
            process.start()
            child.close()
            connections.append(parent)
            processes.append(process)
        identities = [receive(connection) for connection in connections]
        assert identities[0]["device"] == identities[1]["device"]
        page = 2 * 1024 * 1024
        assert call(0, "limit", 2 * page, 1)["ok"]
        assert call(1, "limit", 0, 1)["ok"]
        assert call(0, "map", 0, page)["state"]["charged_bytes"] == 2 * page
        rejected = call(1, "map", 0)
        assert not rejected["ok"]
        assert rejected["state"]["instance_charged_bytes"] == 0
        assert call(0, "unmap", 0)["state"]["charged_bytes"] == page
        assert call(0, "limit", page, 2)["ok"]
        assert call(1, "limit", page, 2)["ok"]
        assert call(1, "map", 0)["state"]["charged_bytes"] == 2 * page
        assert call(0, "read", page)["ok"]
        assert call(1, "read", 0)["ok"]
        assert call(0, "unmap", page)["state"]["charged_bytes"] == page
        assert call(1, "unmap", 0)["state"]["charged_bytes"] == 0
    finally:
        for connection, process in zip(connections, processes):
            if process.is_alive():
                try:
                    connection.send(("stop", ()))
                except (EOFError, BrokenPipeError):
                    pass
            process.join(10)
            if process.is_alive():
                process.terminate()
                process.join(10)
            connection.close()
        assert all(process.exitcode == 0 for process in processes)
