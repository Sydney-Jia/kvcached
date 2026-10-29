# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Opt-in device acceptance for the version-specific zero-copy NPU bridge."""

import gc
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv('KVCACHED_RUN_ASCEND_TESTS') != '1',
    reason='requires an explicitly selected Ascend test device and native bridge',
)


def test_native_format_and_pointer_survive_physical_retirement():
    import torch
    import torch_npu

    from kvcached import vmm_ops
    from kvcached._npu_storage import wrap_nd_storage

    torch.npu.set_device(0)
    page = 2 * 1024**2
    vmm_ops.init_kvcached('npu:0', page, False, f'native-storage-test-{os.getpid()}')
    try:
        source = vmm_ops.create_kv_tensors(4 * page, 1, 'npu:0', 1, 1, 0, True, page)[0]
        address = source.data_ptr()
        native = wrap_nd_storage(source)
        assert native.data_ptr() == address
        assert torch_npu.get_npu_format(native) == 2
        assert torch_npu.get_npu_format(native.view(torch.float8_e4m3fn)) == 2
        with pytest.raises(RuntimeError, match='before slicing'):
            wrap_nd_storage(source[1:])
        del source
        gc.collect()
        assert vmm_ops.map_to_kv_tensors([0, page], 0)
        native[:32].fill_(23)
        native[page:page + 32].fill_(37)
        torch.npu.synchronize()
        assert vmm_ops.unmap_from_kv_tensors([0], 0)
        assert native[page:page + 32].cpu().tolist() == [37] * 32
        assert vmm_ops.map_to_kv_tensors([0], 0)
        native[:32].fill_(41)
        assert native[:32].cpu().tolist() == [41] * 32
        assert native.data_ptr() == address
    finally:
        torch.npu.synchronize()
        vmm_ops.shutdown_kvcached()
