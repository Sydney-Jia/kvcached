# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace

import pytest

from kvcached.device import current_device_string, get_device_module, get_device_type


def test_npu_dispatch_does_not_call_cuda(monkeypatch):
    cuda = object()
    npu = SimpleNamespace(current_device=lambda: 2)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=cuda, npu=npu, version=SimpleNamespace(cuda=None, hip=None)))
    assert get_device_type() == "npu"
    assert get_device_module() is npu
    assert get_device_module("npu:2") is npu
    assert current_device_string() == "npu:2"


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_cuda_and_rocm_keep_cuda_dispatch(monkeypatch, backend):
    cuda = SimpleNamespace(current_device=lambda: 1)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=cuda, version=SimpleNamespace(**{backend: "available"})))
    assert get_device_type() == "cuda"
    assert get_device_module(f"{backend}:1") is cuda
    assert current_device_string() == "cuda:1"


def test_explicit_device_wins_over_default(monkeypatch):
    cuda, npu = object(), object()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=cuda, npu=npu, version=SimpleNamespace(cuda="available", hip=None)))
    assert get_device_module() is cuda
    assert get_device_module("npu:0") is npu
    with pytest.raises(ValueError, match="Unsupported accelerator"):
        get_device_module("cpu")


def test_cli_memory_capacity_uses_the_active_npu(monkeypatch):
    from kvcached.cli.utils import get_total_gpu_memory

    npu = SimpleNamespace(
        is_available=lambda: True,
        get_device_properties=lambda index: SimpleNamespace(total_memory=96 * 1024**3),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=object(), npu=npu, version=SimpleNamespace(cuda=None, hip=None)))
    assert get_total_gpu_memory() == 96 * 1024**3
