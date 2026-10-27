# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Lazy accelerator dispatch shared by the allocator and engine integrations."""

import importlib


def get_device_type() -> str:
    import torch

    version = getattr(torch, "version", None)
    if getattr(version, "cuda", None) or getattr(version, "hip", None):
        return "cuda"
    if hasattr(torch, "npu"):
        return "npu"
    # CPU-only processes may still import scheduling/CLI code. Preserve the
    # existing CUDA module without initializing an accelerator context.
    return "cuda"


def get_device_module(device=None):
    import torch

    if device is None or isinstance(device, int):
        kind = get_device_type()
    else:
        kind = str(device).split(":", 1)[0]
        if kind == "hip":
            kind = "cuda"
    if kind == "npu" and not hasattr(torch, "npu"):
        importlib.import_module("torch_npu")
    if kind not in ("cuda", "npu"):
        raise ValueError(f"Unsupported accelerator device: {device}")
    return getattr(torch, kind)


def current_device_string() -> str:
    kind = get_device_type()
    return f"{kind}:{get_device_module(kind).current_device()}"
