# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Opt-in experiment workaround; not part of the kvcached runtime backend.

On the tested CANN 9.1 / Ascend950 environment, paged TND sparse_mode=3
has failed on mixed partial-prefill shapes even with ordinary tensors.
Mode 4 with an unlimited past and zero future expresses the same causal
window. Independent numerical probes must pass before using this option.
All arms must use the same option; do not merge results across modes.
"""
import functools


def _module(module):
    if module is None:
        import torch_npu

        return torch_npu
    return module


def status(module=None):
    module = _module(module)
    return dict(getattr(module, "_kvcached_audit_fia_window", {"enabled": False}))


def install(module=None):
    module = _module(module)
    if status(module)["enabled"]:
        return status(module)
    original = module.npu_fused_infer_attention_score
    record = {"enabled": True, "scope": "paged TND sparse_mode=3 keyword calls",
              "sparse_mode": 4, "pre_tokens": 2147483647, "next_tokens": 0,
              "redirected_calls": 0}

    @functools.wraps(original)
    def causal_window(*args, **kwargs):
        if (not args and kwargs.get("input_layout") == "TND"
                and kwargs.get("block_table") is not None
                and kwargs.get("sparse_mode") == 3):
            kwargs.update(sparse_mode=4, pre_tokens=2147483647, next_tokens=0)
            record["redirected_calls"] += 1
        return original(*args, **kwargs)

    module._kvcached_audit_fia_window = record
    module.npu_fused_infer_attention_score = causal_window
    return status(module)
