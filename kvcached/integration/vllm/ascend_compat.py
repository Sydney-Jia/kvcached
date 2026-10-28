# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Explicit compatibility options for the Ascend serving environment.

These options are separate from KV allocation. In particular, the causal-window
option is a workaround for the tested CANN 9.1 / Ascend950 paged FIA path, not a
general attention replacement or a claim that every CANN release needs it.
"""
import functools
import os

from kvcached.utils import KVCachedConfigError, get_kvcached_logger

_MARKER = "_kvcached_ascend_fia_window"
_ENV = "KVCACHED_ASCEND_FIA_MODE"


def configured_mode():
    mode = os.getenv(_ENV, "native").strip().lower()
    if mode not in ("native", "causal-window"):
        raise KVCachedConfigError(
            f"{_ENV} must be 'native' or 'causal-window', got {mode!r}")
    return mode


def status(module=None):
    if module is None:
        import torch_npu as module
    return dict(getattr(module, _MARKER, {"enabled": False}))


def install_causal_window(module=None):
    """Install only when explicitly requested by the caller.

    Mode 3 ignores pre/next tokens; mode 4 with unlimited past and zero future
    expresses the same right-aligned causal window. Preserve the original mask,
    tensors, block table and all other arguments. Graph/out and unrelated
    layouts are deliberately not redirected.
    """
    if module is None:
        import torch_npu as module
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

    setattr(module, _MARKER, record)
    module.npu_fused_infer_attention_score = causal_window
    return status(module)


def validate_worker(config):
    """Reject incompatible opt-ins before initializing a device."""
    mode = configured_mode()
    if mode == "causal-window" and not config.model_config.enforce_eager:
        raise KVCachedConfigError("Ascend causal-window compatibility requires eager execution")
    return mode


def configure_worker(config, module=None):
    """Called after the native worker establishes its device context.

    The opt-in can also be used with ENABLE_KVCACHED=false for a native-memory
    reference using exactly the same attention path. With the default 'native'
    setting, no attention function is changed.
    """
    mode = validate_worker(config)
    if mode == "native":
        return {"enabled": False}
    if module is None:
        import torch_npu as module
    result = install_causal_window(module)
    get_kvcached_logger().warning(
        "Explicit Ascend FIA compatibility enabled: paged TND causal-window; "
        "validated on Ascend950/CANN 9.1, not a general kernel fix. "
        "Set %s=native to preserve the original attention path.", _ENV)
    return result
