# SPDX-License-Identifier: Apache-2.0
"""Opt-in geometry for a shared, block-interleaved Ascend hybrid candidate.

Native Ascend flattens state components into separate spans and may include a
convolution-state prefix in its padded attention allocation. kvcached instead
uses one uniform cell per logical block. Normalize *before* the engine plans
groups/budgets, not after scheduler and worker have adopted different sizes.
No state shape, dtype, token block size or kernel implementation is changed.
"""

from dataclasses import replace
from functools import wraps
from math import prod

from kvcached.utils import KVCachedConfigError


def validate_hybrid_model(model_config):
    text_config = getattr(model_config, "hf_text_config", None)
    if getattr(text_config, "model_type", None) != "qwen3_5_text":
        raise KVCachedConfigError("Ascend hybrid candidate is limited to Qwen3.5 GDN text models")


def normalize_hybrid_specs(specs):
    attention = [s for s in specs.values() if type(s).__name__ == "FullAttentionSpec"]
    states = [s for s in specs.values() if type(s).__name__ == "MambaSpec"]
    if not attention or not states or len(attention) + len(states) != len(specs):
        raise KVCachedConfigError("Hybrid candidate requires only full attention and Mamba states")
    sizes = {2 * s.block_size * s.num_kv_heads * s.head_size * s.dtype.itemsize for s in attention}
    if len(sizes) != 1 or any(
        getattr(s, "head_size_v", s.head_size) != s.head_size for s in attention
    ):
        raise KVCachedConfigError("Hybrid candidate requires homogeneous equal-size K/V geometry")
    size = sizes.pop()
    for spec in states:
        if spec.mamba_cache_mode != "none" or spec.num_speculative_blocks:
            raise KVCachedConfigError("Hybrid candidate excludes cached or speculative states")
        required = sum(
            prod(shape) * dtype.itemsize for shape, dtype in zip(spec.shapes, spec.dtypes)
        )
        if required > size:
            raise KVCachedConfigError("Mamba state cannot fit in the attention cell")
    return {name: replace(spec, page_size_padded=size) for name, spec in specs.items()}


def call_with_compact_states(original, args, kwargs, state_key, index_key):
    """Bridge the fixed Ascend GDN ABI without touching unbacked state slots.

    The bounded non-speculative path supplies one state index per request;
    live request indices are distinct. Negative padding indices refer to the
    permanently backed null block for gathering but remain negative for the op.
    This copies active state rows; it is not a zero-copy layout adaptation.
    """
    import torch

    state = kwargs.get(state_key)
    if state is None or state.is_contiguous():
        return original(*args, **kwargs)
    indices = kwargs.get(index_key)
    if indices is None:
        raise KVCachedConfigError("Strided GDN state needs explicit active indices")
    safe_indices = indices.flatten().to(torch.int64).clamp_min(0)
    compact = state.index_select(0, safe_indices).contiguous()
    local = torch.arange(indices.numel(), dtype=indices.dtype,
                         device=indices.device).reshape_as(indices)
    local = torch.where(indices < 0, indices, local)
    call_kwargs = dict(kwargs, **{state_key: compact, index_key: local})
    result = original(*args, **call_kwargs)
    # Same stream: writeback finishes before the next state use or the worker's
    # synchronization before unmapping. No full virtual tensor materialization.
    state.index_copy_(0, safe_indices, compact)
    return result


def install_state_bridge():
    """Worker-local, opt-in wrappers for this snapshot's two native GDN ops."""
    import torch

    namespace = torch.ops._C_ascend
    operations = (
        ("npu_causal_conv1d_custom", "conv_state", "cache_indices_opt"),
        ("npu_recurrent_gated_delta_rule", "state", "ssm_state_indices"),
    )
    for name, state_key, index_key in operations:
        original = getattr(namespace, name)
        if getattr(original, "__kvcached_state_bridge__", False):
            continue

        def make_wrapper(function, state_name, index_name):
            @wraps(function)
            def wrapper(*args, **kwargs):
                return call_with_compact_states(function, args, kwargs, state_name, index_name)
            wrapper.__kvcached_state_bridge__ = True
            return wrapper

        setattr(namespace, name, make_wrapper(original, state_key, index_key))
