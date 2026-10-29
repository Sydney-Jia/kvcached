# SPDX-License-Identifier: Apache-2.0
"""Opt-in geometry for a shared, block-interleaved Ascend hybrid candidate.

Native Ascend flattens state components into separate spans and may include a
convolution-state prefix in its padded attention allocation. kvcached instead
uses one uniform cell per logical block. Normalize *before* the engine plans
groups/budgets, not after scheduler and worker have adopted different sizes.
No state shape, dtype, token block size or kernel implementation is changed.
"""

from dataclasses import replace
from math import prod

from kvcached.utils import KVCachedConfigError


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
