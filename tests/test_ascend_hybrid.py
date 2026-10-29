# SPDX-License-Identifier: Apache-2.0
from dataclasses import make_dataclass, replace
from types import SimpleNamespace

import pytest

from kvcached.integration.vllm.ascend_hybrid import normalize_hybrid_specs
from kvcached.utils import KVCachedConfigError


def specs():
    Full = make_dataclass(
        "FullAttentionSpec",
        ["block_size", "num_kv_heads", "head_size", "head_size_v", "dtype", "page_size_padded"],
        frozen=True,
    )
    Mamba = make_dataclass(
        "MambaSpec",
        ["shapes", "dtypes", "mamba_cache_mode", "num_speculative_blocks", "page_size_padded"],
        frozen=True,
    )
    bf16, fp32 = SimpleNamespace(itemsize=2), SimpleNamespace(itemsize=4)
    return {
        "attn": Full(2048, 4, 256, 256, bf16, 8450048),
        "gdn": Mamba(((3, 10240), (48, 128, 128)), (bf16, fp32), "none", 0, 8450048),
    }


def test_normalize_before_planning_without_changing_state_semantics():
    original = specs()
    result = normalize_hybrid_specs(original)
    assert all(s.page_size_padded == 8388608 for s in result.values())
    assert original["attn"].page_size_padded == 8450048
    assert result["gdn"].shapes == original["gdn"].shapes
    assert result["gdn"].dtypes == original["gdn"].dtypes
    assert result["attn"].block_size == original["attn"].block_size


@pytest.mark.parametrize(
    "field,value",
    [("mamba_cache_mode", "align"), ("num_speculative_blocks", 1), ("shapes", ((9000000,),))],
)
def test_unsupported_state_contract_fails_closed(field, value):
    original = specs()
    original["gdn"] = replace(original["gdn"], **{field: value})
    with pytest.raises(KVCachedConfigError):
        normalize_hybrid_specs(original)
