# SPDX-License-Identifier: Apache-2.0
from dataclasses import make_dataclass, replace
from types import SimpleNamespace

import pytest

from kvcached.integration.vllm.ascend_hybrid import normalize_hybrid_specs
from kvcached.utils import KVCachedConfigError


def test_local_state_admission_keeps_live_state_without_phantom_external_block():
    from kvcached.integration.vllm.ascend_hybrid import install_local_state_admission

    class MambaManager:
        def __init__(self):
            self.held = {"running": [7]}

        def get_num_blocks_to_allocate(self, request_id, num_tokens, **kwargs):
            return max(0, (num_tokens + 2047) // 2048 - len(self.held.get(request_id, [])))

    class AscendMambaManager(MambaManager):
        def get_num_blocks_to_allocate(self, request_id, num_tokens, **kwargs):
            return super().get_num_blocks_to_allocate(request_id, num_tokens) + 1

    AscendMambaManager.__module__ = "vllm_ascend.patch.platform.patch_mamba_manager"
    local, untouched = AscendMambaManager(), AscendMambaManager()
    engine = SimpleNamespace(
        vllm_config=SimpleNamespace(kv_transfer_config=None),
        scheduler=SimpleNamespace(kv_cache_manager=SimpleNamespace(
            coordinator=SimpleNamespace(single_type_managers=[local]))),
    )
    assert local.get_num_blocks_to_allocate("running", 600) == 1
    install_local_state_admission(engine)
    install_local_state_admission(engine)
    assert local.get_num_blocks_to_allocate("running", 600) == 0
    assert local.get_num_blocks_to_allocate("new", 600) == 1
    assert local.get_num_blocks_to_allocate("running", 2100) == 1
    assert local.held == {"running": [7]}
    assert untouched.get_num_blocks_to_allocate("running", 600) == 1


def test_local_state_admission_rejects_external_connector():
    from kvcached.integration.vllm.ascend_hybrid import install_local_state_admission

    engine = SimpleNamespace(vllm_config=SimpleNamespace(kv_transfer_config=object()))
    with pytest.raises(KVCachedConfigError, match="external KV"):
        install_local_state_admission(engine)


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


@pytest.mark.parametrize('state_key,index_key', [('state', 'ssm_state_indices'),
                                              ('conv_state', 'cache_indices_opt')])
def test_active_state_bridge_preserves_other_rows_and_padding(state_key, index_key):
    import torch
    from kvcached.integration.vllm.ascend_hybrid import call_with_compact_states

    backing = torch.arange(8 * 64, dtype=torch.float32).reshape(8, 64)
    state = backing[:, :12].view(8, 3, 4)
    before = backing.clone()
    indices = torch.tensor([5, -1, 2], dtype=torch.int32)

    def native(**kwargs):
        compact, local = kwargs[state_key], kwargs[index_key]
        assert compact.shape == (3, 3, 4) and compact.is_contiguous()
        assert local.tolist() == [0, -1, 2]
        compact[0].add_(10)
        compact[2].add_(20)
        return compact.sum()

    call_with_compact_states(native, (), {state_key: state, index_key: indices}, state_key, index_key)
    expected = before.clone()
    expected[5, :12] += 10
    expected[2, :12] += 20
    assert torch.equal(backing, expected)


@pytest.mark.parametrize('model_type', [None, 'mamba', 'qwen3_5_text'])
def test_hybrid_model_scope_is_explicit(model_type):
    from kvcached.integration.vllm.ascend_hybrid import validate_hybrid_model
    config = SimpleNamespace(hf_text_config=SimpleNamespace(model_type=model_type))
    if model_type == 'qwen3_5_text':
        validate_hybrid_model(config)
    else:
        with pytest.raises(KVCachedConfigError):
            validate_hybrid_model(config)
