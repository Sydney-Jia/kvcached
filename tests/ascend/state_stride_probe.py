# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 -- Import the explicitly selected runtime after parsing it.
"""Isolate native GDN operator stride/writeback from VMM and model scheduling."""

import argparse
import json
from pathlib import Path
import sys

p = argparse.ArgumentParser()
p.add_argument("--runtime", required=True)
p.add_argument("--output", type=Path, required=True)
a, _ = p.parse_known_args()
sys.path.insert(0, a.runtime)
import torch
import torch_npu
from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env()
import vllm_ascend.vllm_ascend_C  # noqa: F401 -- Register the native operators.

torch.npu.set_device(0)
torch.manual_seed(19)
result = {"status": "diagnostic", "cases": {}}


def random(shape, dtype=torch.bfloat16):
    return (torch.randn(shape) * 0.1).to(dtype).npu()


def states(shape, dtype):
    base = random(shape, dtype)
    backing = torch.zeros(4 * 8 * 1024**2 // dtype.itemsize, dtype=dtype, device="npu")
    view = torch.as_strided(backing, shape, (8 * 1024**2 // dtype.itemsize, *base.stride()[1:]))
    view.copy_(base)
    return base, view, backing


def compare(x, y):
    torch.npu.synchronize()
    x, y = x.cpu(), y.cpu()
    return {
        "equal": torch.equal(x, y),
        "max_abs_error": float((x.float() - y.float()).abs().max()),
        "finite": bool(torch.isfinite(x).all() and torch.isfinite(y).all()),
    }


indices = torch.tensor([1, 3], dtype=torch.int32, device="npu")
# Match this fixed runtime's _build_actual_seq_lengths: leading zero followed
# by per-sequence lengths, not the conventional cumulative-sequence encoding.
lengths = torch.tensor([0, 1, 1], dtype=torch.int32, device="npu")
a.output.with_suffix(".measure").write_text("isolated native operator stride check\n")
for name in ["conv", "recurrent"]:
    try:
        if name == "conv":
            native, strided, owner = states((4, 3, 10240), torch.bfloat16)
            x = random((2, 10240))
            weight = random((4, 10240))

            def call(state):
                output = torch.empty_like(x)
                torch.ops._C_ascend.npu_causal_conv1d_custom(
                    output,
                    x,
                    weight,
                    conv_state=state,
                    bias_opt=None,
                    query_start_loc_opt=torch.tensor([0, 1, 2], dtype=torch.int32, device="npu"),
                    cache_indices_opt=indices,
                    initial_state_mode_opt=None,
                    num_accepted_tokens_opt=None,
                    activation_mode=1,
                    pad_slot_id=-1,
                    run_mode=1,
                )
                return output
        else:
            native, strided, owner = states((4, 48, 128, 128), torch.float32)
            q = random((2, 16, 128))
            k = random((2, 16, 128))
            v = random((2, 48, 128))
            g = torch.full((2, 48), -0.3, dtype=torch.float32, device="npu")
            beta = torch.full((2, 48), 0.4, dtype=torch.bfloat16, device="npu")

            def call(state):
                return torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
                    query=q,
                    key=k,
                    value=v,
                    g=g,
                    beta=beta,
                    state=state,
                    scale=128**-0.5,
                    actual_seq_lengths=lengths,
                    ssm_state_indices=indices,
                )

        initial = native.clone()
        repeat = native.clone()
        reference = call(native)
        repeated = call(repeat)
        observed = call(strided)
        result["cases"][name] = {
            "native_stride": list(native.stride()),
            "candidate_stride": list(strided.stride()),
            "native_repeat": compare(reference, repeated),
            "output": compare(reference, observed),
            "updated_state": compare(native, strided),
            "native_update": compare(initial, native),
            "strided_update": compare(initial, strided),
        }
        # Diagnostic remedy: explicit compact copy and writeback. This is not
        # installed into the model, nor claimed as a zero-copy implementation.
        strided.copy_(initial)
        compact = strided.contiguous()
        bridged = call(compact)
        strided.copy_(compact)
        result["cases"][name]["explicit_copy_output"] = compare(reference, bridged)
        result["cases"][name]["explicit_copy_state"] = compare(native, strided)
        del owner, strided, native, initial, repeat
    except Exception as error:
        result["cases"][name] = {"error": repr(error)}
    a.output.write_text(json.dumps(result, indent=2))
try:
    shape = (4, 128, 4, 256)
    key = random((2, 4, 256))
    value = random((2, 4, 256))
    kn = torch.zeros(shape, dtype=torch.bfloat16, device="npu")
    vn = torch.zeros_like(kn)
    raw = torch.zeros((4, 2, 128, 4, 256), dtype=torch.bfloat16, device="npu")
    ks, vs = raw[:, 0], raw[:, 1]
    slots = torch.tensor([128, 385], dtype=torch.int64, device="npu")

    def scatter(k, v):
        torch_npu.npu_scatter_pa_kv_cache(
            key=key, value=value, key_cache=k, value_cache=v, slot_mapping=slots, cache_mode="Norm"
        )

    scatter(kn, vn)
    scatter(ks, vs)
    result["cases"]["kv_scatter"] = {
        "native_stride": list(kn.stride()),
        "candidate_stride": list(ks.stride()),
        "key": compare(kn, ks),
        "value": compare(vn, vs),
        "native_key_written": compare(kn[1, 0], key[0]),
        "candidate_key_written": compare(ks[1, 0], key[0]),
    }
except Exception as error:
    result["cases"]["kv_scatter"] = {"error": repr(error)}
a.output.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
a.output.with_suffix(".measure_done").write_text("diagnostic complete\n")
