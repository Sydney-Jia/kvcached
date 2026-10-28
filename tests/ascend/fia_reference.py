# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Replay captured FIA geometry with ordinary NPU tensors and synthetic values.

No kvcached import, VMM mapping, model, second engine, or pressure ballast.
The host must admit either an idle device or an explicitly bounded shared run.
"""
import argparse
import json
import time

import torch
import torch_npu  # noqa: F401

parser = argparse.ArgumentParser()
parser.add_argument("case")
parser.add_argument("--repeat", type=int, default=3)
parser.add_argument("--kv-blocks", type=int)
parser.add_argument("--last-query-tokens", type=int)
parser.add_argument("--allocator-limit-mib", type=int)
parser.add_argument("--min-free-mib", type=int, default=0)
parser.add_argument("--decode-count", type=int, choices=range(8), default=7)
parser.add_argument("--api", choices=["legacy", "v2"], default="legacy")
parser.add_argument("--sparse-mode", type=int, choices=[3, 4], default=3)
parser.add_argument("--verify", action="store_true", help="Compare nonzero seeded inputs with CPU float32 causal GQA")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
case = json.load(open(args.case))
if args.kv_blocks is not None:
    assert max(v for row in case['kwargs']['block_table']['values'] for v in row) < args.kv_blocks
    for name in ('key', 'value'):
        case['kwargs'][name]['shape'][0] = args.kv_blocks
if args.last_query_tokens is not None:
    lengths = case['kwargs']['actual_seq_lengths']
    assert 0 < args.last_query_tokens <= case['kwargs']['actual_seq_lengths_kv'][-1]
    lengths[-1] = lengths[-2] + args.last_query_tokens
    case['kwargs']['query']['shape'][0] = lengths[-1]
if args.decode_count != 7:
    kw = case['kwargs']
    last_length = kw['actual_seq_lengths'][-1] - kw['actual_seq_lengths'][-2]
    count = args.decode_count
    kw['actual_seq_lengths'] = list(range(1, count + 1)) + [count + last_length]
    kw['query']['shape'][0] = count + last_length
    kw['actual_seq_lengths_kv'] = kw['actual_seq_lengths_kv'][:count] + kw['actual_seq_lengths_kv'][-1:]
    kw['block_table']['values'] = kw['block_table']['values'][:count] + kw['block_table']['values'][-1:]
    kw['block_table']['shape'][0] = count + 1
case['kwargs']['sparse_mode'] = args.sparse_mode
if args.sparse_mode == 4:
    case['kwargs'].update(pre_tokens=2147483647, next_tokens=0)
torch.manual_seed(args.seed)
torch.npu.set_device(0)
if args.allocator_limit_mib is not None:
    import ctypes

    runtime = ctypes.CDLL('libascendcl.so')
    get_info = runtime.aclrtGetMemInfo
    get_info.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    get_info.restype = ctypes.c_int
    free_bytes, total_bytes = ctypes.c_size_t(), ctypes.c_size_t()
    assert get_info(1, ctypes.byref(free_bytes), ctypes.byref(total_bytes)) == 0
    assert free_bytes.value >= args.min_free_mib * 1024**2, 'Driver reports insufficient headroom'
    assert 0 < args.allocator_limit_mib <= 256
    torch.npu.set_per_process_memory_fraction(args.allocator_limit_mib * 1024**2 / total_bytes.value, 0)
    print('MEMORY_LIMIT', json.dumps({'torch_allocator_limit_mib': args.allocator_limit_mib,
        'driver_free_mib': free_bytes.value / 1024**2,
        'scope': 'PyTorch caching allocator only; runtime memory is monitored separately'}), flush=True)


def decode(name, value):
    if not isinstance(value, dict) or "shape" not in value:
        return value
    dtype = getattr(torch, value["dtype"].removeprefix("torch."))
    device = "npu:0" if value["device"].startswith("npu") else value["device"]
    tensor = torch.empty_strided(value["shape"], value["stride"], dtype=dtype, device=device)
    if "values" in value:
        tensor.copy_(torch.tensor(value["values"], dtype=dtype, device=device))
    elif name == "atten_mask":
        tensor.copy_(torch.ones(value["shape"], dtype=dtype, device=device).triu(1))
    elif args.verify and name in ("query", "key", "value"):
        tensor.copy_((torch.randn(value["shape"], dtype=torch.float32) * 0.5).to(dtype))
    else:
        tensor.zero_()
    return tensor


positional = [decode(str(i), value) for i, value in enumerate(case["args"])]
keywords = {key: decode(key, value) for key, value in case["kwargs"].items()}
torch.npu.synchronize()
reference = None
if args.verify:
    q, k, v = [keywords[name].cpu().float() for name in ("query", "key", "value")]
    table = keywords['block_table'].cpu().long()
    nh, nk = keywords['num_heads'], keywords['num_key_value_heads']
    bs, dim = keywords['block_size'], q.shape[-1]
    expected, start = [], 0
    for row, (end, kvlen) in enumerate(zip(keywords['actual_seq_lengths'], keywords['actual_seq_lengths_kv'])):
        qi = q[start:end]
        indices = table[row, :(kvlen + bs - 1) // bs]
        ki, vi = [x[indices].reshape(-1, nk, dim)[:kvlen].repeat_interleave(nh // nk, dim=1) for x in (k, v)]
        scores = torch.einsum('qhd,khd->hqk', qi, ki) * keywords['scale']
        allowed = torch.arange(kvlen)[None, :] <= (kvlen - (end - start) + torch.arange(end - start)[:, None])
        scores.masked_fill_(~allowed[None, :, :], -float('inf'))
        expected.append(torch.einsum('hqk,khd->qhd', scores.softmax(-1), vi))
        start = end
    reference = torch.cat(expected)
    print('ORACLE_REFERENCE', json.dumps({'shape': list(reference.shape), 'dtype': 'float32',
        'seed': args.seed, 'semantics': 'right-aligned causal GQA over block-table gathered KV',
        'atol': 0.005, 'rtol': 0.02}), flush=True)
if args.api == 'v2':
    for old, new in [('num_heads', 'num_query_heads'), ('scale', 'softmax_scale'),
                     ('actual_seq_lengths', 'actual_seq_qlen'), ('actual_seq_lengths_kv', 'actual_seq_kvlen')]:
        keywords[new] = keywords.pop(old)
    invoke = torch_npu.npu_fused_infer_attention_score_v2
else:
    invoke = torch_npu.npu_fused_infer_attention_score
print("GEOMETRY", json.dumps({'api': args.api, 'verify': args.verify, 'case': case}), flush=True)
for iteration in range(args.repeat):
    begin = time.monotonic()
    print("CALL", iteration, flush=True)
    output = invoke(*positional, **keywords)
    torch.npu.synchronize()
    print("RETURN", iteration, [list(t.shape) for t in output],
          "seconds", time.monotonic() - begin, flush=True)
    if reference is not None:
        actual = output[0].cpu().float()
        diff = actual - reference
        passed = bool(torch.allclose(actual, reference, atol=0.005, rtol=0.02))
        print('ORACLE', json.dumps({'iteration': iteration, 'passed': passed,
            'max_abs_error': diff.abs().max().item(),
            'relative_l2_error': (diff.norm() / reference.norm().clamp_min(1e-12)).item()}), flush=True)
        assert passed, 'FIA differs from independent causal GQA reference'
print('TORCH_PEAK', json.dumps({'allocated_bytes': torch.npu.max_memory_allocated(),
                               'reserved_bytes': torch.npu.max_memory_reserved()}), flush=True)
