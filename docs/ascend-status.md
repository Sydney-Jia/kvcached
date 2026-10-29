# Ascend implementation checkpoint

Status date: 2026-10-29. This document describes the experimental Ascend branch,
not the support matrix of every kvcached backend. Installation and configuration
remain in [ascend.md](ascend.md); probes are indexed in the
[Ascend test guide](../tests/ascend/README.md).

## Implementation boundary

The implementation adds an AscendCL VMM backend and a vLLM-Ascend adapter for
single-device, V1, eager, ordinary MHA/GQA execution. It preserves vLLM's request
and block-table semantics while hooking block allocation/reclamation through
`ElasticBlockPool` and replacing the KV tensors' backing storage. The NPU runner
receives separate K/V views in its native layout.

| Area | Implementation |
|---|---|
| Device memory | `csrc/inc/ascend_vmm.hpp`: address reservation, physical allocation, mapping, original-boundary unmapping, release and device identity |
| Tensor storage | `csrc/ftensor.cpp`, `csrc/inc/torch_utils.hpp`: PrivateUse1/NPU storage backed by stable virtual addresses |
| Engine adapter | `kvcached/integration/vllm/ascend.py`: NPU runner/worker import hooks, allocation/view adaptation and explicit configuration checks |
| Device dispatch | `kvcached/device.py`, `tp_ipc_util.py`: NPU device contexts and synchronization before physical unmapping |
| Reused mechanisms | Existing page/block management, transactional operations, prefix eviction, deferred retirement, runtime limits and lifecycle protocols |

The functional artifact is commit
`37404b7ced8b8f814cd70e8db2b9978d6e92e875`, compared against upstream baseline
`3170a0abef78c41a309ccb545822e8d326947993`. Validation scripts through
`f5c35aee9d5715cb0f64d2d53b048a0b6111ba51` do not change that functional artifact.
The tested wheel SHA-256 is
`8b01784771e59f052a1e562ed0004b43245af3fff9c22aaae3ad1cfa76ee56b8`.

## What has been demonstrated

| Check | Observation | Limit |
|---|---|---|
| Physical return | Independent driver free-memory readings matched 112/336 MiB reclamation in clean sampling windows | Allocator ledgers alone are not proof of HBM return |
| Cross-engine pressure | Fixed-pool / elastic-held / elastic-reclaim arms met a joint latency target for 30/36, 23/36 and 35/36 requests | Two Qwen3-0.6B engines, real allocation pressure, non-default 64 MiB guard; all requests eventually completed |
| Different-model lifecycle | Qwen3-0.6B and Qwen2.5-0.5B reclaimed and recovered; 608 tokens across 19 requests matched native references | Ample-memory functional check, not an additional pressure result |
| Larger model | Qwen3-8B: 36 comparisons, 974 tokens matched native allocation, including 8K input and mixed/burst requests | Eager, prefix caching off, preallocation/reserve off; bounded runs |
| Resize latency | Idle completion 35–220 ms; active shrink could take 5.5–5.9 s while output continued, with a 79–84 ms maximum token-arrival gap | Not zero overhead or production tail-latency guarantees |
| Large-service co-location | Native DeepSeek TP2/DP4/EP8 plus elastic single-card Qwen8B: 76 structured checks and 302 complete responses; observed card peak 84.93 GiB | DeepSeek remained static; about 11.07 GiB headroom remained, so this is not a capacity-pressure benefit result |

These are distinct experiments, not one combined benchmark. The installed
Qwen stack used Ascend950DT, CANN 9.1, PyTorch/torch_npu 2.10 and custom
vLLM/vLLM-Ascend 0.23 source snapshots. The DeepSeek service used a different
fixed framework/CANN stack. The optional FIA causal-window workaround and its
limitations are documented in [ascend.md](ascend.md).

## Explicitly outside the accepted contract

- Multiple KV groups, compressed/sparse or quantized caches, persistent hybrid
  state, and DeepSeek's multi-spec cache layout.
- TP/PP/DP/CP, graphs, speculative decoding, V2 runner, KV connectors and
  SGLang NPU integration. Generic backend support does not imply NPU coverage.
- Concurrent ownership of the same KV storage by camem sleep and kvcached.
- Automatic workload/SLO-based quota policy, model-weight eviction and the
  complete Prism system.
- Production long-run stability, all Ascend devices/framework releases, or
  upstream acceptance. No upstream PR has been submitted for this adaptation.

## Next acceptance gates

1. Complete the supported single-card path: default page reserves, prefix
   retention/eviction with long input/output, cancellation, repeated recovery
   and long-run cleanup. Measure latency as well as physical return.
2. Add multi-spec storage integration while preserving native descriptors,
   strides, offsets and aliases. Prove that release by one cache group cannot
   unmap a page still used by another group or an in-flight operation.
3. Validate the target engine revision and cooperative execution groups
   separately; then add speculative decoding and graphs as separate gates.
4. Demonstrate capacity benefit under controlled pressure with native,
   elastic-held and elastic-reclaim arms, accounting for both donor and receiver
   latency and prefix-recomputation costs.

## Repository boundaries

Keep reusable implementation, generic probes and portable documentation in this
repository. Keep model files, wheels, raw logs, result archives, exact machine
bindings and operator-specific launch configuration in the surrounding private
workspace. Probe presence does not mean it has passed on every configuration.
Historical experimental results should remain immutable rather than being
rewritten when a newer checkpoint supersedes them.
