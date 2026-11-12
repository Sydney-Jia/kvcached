# Native Ascend adapter: 2026-11-12 checkpoint

This is experimental evidence for the [explicit native V2 adapter](ascend-native-residency.md),
not the legacy autopatch support matrix or a claim of production readiness.

## Fixed sources

- Model runtime: `decad6a`; test-runner commit: `253042c` (no runtime changes).
- Ascend950DT; fixed image digest
  `sha256:8d8e8d7ed793be3dbbca16970aef1814882bbe0e69d289303f79bc250adca224`.
- Installed vLLM: `0.31.0.dev0+gc4df37df`; torch 2.10.0 and torch_npu 2.10.0.post4.
- Both models use the same `NativeElasticWorker`, `NativeResidencyScheduler`,
  ownership/page-union implementation and compiled Ascend VMM backend.
- DeepSeek graph execution additionally uses the previously validated decoder
  wrapper, fixed at `dcb5504ae641e962faeb88089752490a2987fbfb` in the framework
  checkout. It is a framework execution prerequisite, not part of this package
  or an upstream standard feature. The local model checkpoints and image are
  not distributed with this repository.

## Observed acceptance

The inspected upstream snapshot `b2737741193ef12fdc5dbc7cdc72d7826adc61fd`
already declares hybrid attention support and contains CUDA graph profiling /
temporary-cache lifecycle handling in `model_runner_v2.py` and `patches.py`.
This work is not the first implementation of hybrid caches or graphs in
kvcached. Its contribution is the explicit native Ascend storage/ownership
contract and NPU evidence for the tested multi-group execution paths. Upstream
GPU support does not automatically establish Ascend support.

| Workload | Evidence and scope |
|---|---|
| Local quantized Qwen27B, TP1 | Native architecture `Qwen3_5ForConditionalGeneration`; four groups, one allocation, 320 view spans. Independent native and candidate eager / FULL_DECODE_ONLY services. Native-repeat, held, elastic and restored outputs match; 256-token live shrink matches held. Each execution mode passes 23 concurrent comparisons and three post-cancellation comparisons. |
| Qwen graph execution | Four actual NPUGraph captures; serving replay counters increase 26 → 52 → 78 → 588 in the basic sequence. Eager counters remain zero. Bases remain stable through shrink/restore. |
| Local DeepSeek V4.1, TP2/DP2/EP4 | Each rank has 13 groups, four allocations and 59 view spans. Native view geometry is retained. Three states and live shrink match references; all four rank logs are retained. |
| DeepSeek graph execution | Fifteen ordinary decoder layers use the existing wrapper; other work remains eager. Measurement sequences observe 9,900 and 9,840 actual layer replays. The latter reuses captured plans after allocator relief; no new shadow training. Not full-model graph coverage. |
| DeepSeek eager continuation | Same runtime build, wrapper switched to eager: long-prefix restore, 23 concurrent comparisons, live shrink, 90-second idle and three subsequent requests pass. Graph pools remain resident, so this is not a pure eager performance baseline. |
| Actual Qwen scheduler preemption | Eight independent long prompts, 768 generated tokens, prefix cache reset before each arm. Native, native-repeat, held and elastic each incur 306 preemptions; all eight full content outputs match. Native ≈70–86 s, held ≈75–91 s, elastic ≈95–111 s. This is a functional stress sample, not an SLO benchmark. |
| True cross-engine borrowing | Donor Qwen27B with 8 GiB configured KV; receiver Qwen0.6B with 4 GiB configured KV. With constant live ballast, receiver capacity allocation is rejected while donor is held, then succeeds after donor reclamation. Both models' concurrent outputs match their own standalone references. Donor full restoration while receiver is resident is cleanly refused; control and subsequent inference remain usable. After receiver exit and observed driver release, donor restores under the same ballast with matching outputs. |
| Native fixed-pool pressure control | Same card, native 8 GiB donor and identical receiver. Native allocation/execution are unchanged; a worker subclass adds only a read-only memory query. Receiver fails with the same explicit capacity rejection and donor continues with matching outputs. Independent ballast is 45,176 MiB versus 45,162 MiB for held/elastic; measured free HBM before receiver is 4,916.96 versus 4,904.40 MiB. Held/elastic is the within-process causal pair; the native arm is a near-matched separate control. |
| CPU regression | `253042c`: 2,115 passed, 42 skipped, 30 warnings, 754.53 s. The first run failed during collection due to a missing manifest entry and shell error propagation; that record is retained separately. Skipped device tests are not NPU passes. |

The installed runtime's multi-DP HTTP control returns only the first engine's
results. All-rank logs are necessary; HTTP success alone is not acceptance.

## Physical effects and costs

- Qwen 2 GiB configured KV maps 2,016 MiB. Idle reclamation returns 1,664 MiB,
  leaving 352 MiB: graph/eager control duration 430/406 ms, restoration 529/527 ms.
  Live reclamation returns 1,440 MiB in 382/365 ms. These are completed worker
  control durations, not isolated API time or request latency percentiles.
- The 8 GiB pressure donor maps 8,176 MiB. Reclamation returns 8,096 MiB in
  7.16 s, leaving 80 MiB; full restoration takes 8.18 s. This larger case exposes
  a material granularity/transaction cost. It is not an invisible optimization.
- A receiver process exiting did not immediately return driver-visible HBM.
  In the completed trial, sufficient free capacity appeared 4.13 s after the
  first post-exit observation. Waiting used measured driver capacity, not a
  fabricated budget refund or a hard-coded sleep taken as proof.
- Some pressure observations include failed-receiver cleanup. Do not attribute
  the entire driver-free delta to KV retirement. Report KV mapped bytes,
  driver samples, ballast bytes and receiver lifetime independently.
- The failed receiver hits the allocator's unchanged 95% physical budget;
  this is a real capacity constraint with headroom, not proof of raw driver OOM.

## Failures and unresolved boundaries

1. A previous DeepSeek run encountered a MegaMoE rank-barrier timeout. Later
   eager/graph/idle continuations pass, but no causal fix for that isolated
   timeout is established. Long-running distributed stability remains open.
2. An earlier 3,072-token forced-`ignore_eos` test produced mismatches in all
   candidate comparisons. The unchanged native repeat also differed in all
   eight outputs, often after generated end markers. This reference is not
   stable; it proves neither candidate correctness nor a candidate-only defect.
   The controlled 768-token result above is additional evidence, not a rewrite
   of this failure as a pass.
3. Normal growth has bounded allocator relief, but no distributed physical
   admission/wait queue. Persistent real capacity shortage may still terminate
   a worker. Expected `held` capacity refusals return `ok:false` locally; policy
   changes are not atomic across ranks.
4. Async scheduling, connectors, PP/CP and simultaneous sleep ownership are
   explicitly rejected pending lifetime contracts. Unknown allocation/view
   layouts fail validation; there is no guarantee for every model or framework
   revision. Multimodal inputs were not tested.
5. Reclamation uses conservative device synchronization and retains cached
   prefixes. It can reduce throughput, and retained pages can limit savings.
   No comparison with CaMem on this exact new runtime was performed.

Two specific capacity bugs were independently diagnosed and fixed: separate
PyTorch free arenas obstructed VMM growth, and the one-second budget sample
remained stale immediately after returning those arenas. The final build
fences, releases unused allocator storage, explicitly refreshes the budget,
and retries once after a clean rollback. Four DeepSeek ranks recorded unchanged
live torch allocation/KV mapped bytes and increased driver-free/budget values;
the previously failing full restoration then completed. This allocator handoff
is separate from reclaiming unused KV pages.

## Reproducing a controlled check

Use fixed native/candidate services with identical checkpoint, quantization,
parallelism, logical KV capacity, scheduling and graph configuration. Follow
the entry-point options in the adapter guide. Do not apply the legacy patcher.

```sh
python tests/ascend/native_residency_http.py \
  --reference-url http://127.0.0.1:8000 \
  --candidate-url http://127.0.0.1:8001 \
  --model MODEL_ALIAS --out results/native-residency-smoke
```

The portable probe is a serial smoke helper, added after the recorded hardware
run; it is not the exact script that produced the table. It preserves complete
responses and rejects unsuccessful controls. It does not deploy services,
reserve cards, perform pressure tests or check every DP rank automatically.

For the full acceptance protocol, also preserve per-rank controls, real replay
counters, live-shrink paired outputs, concurrency/cancel cases, actual preemption
metrics and a native-repeat oracle. Pressure tests must keep ballast fixed
within each trial, verify a capacity-specific receiver failure before reclaim,
prove receiver startup and concurrent inference afterward, test clean restore
refusal, then observe physical return before restoring. Stop only experiment
tasks and invalidate a run if foreign device work enters.
