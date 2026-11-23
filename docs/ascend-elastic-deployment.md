# Ascend elastic residency deployment — suggested configuration

Deployment dated 2026-11-20, gate and tensor-parallel notes added 2026-11-23;
measured on the `feat/ascend-elastic-residency` branch. Only newly created,
fixed-commit personal test services were used.

This page records the exact configuration behind the 36-trace measurement and
why each switch is set that way. It is a deployment note, not a claim that the
configuration is universal. Two layouts were measured: one LLM and one VLM on
one card at tensor parallel 1 (36 traces), and each engine at tensor parallel
2 on its own card pair (6 traces).

## Topology

| Item | Value |
|---|---|
| Host | `10.83.185.115`, container `zzy-a5-native` (root) |
| Device | Ascend 950DT, 96 GB HBM, physical card 7 (`ASCEND_RT_VISIBLE_DEVICES=7`) |
| torch / torch_npu | 2.10.0+cpu / 2.10.0.post4 |
| vLLM | 0.31.0.dev0+gc4df37df |
| Model | `/models/Qwen3.8-27B-w8a8-mxfp8` (hybrid GDN + vision tower) |
| LLM service | `127.0.0.1:8031`, elastic gate (no capacity admission) |
| VLM service | `127.0.0.1:8030`, elastic gate (no capacity admission) |

Both services load a full copy of the model, so one card holds two weight
copies (~72 GB) and the shared KV budget is what remains. That is why every
control below is about KV headroom rather than weights. At tensor parallel 2
each engine owns a card pair instead: weights halve per card and the measured
per-rank KV pool grows from 12.27 GiB to 29.94 GiB.

## Why the gate replaces the capacity hot path

The capacity protocol (prepare-before-allocate over all candidates plus a
shared budget sampled against mapped bytes) is safe but expensive on a hot
path: every step whose owner set changed ran a physical transition (fence,
reclaim, ensure). The 36-trace baseline produced 485 `CAPACITY_RELIEF` events
and 4987.7 s, and the step profile showed `transition` at 12–25% of wall time
even after the admission RPC was removed.

The gate keeps the elastic mapping and moves retirement off the decode path:

- a decode step only maps the pages it requires (`arena.ensure(required)`);
- unowned pages are retired at an **empty scheduling point**, where the fence
  is free;
- cold prefix blocks are evicted only **after** a rank reports pressure, never
  on the assumption of it, because their pages stay owned until evicted and
  each eviction costs a recompute later;
- near the budget wall the scheduler also evicts cold prefixes without an idle
  point, and the worker's first capacity rejection runs one synchronous
  transition so those pages are really released before it retries;
- with tensor parallel > 1 the scheduler samples per-rank pressure on a slow
  heartbeat while busy (`KVCACHED_NATIVE_GATE_PROBE_EPOCHS`) instead of
  fencing, and retires only at empty steps.

The gate must be registered **after** reshape publishes the owner table;
registering before it captured `None` and made every idle retirement a silent
no-op that still reported success.

Both engines use the same gate. Running the VLM under capacity admission was
measured and rejected: its video requests were rejected repeatedly against a
small pool and hit the client timeout (six `vlm HTTP 0` events in one run),
which extended the slowest traces to ~3000 s.

## Environment variables

Shared by both services (`_env.sh`):

| Variable | Value | Meaning |
|---|---|---|
| `PYTHONPATH` | extracted source tree | Run the branch without installing |
| `ASCEND_RT_VISIBLE_DEVICES` | `7` | Pin the card |
| `KVCACHED_NATIVE_CAPACITY` | `0` | Base value; overridden per service below |
| `KVCACHED_NATIVE_LAZY_STORAGE` | `1` | Sparse physical mapping independent of admission |
| `KVCACHED_NATIVE_RETIRE_ASYNC` | `0` | Keep asynchronous retirement off (see Known issues) |
| `KVCACHED_NATIVE_PIPELINE_PREPARE` | `0` | Cross-step speculative prepare stays off |
| `KVCACHED_NATIVE_STEP_GRANTS` | `1` | Only used when capacity admission is on |
| `KVCACHED_NATIVE_SPARE_BYTES` | `0` | No spare page retention |
| `KVCACHED_NATIVE_PROFILE` | `1` | Per-stage timing in the worker profile |
| `KVCACHED_NATIVE_GATE` | `1` | Map-only steps, idle/pressure retirement |
| `KVCACHED_NATIVE_ENSURE_WAIT_S` | `60` | Bound on waiting for a peer-held budget |
| `KVCACHED_NATIVE_IDLE_RECLAIM_UTILIZATION` | `0.75` | Ledger fraction that starts idle retirement |
| `KVCACHED_NATIVE_IDLE_PREFIX_EVICT` | `8` | Cold prefix blocks evicted per idle/pressure point |
| `KVCACHED_NATIVE_GATE_PROBE_EPOCHS` | `16` | Busy-step period for the pressure-only probe (TP > 1) |

LLM only (`_llm.sh`):

| Variable | Value | Meaning |
|---|---|---|
| `KVCACHED_NATIVE_CAPACITY` | `0` | Elastic: no scheduler-side admission |
| `KVCACHED_NATIVE_LAZY_STORAGE` | `1` | Reserve the pool, map pages on demand |
VLM only (`_vlm.sh`):

| Variable | Value | Meaning |
|---|---|---|
| `KVCACHED_NATIVE_CAPACITY` | `0` | Same gate as the LLM |
| `KVCACHED_NATIVE_LAZY_STORAGE` | `1` | Sparse backing under the gate |

`KVCACHED_NATIVE_EXEC_RECLAIM_UTILIZATION` and
`KVCACHED_NATIVE_ELASTIC_WAIT_S` remain set in the LLM script for the
non-gated path; with the gate installed the map-only branch runs instead.

Knobs that exist but stay at defaults in this deployment:
`KVCACHED_NATIVE_RETIRE_UTILIZATION=0.95`,
`KVCACHED_NATIVE_RETIRE_TARGET=0.7`, `KVCACHED_NATIVE_RETIRE_GRACE_S=3`,
`KVCACHED_NATIVE_RETIRE_BATCH_BYTES=4MiB`,
`KVCACHED_NATIVE_RETIRE_INTERVAL_S=0.5`, `KVCACHED_NATIVE_IDLE_RECLAIM_S=2`.

## Launch commands

LLM:

```bash
source /opt/a5-sdk/set_env.sh
export KVCACHED_NATIVE_CAPACITY=0
export KVCACHED_NATIVE_LAZY_STORAGE=1
export KVCACHED_NATIVE_EXEC_RECLAIM_UTILIZATION=0.70
export KVCACHED_NATIVE_ELASTIC_WAIT_S=60
export KVCACHED_NATIVE_GATE=1
/opt/a5-python/bin/vllm serve /models/Qwen3.8-27B-w8a8-mxfp8 \
  --served-model-name qwen3.8 --host 127.0.0.1 --port 8031 \
  --tensor-parallel-size 1 --data-parallel-size 1 --quantization ascend --dtype auto \
  --max-model-len 49152 --max-num-seqs 32 --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.543 --enable-prefix-caching \
  --distributed-executor-backend kvcached.integration.vllm.native_executor.NativeCapacityUniProcExecutor \
  --limit-mm-per-prompt '{"image": 32, "video": 4}' \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --additional-config '{"enable_cpu_binding": false}' \
  --worker-cls kvcached.integration.vllm.ascend_generic_worker.NativeElasticWorker \
  --scheduler-cls kvcached.integration.vllm.native_scheduler.NativeResidencyScheduler \
  --no-async-scheduling
```

VLM:

```bash
source /opt/a5-sdk/set_env.sh
export KVCACHED_NATIVE_CAPACITY=0
export KVCACHED_NATIVE_LAZY_STORAGE=1
export KVCACHED_NATIVE_GATE=1
/opt/a5-python/bin/vllm serve /models/Qwen3.8-27B-w8a8-mxfp8 \
  --served-model-name qwen3.8 --host 127.0.0.1 --port 8030 \
  --tensor-parallel-size 1 --data-parallel-size 1 --quantization ascend --dtype auto \
  --max-model-len 18432 --max-num-seqs 16 --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.435 --kv-cache-memory-bytes 2040109465 --enable-prefix-caching \
  --distributed-executor-backend kvcached.integration.vllm.native_executor.NativeCapacityUniProcExecutor \
  --limit-mm-per-prompt '{"image": 32, "video": 4}' \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --additional-config '{"enable_cpu_binding": false}' \
  --worker-cls kvcached.integration.vllm.ascend_generic_worker.NativeElasticWorker \
  --scheduler-cls kvcached.integration.vllm.native_scheduler.NativeResidencyScheduler \
  --no-async-scheduling
```

`NativeCapacityUniProcExecutor` is used for TP1 even in elastic mode: it is
the in-process executor, so no worker IPC is paid for the single rank.

## Startup behavior

Both warm up with a dummy batch, and sparse mapping makes that batch resident.
The worker therefore retires the warm-up set at the end of
`compile_or_warm_up_model`:

```
KVCACHED_NATIVE_STARTUP_RECLAIM {"rank": 0, "mapped_bytes": 0}
```

Without that step the second engine measured a 1.1 GiB KV pool (it needs at
least 1.48 GiB for the 18432-token VLM context) and refused to start. With it
the service order no longer matters; this deployment still starts the LLM
first.

Expected steady state: LLM pool ≈ 12.27 GiB, VLM pool ≈ 2.23–2.36 GiB.

## Replay configuration

| Item | Value |
|---|---|
| Traces | `traces/e5b_dual49`, 36 traces |
| Concurrency / jitter | 6 / 20 s |
| Sampling | `temperature=1.0, top_k=20, top_p=0.95` |
| Output length | fixed to the trace's `output_tokens` |
| Media routing | images as `image_url` + `data:image/jpeg`, videos as `video_url` + `data:video/mp4` |

## Measured results

Tensor parallel 1, both engines on one card:

| Run | Total wall | Median | Max | ok / failed | Relief | Server down |
|---|---|---|---|---|---|---|
| Capacity baseline (both engines) | 4987.7 s | 522.6 s | — | 497 / 5 | 485 | no |
| Elastic, no gate | 4904.7 s | 272 s | 3096 s | 491 / 11 | 1 | no |
| **Gate, both engines elastic** | **2887.2 s** | **293 s** | 968 s | **497 / 5** | 0 | no |
| Reference: the previous personal branch | 2835.1 s | 287 s | 949 s | 497 / 5 | 8 | no |

The five failures are `task_159`'s known VLM HTTP 400s in every run.

**These numbers predate the gate registration fix.** In that code the gate was
registered before reshape published the owner table, so idle retirement always
returned 0 while the scheduler had to treat the ledger as pressured and evicted
cold prefixes every empty step. The throughput came from map-only execution
alone. Re-measure before quoting these totals.

Tensor parallel 2, each engine on its own card pair (LLM cards 4,5 / VLM
cards 6,7), 6 traces:

| Run | Total wall | Median | Max | ok / failed | Server down |
|---|---|---|---|---|---|
| TP1, same configuration | 883–891 s | ~220 s | ~882 s | 67 / 0 | no |
| **TP2** | **899.5 s** | 216 s | 881 s | **67 / 0** | no |

Per-rank KV pool at TP2 is 29.94 GiB versus 12.27 GiB at TP1. The 6-trace
total is dominated by one fixed-length decode, so TP2 is within ~2% there;
the larger pool is the part that should matter under sustained load.

Host load matters: the reference runs were taken at load average ≈ 34 on the
192-core host, other runs at 100–200. Treat cross-run wall times as indicative
rather than a controlled A/B.

## Tensor parallel behavior

At TP1 with an in-process executor the scheduler drives retirement directly
in its own process; no RPC is involved. For multi-process ranks the scheduler
calls `kvcached_native_gate` once per idle point (and on a slow heartbeat
while any rank reports pressure); each rank retires its own unowned pages and
returns `pressure` / `reclaimed`. `KVCACHED_NATIVE_GATE=1` with
`world_size > 1` requires `NativeCapacityExecutor` and fails at startup
otherwise.

Hardware notes from the validation host:

- Only cards that exist as real devices in the container can form a
  communicator. On this host `/dev/davinci0-3` are not usable; TP2 works on
  4,7 and fails HCCL init (`RunAicpuCommInit`, `aicpu exception 0x2a`) on 0,1.
  Plain vLLM without kvcached fails the same way, so this is environment, not
  the gate.
- Two tensor-parallel engines cannot share the same card pair: the second
  engine's HCCL communicator fails to initialize. Give each engine its own
  pair.

## Known issues

- **Do not enable `KVCACHED_NATIVE_RETIRE_ASYNC`.** An asynchronous unmap can
  race an in-flight kernel and fault the device with error `507011`
  (`device error type 3`). The default is `0`; retirement happens at idle
  points and in the synchronous pressure transition.
- A capacity rejection waits up to `KVCACHED_NATIVE_ENSURE_WAIT_S`; running the
  VLM under capacity admission instead produced repeated client timeouts
  (`vlm HTTP 0`), so both engines use the gate.
- `task_159` returns five to six VLM HTTP 400s
  (`Number of samples, ... must be non-negative`); this predates the branch
  and is also present in the capacity baseline.
- The shared KV budget is derived from sampled free device memory. With two
  model copies resident, the budget is inherently tight; every switch above
  exists to keep a few hundred MiB of working headroom.
