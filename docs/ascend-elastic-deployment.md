# Ascend elastic residency deployment — suggested configuration

Deployment dated 2026-11-20; measured on the `feat/ascend-elastic-residency`
branch. Only newly created, fixed-commit personal test services were used.

This page records the exact configuration behind the 36-trace measurement and
why each switch is set that way. It is a deployment note, not a claim that the
configuration is universal: the split below assumes one LLM and one VLM on one
card, tensor parallel 1, and a shared physical KV budget.

## Topology

| Item | Value |
|---|---|
| Host | `10.83.185.115`, container `zzy-a5-native` (root) |
| Device | Ascend 950DT, 96 GB HBM, physical card 7 (`ASCEND_RT_VISIBLE_DEVICES=7`) |
| torch / torch_npu | 2.10.0+cpu / 2.10.0.post4 |
| vLLM | 0.31.0.dev0+gc4df37df |
| Model | `/models/Qwen3.8-27B-w8a8-mxfp8` (hybrid GDN + vision tower) |
| LLM service | `127.0.0.1:8031`, elastic residency (no capacity admission) |
| VLM service | `127.0.0.1:8030`, capacity admission |

Both services load a full copy of the model, so the card holds two weight
copies (~72 GB) and the shared KV budget is what remains. That is why every
control below is about KV headroom rather than weights.

## Why the two engines use different modes

The capacity protocol (prepare-before-allocate over all candidates plus a
shared budget sampled against mapped bytes) is safe but expensive on a hot
path: the 36-trace baseline produced 485 `CAPACITY_RELIEF` events and
4987.7 s. The elastic path skips the admission RPC and the candidate pre-map,
but it has no backpressure — a request that cannot be backed either waits or
fails.

So the split is: the LLM, which carries nearly all requests, runs elastic; the
VLM, which sees a handful of multimodal requests, runs capacity admission and
queues when the ledger is tight instead of faulting.

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

LLM only (`_llm.sh`):

| Variable | Value | Meaning |
|---|---|---|
| `KVCACHED_NATIVE_CAPACITY` | `0` | Elastic: no scheduler-side admission |
| `KVCACHED_NATIVE_LAZY_STORAGE` | `1` | Reserve the pool, map pages on demand |
| `KVCACHED_NATIVE_EXEC_RECLAIM_UTILIZATION` | `0.70` | Reclaim synchronously once the shared ledger is 70% mapped, keeping working headroom |
| `KVCACHED_NATIVE_ELASTIC_WAIT_S` | `60` | Wait this long for a peer-held budget before failing |

VLM only (`_vlm.sh`):

| Variable | Value | Meaning |
|---|---|---|
| `KVCACHED_NATIVE_CAPACITY` | `1` | Admission queues instead of faulting |
| `KVCACHED_NATIVE_LAZY_STORAGE` | `1` | Sparse backing under admission |

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
export KVCACHED_NATIVE_CAPACITY=1
export KVCACHED_NATIVE_LAZY_STORAGE=1
/opt/a5-python/bin/vllm serve /models/Qwen3.8-27B-w8a8-mxfp8 \
  --served-model-name qwen3.8 --host 127.0.0.1 --port 8030 \
  --tensor-parallel-size 1 --data-parallel-size 1 --quantization ascend --dtype auto \
  --max-model-len 18432 --max-num-seqs 16 --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.435 --enable-prefix-caching \
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

| Run | Total wall | Relief events | Server down |
|---|---|---|---|
| Capacity baseline (both engines) | 4987.7 s | 485 | no |
| This configuration, 36 traces | **3745.5 s** | **18** | no |
| This configuration, 6 traces | 891 s | ≤ 20 | no |

Every one of the 36 traces finished faster than the baseline; the per-trace
median dropped from 522.6 s to 272.7 s.

Host load matters for these numbers: the baseline was recorded at load
average ≈ 34 on the 192-core host, this run at load ≈ 120–200. The comparison
above is therefore conservative, but not a controlled same-condition A/B.

## Known issues

- **Do not enable `KVCACHED_NATIVE_RETIRE_ASYNC`.** An asynchronous unmap can
  race an in-flight kernel and fault the device with error `507011`
  (`device error type 3`). The default is now `0`; retirement happens in the
  synchronous pressure transition.
- Elastic mode has no admission backpressure. When the peer engine holds the
  shared ledger, the step blocks up to `ELASTIC_WAIT_S` and then raises. This
  is why the VLM runs capacity admission.
- `task_159` returns five to six VLM HTTP 400s
  (`Number of samples, ... must be non-negative`); this predates the branch
  and is also present in the capacity baseline.
- The shared KV budget is derived from sampled free device memory. With two
  model copies resident, the budget is inherently tight; every switch above
  exists to keep a few hundred MiB of working headroom.
