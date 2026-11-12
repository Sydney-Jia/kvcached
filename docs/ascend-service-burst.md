# Native residency traffic protocol

`tests/ascend/service_burst.py` ports reusable measurement behavior from
`archive/ascend-service-burst-20261105` to the current native adapter. It attaches
to existing, explicitly owned HTTP services. It does not launch models, reserve
devices, inject memory pressure or kill processes. No historical burst performance
result is claimed for this rewritten tool.

## Configuration and runs

Start from `tests/ascend/fixtures/native_burst.json`. Replace service URLs/model
aliases and choose prompts with independently established full-output oracles.
`expected_numbers` accepts only the specified nonnegative sequence with whitespace,
comma or semicolon separators; `expected_text` checks exact text. Native-repeat
stability should be checked before comparing an elastic arm. Token IDs, reasoning,
finish reason and full SSE events are retained when the endpoint supplies them.

```sh
python tests/ascend/service_burst.py --config trace.json --out results/held \
  --abort-file control/STOP
python tests/ascend/service_burst.py --config trace.json --out results/elastic \
  --abort-file control/STOP --dynamic --reference results/held
```

The operator establishes the initial held/native state before either run. Without
`--dynamic`, events are timestamped but no mutation is sent. With it, only explicit
`held`/`elastic` events call `kvcached_native_control`. A capacity rejection or an
incomplete worker response fails the run; mutations are never retried silently.
`control_results` specifies frontend-visible results, not proof that every DP
rank responded. Collect all worker `KVCACHED_GENERIC_CONTROL` logs separately.

Services with `control: none` support native reference traffic. Multiple labels
can represent different models, replicas or independent tenants. Case/phase/SLO
content is fingerprinted for paired runs; endpoint aliases may differ. Checkpoint,
framework build, cache warmup, parallelism and physical pressure must be recorded
and matched externally. A workload hash alone cannot establish those conditions.

## What measurements mean

- Fixed offered arrival times are independent of response completion. Client
  worker saturation appears as dispatch lag and remains part of offered latency.
- Every offered request is counted, including unsent/cancelled/missing responses.
  Successful SSE termination requires a finish reason and `[DONE]`.
- TTFT/end-to-end latency and client average TPOT are retained per request;
  completed-request percentiles include explicit denominators. Average TPOT is
  not per-token p99 or a token-gap measurement. SLO counts are absent unless
  thresholds are explicitly configured.
- Lightweight HTTP queue/preemption metrics are sampled. Driver HBM readings,
  allocator ledger measurements and control latency must be inspected separately;
  quota reduction or HTTP success is not evidence of physical reclaim.
- A fresh output directory is required. Raw streams, controls, offered trace,
  metrics and failure summaries remain on errors. Creating the abort file stops
  new offers, allows bounded in-flight calls to finish, and never stops servers.

The old launcher, quota observer/retry policy, intrusive measurement worker,
pressure setup and historical calibration/result records remain under the archive
tag. Their APIs are not silently emulated by this tool. The protocol tests use
local HTTP fixtures; they do not establish serving performance or NPU correctness.
