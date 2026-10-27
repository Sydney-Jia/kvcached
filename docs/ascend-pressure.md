# Ascend three-arm experiment

This is an experimental protocol, not a performance result. Run only on an
available device. Real physical pressure additionally requires explicit operator
authorization; an empty `npu-smi` snapshot is insufficient. Record an externally
arranged exclusive reservation separately from an explicit user instruction to
test an idle shared-server device. Never manufacture a reservation from idleness.
The latter mode requires a fresh idle check and aborting our run on foreign work.
Monitoring cannot guarantee that a new foreign process will avoid an allocation
failure before detection; use enforced isolation when zero interference is required.

## Question and controls

Can reclaiming idle engine A's KV backing let engine B finish more requests
within a latency target, while preserving outputs and accounting for A's
subsequent recomputation?

| Arm | KV capacity | A at B's arrival |
| --- | --- | --- |
| native | 448 MiB static pool per engine | Keeps its allocation |
| held | 896 MiB logical ceiling per engine | Keeps its mapped KV |
| reclaim | Same elastic ceilings as held | Receives a shrink request |

The causal comparison is **held versus reclaim**. Native versus elastic alone
is insufficient: changing B's configured logical ceiling can improve a result
even when no physical reclamation is needed. A workload-aware best static
partition may also outperform the chosen fixed partition; these three arms do
not establish superiority over every possible static configuration.

Both engines use the same Qwen3-0.6B checkpoint, BF16, single-device eager V1,
12 maximum sequences, and 2048 batched tokens. A caches four distinct 800-token
prompts and generates 16 tokens each, targeting 448 MiB mapped KV. B receives
12 simultaneous 640-token prompts and generates 64 tokens each. Prefix caching
is enabled for A and disabled for B in all arms. The trace is fixed and hashed.

An observed footprint different from the target invalidates the setup. Do not
silently alter the workload until a favorable result appears.

## Physical pressure versus software limits

First run an ample-capacity control without physical ballast. Held and reclaim
should not gain request capacity from A's release when B already has enough
physical memory. This is a negative control, not a fixed-physical-budget result.

For a pressure run, warm both engines and their intended workload. Derive one
fixed ballast size from the native calibration, leaving 64 MiB driver-reported
free headroom. Use that **same byte count** in all three pressure arms. Ballast
is an ordinary NPU allocation in A's worker, separate from kvcached accounting.
Verify its actual driver-memory delta. Neither a tensor's declared size nor a
lowered kvcached limit proves physical pressure.

The pressure arms consequently have the same remaining physical envelope,
including weights, KV and workspaces. Differences in activation/workspace
requirements are real costs and must remain in the measurements. Unexpected
driver changes, foreign jobs, allocation failures or different ballast sizes
invalidate the comparison; they must not be hidden as successful runs.

## Timing and correctness

B's offered arrival time is recorded **before** A's shrink request, so control
delay is included in TTFT. The runner measures client-observed first output,
completion, and average TPOT using the engine's streamed steps. It does not
substitute initialization time or batch wall time for per-request TTFT.

The exploratory thresholds are TTFT <= 5 s, average TPOT <= 100 ms, and total
latency <= 30 s. They are declared in the trace before measurement and are not
a production SLA. Count completions satisfying every threshold over **all 12
offered requests**. Queue acceptance is not successful service capacity.
Report timeouts and invalid runs; latency percentiles over completed requests
alone must not conceal failures.

The worker sends bounded progress snapshots to the parent during generation.
If the engine fails or exceeds its deadline, keep completed and pending requests
under `partial_requests` with the original offered-load denominator. These
observations explain the failure; they do not turn an invalid or interrupted
run into a successful performance comparison.

Save full output tokens, actual cached-token counts, driver memory, control
state, and A's returning-request latency. Compare matching native execution
histories. Batch/prefix changes can affect floating-point paths, so any token
mismatch needs investigation; do not automatically attribute it to either
corruption or harmless numerics. A faster but unexplained-mismatch run does not
pass correctness acceptance.

## Execution and safety

The entrypoint is `tests/ascend/three_arm.py`; its helpers are in the same
directory. Add that directory and the fixed release to `PYTHONPATH`. Use the
same runtime, trace, checkpoint, and experiment commit in every arm.

Pin the native `ascend` platform and its required general plugins, and record
their resolved function modules for both engines. This environment can also
contain `ascend_vllm` and duplicate plugin names; a matching package version
alone does not establish identical execution paths. The worker rejects a
non-native platform or unexpected plugin selection before constructing an
engine. Keep the same explicit plugin allowlist and Python hash seed across
all arms; do not mix earlier uncontrolled runs into this comparison.

The host supervisor must check health and actual processes immediately before
launch, monitor foreign process entry, write the abort file, and clean up only
processes identified by this experiment's unique token or exact, run-unique
inherited log descriptor. Process-title changes can overwrite the environment
memory exposed by `/proc`; an environment-only check can misidentify an engine
and miss an orphan after reparenting. A shared log directory prefix is not a
sufficient identity. Safe process signalling must handle PID reuse using pidfds.
The host supervisor is also responsible for recording
the real environment and verifying exit; a request-level PASS is insufficient.

Pressure mode requires `--lease-file`, `--node`, `--physical-device`, and a
positive `--ballast-bytes`. A record must match node/device and have a start time
and expiry with at least 120 seconds remaining. A reservation uses
`exclusive: true` and a real `reservation_reference`. An explicit user-directed
idle-device test instead uses `authorization_kind: user_directed_idle_device`,
`pressure_authorized: true`, a nonempty `authorization_reference`,
`exclusive: false`, `external_reservation: false`, and
`foreign_process_policy: abort_own_run`. Neither record enforces access control.
The worker also rejects ballast allocation unless
`KVCACHED_PRESSURE_AUDIT_AUTHORIZED=1` is set by the supervisor. Cleanup releases
ballast before shutting down model workers.

Run the CPU-only protocol checks with:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 ENABLE_KVCACHED=false \
python -m pytest tests/test_ascend_pressure_protocol.py \
  tests/test_ascend_pressure_stream.py \
  tests/test_ascend_process_identity.py -q
```

Before publishing numbers, complete all arms, repeat the paired workload with
predeclared order/thresholds, review contamination records and token differences,
and distinguish a capacity improvement from SLO, throughput and latency gains.
