# Single-device two-service burst experiment

This opt-in harness measures whether reclaiming idle physical KV capacity from
one live service helps another service drain its queue. It is a serving test,
not just a `set_instance_memory_limit` API test. It uses the installed kvcached
package; the worker extension only adds measurement RPCs.

## Scope and prerequisites

- Two independent vLLM-Ascend processes on one reserved device, with separate
  `KVCACHED_IPC_NAME` values and a shared IPC namespace for physical accounting.
- Validated MHA/GQA integration, eager BF16 execution, prefix caching disabled.
  This test does not establish hybrid attention, graph, TP or prefix-cache support.
- A local compatible model/tokenizer. The default task is a 5,000-token input
  followed by a deterministic counting instruction. Both services use the same
  checkpoint to isolate allocation effects. Correct counting is a smoke test,
  not a general model-quality evaluation or proof of bitwise equivalence.
- Named worker RPC and development endpoints must stay inside an isolated
  trusted environment. Device availability and ownership must be checked by the
  launcher before and during execution. The harness does not reserve devices.

Run `tests/ascend/service_burst.py --help` from a fixed source revision. Set the
tested runtime's Python and CANN environment, and add that revision's
`tests/ascend` directory to `PYTHONPATH`. All modes need `--output`, `--model`
and `--abort-file`. Use a new output directory for each service pair.

## Experiment arms

1. `start --native`: ordinary vLLM-Ascend pools, each with `--initial-mib` bytes.
2. `start`: kvcached services with a larger reserved virtual address range and
   initial physical quotas. `trace` leaves the quotas unchanged.
3. The same pair with `trace --dynamic`: donor shrink followed by receiver grow.
   A fixed open-loop arrival trace is replayed, including requests during resize.

The direct causal comparison is arm 2 versus arm 3: same engine integration,
virtual capacity, requests and physical pressure; only quota actions differ.
The native arm provides context but also changes the allocator/integration.

By default, dynamic actions call driver-memory RPCs before, between and after
resize operations. These synchronize the device and query `npu-smi`; **their
latency is included in requests**. Use those runs for detailed memory evidence.
Then repeat with `--dynamic --light-control`, which retains IPC observations and
HTTP metrics but removes these intrusive RPCs from timed control actions.
Keep both sets of results rather than replacing the slower measured runs.

## Trace configuration and pressure

`--trace` takes JSON with `initial_mib`, `ballast_bytes`, `slo`, `phases` and
`events`. Each phase has `name`, `duration_s`, and per-service `rates` (`A`, `B`).
Each event has `at_s`, `donor`, `receiver`, `donor_mib`, `receiver_mib`.
Arrivals are generated independently of response completion. Phase boundaries
reset the periodic arrival clock, so inspect the saved `arrivals.json`; the
low-rate stream can contain a shorter interval at a boundary.

Calibration must precede freezing the trace. `calibrate` submits batches of
1/2/4/8 requests. Exclude cold initialization from rate selection and retain its
record. Choose a hot arrival rate that exceeds the smaller capacity's service
rate but can be handled by the larger capacity. Do not silently tune rates after
seeing a comparison result.

The reference setup explicitly sets preallocation reserve bounds to 12/28 pages
and cached-token reserve to zero. These are **experimental settings**, not
upstream defaults. They retain a measurable idle pool, and must be reported.
For a 36-layer GQA model with 8 KV heads, head size 128 and BF16, KV uses
144 KiB/token; 2 MiB physical pages across 72 K/V layer tensors form a 144 MiB
allocation bundle. Requested limits therefore round down to bundle boundaries.

Optional ballast uses actual CANN allocations outside PyTorch, with independent
driver-delta validation. It requires `KVCACHED_PRESSURE_AUDIT_AUTHORIZED=1` and
must be used only on a reserved device. Keep the **same exact bytes** in all arms
and retain the allocator's physical headroom. This simulates a smaller serving
memory envelope; it does not prove two small checkpoints naturally fill a large
device. Ballast is released in `finally` after traces and pressure probes.

## Evidence and separate probes

- `trace`: saves every offered request, stream chunks, output, client dispatch
  lag, samples, controller errors, trace hash and ballast allocation/release.
  Report success over all offered requests, both hot and low-load tenants,
  TTFT/end-to-end latency, stream gaps, queue depth, and resize convergence.
  Client-average TPOT is not per-token tail latency.
- `pressure-proof`: warm both pools, apply fixed ballast, raise A's quota while
  B keeps idle physical pages, run eight requests; then reclaim B and repeat
  with **A's quota unchanged**. Capture physical admission/accounting, driver
  free memory and service time. This distinguishes logical quota from usable
  physical capacity. It is a separate mechanical probe, not a traffic replay.
- `correctness`: four alternating resize cycles, with different expected
  counting ranges across requests and services. This helps expose accidental
  state mixing that identical prompts would hide.
- `prepare`, `status`, `stop`: inspect/reset the pair or stop only its recorded
  frontend PIDs after verifying process start identity. A host supervisor should
  check workers have also exited; container lifecycle remains the launcher's job.

Quota convergence does not mean the receiver has already mapped all future
capacity. Growth happens as requests require pages. Likewise, an idle donor may
temporarily release more than the steady-state quota transfer, then regrow as
requests return. Compare steady-state charged capacity, driver observations and
workload behavior; do not equate a transient free-memory spike with permanent
capacity gain.

## Initial experiment findings (2026-11-05)

This is a limited experiment, not a production acceptance result. Two independent
Qwen3-8B services shared one A5 device, using a fixed 51.5546875 GiB CANN ballast,
async scheduling, eager execution, and no prefix cache. A 310-second trace offered
200 requests, with 5,000 input tokens and deterministic outputs. The declared
joint latency target was TTFT <= 5 s, completion <= 12 s, and client-average
TPOT <= 80 ms. The installed functional package was fixed at `37404b7`.

| Arm | Complete passing runs | Joint target attainment | A-hot TTFT P95 |
| --- | ---: | ---: | ---: |
| Native fixed pools | 2 | 51.0-66.5% | 10.40-12.75 s |
| kvcached fixed quotas | 3 | 26.0-39.0% | 14.10-32.32 s |
| Dynamic, light control | 2 | 64.5-85.0% | 4.42-5.49 s |
| Dynamic, intrusive driver observations | 1 | 73.5% | 4.78 s |

Ranges are observed run minima/maxima, not confidence intervals. All 1,600
requests in these eight complete runs matched the same 87-token native reference.
The low-load B cohort during A-hot passed its latency targets in both light runs
(8/8 each); its completion P95 was 4.06-4.53 s. A separate four-cycle test used
different expected outputs across services and passed 16/16 requests.

The separate physical-pressure probe held A's quota at 2,560 MiB in both cases.
Reclaiming B increased driver free memory by 1,584 MiB without changing Torch
reserved memory. A then grew from 1,872 to 2,448 MiB of physical KV capacity,
increased maximum running requests from two to three, and finished eight requests
in 12.01 s instead of 16.22 s. This probe had an idle donor; the traffic tests
above assess low-load service protection.

Important counterevidence and open issues:

- One earlier dynamic run returned all 200 correct outputs but its final control
  observation failed. It is excluded from the complete-run table. The harness
  now retries transient lock-free observation misses and fails incomplete runs.
- One earlier fixed run stalled both workers in NPU event synchronization:
  29/200 requests completed, 171 timed out. Its requests, logs and native stacks
  were retained. The root cause remains unresolved; subsequent successful runs
  do not establish stability.
- Light controls took 0.40-4.38 s and increased request preemption counts. Safe
  physical page reclamation does not imply zero interruption or recomputation at
  the engine scheduler layer.
- Reverse B-hot TTFT P95 was 5.43 s and 16.07 s in the light runs; the second was
  worse than both native runs. Overall ranges overlap. Timed quota changes are
  not yet a validated adaptive controller.
- Fixed kvcached was slower than native in this configuration. Page rounding,
  integration/admission paths and unisolated host CPU are possible confounders;
  attribution is incomplete. Ballast requested bytes were identical, while the
  device-wide allocation delta differed by 0-4 MiB between runs.

Prioritize reproducing the event wait and explaining fixed-quota baseline costs
before adding a load-aware policy. That policy should account for queued and
in-flight KV demand, donor latency and preemption cost, rather than treating
quota convergence alone as success.
