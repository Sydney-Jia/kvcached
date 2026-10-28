# Experimental Ascend backend

The Ascend backend implements the native VMM boundary with AscendCL and adds
an opt-in vLLM-Ascend integration. Native allocation, single-engine Qwen
inference, and two-engine reclamation have been tested on Ascend950DT. Do not infer
support for other Ascend devices, engine versions, or execution modes from a
successful build.

## Ownership and integration

- vLLM continues to own requests, logical KV blocks, block tables, and prefix
  cache validity. The Ascend attention implementation consumes separate K/V
  tensor views in its existing layout.
- kvcached owns the physical backing of its KV tensors. AscendCL reserves
  virtual addresses separately from physical HBM allocation. Mapping and
  unmapping pages change physical occupancy without moving the tensor base.
- Tensor storage uses the PyTorch stable ABI and the registered `npu`
  PrivateUse1 device. Native read/write tests cover this storage boundary;
  it does not require an alternative attention kernel.
- A physical device UUID identifies the shared allocation budget across
  processes with different visible-device numbering. Each process retains its
  own tensor addresses and KV contents: capacity sharing is not content sharing.
- Worker synchronization precedes physical unmapping. The existing transactional
  map/unmap and deferred-retirement protocol still controls when a page may be
  reused. Cached prefixes cannot remain valid after their backing is reclaimed.

Ascend's VMM API differs from CUDA in several important details. Physical
allocations use `ACL_HBM_MEM_HUGE`. Virtual-address reservation does not accept
the CUDA preferred-address hint. Each unmap removes one original mapping,
so teardown visits original page boundaries. Local-device access is established
by mapping; peer-device access is outside this backend's current contract.

## Build

Use a dedicated environment containing matching PyTorch and torch_npu packages,
CANN headers/libraries, and the usual kvcached Python dependencies. The extension
requires the same PyTorch stable ABI baseline as other kvcached backends
(PyTorch 2.10 or later).

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
python -m pip install --no-build-isolation .
```

The build selects Ascend when torch_npu is installed and the PyTorch build has
neither CUDA nor HIP enabled. It links against `ascendcl`; a CUDA installation
is not required. The installed `.pth` file registers worker import hooks in
spawned processes as well as the frontend process. Importing only the parent
process's patch module is insufficient for a multiprocess engine.

Install the declared Python dependencies as well. A `--no-deps` wheel install
does not install `wrapt` or `posix_ipc`; a missing `wrapt` can leave the `.pth`
hook inactive while Python continues. Check stderr and the elastic instance's
control membership. Launch vLLM with the interpreter that has kvcached installed
(`python -m vllm.entrypoints.cli.main serve ...` is the CLI module entrypoint).

## Initial engine contract

The integration currently accepts single-device, V1, eager MHA/GQA with one
KV cache group, equal K/V head dimensions, and `cache_dtype=auto`.

```bash
export ASCEND_RT_VISIBLE_DEVICES=<available-device>
export ENABLE_KVCACHED=true
export KVCACHED_CONTIGUOUS_LAYOUT=false
export VLLM_USE_V2_MODEL_RUNNER=0
export KVCACHED_IPC_NAME=example-a
vllm serve Qwen/Qwen3-0.6B --enforce-eager \
  --max-model-len 1024 --max-num-seqs 4 \
  --kv-cache-memory-bytes 536870912 --shutdown-timeout 60
```

Choose an available device and an appropriate budget for the actual hardware.
Use a different `KVCACHED_IPC_NAME` for every engine. Explicit KV bytes describe
the engine's virtual capacity; live physical backing grows through kvcached.
Other memory remains necessary for weights, runtime state, activations, and
the allocator's zero-page backing. Default reserve/prefix-cache policies may
retain additional pages, so an idle engine need not consume zero KV memory.

Sleep mode cannot simultaneously own the same KV storage. Multi-device
parallelism, graphs, KV-transfer connectors, speculative decoding, quantized
KV, compressed/sparse attention, hybrid linear-attention state, and SGLang NPU
integration are outside this initial implementation. Unsupported configurations
are rejected rather than served through a silently static KV allocation.

## Validation

### Larger-model text workloads

The installed package has also been compared with native KV allocation using
Qwen3-8B BF16 on two Ascend950DT machines: 36 response comparisons and 974 output
tokens matched, covering text extraction, code, summaries, 1K/4K/8K retrieval,
and mixed-length/burst requests. This used eager execution, no prefix cache,
zero physical-page reserve, and the optional FIA setting described below.
No runtime quota was published. Stable driver observations confirmed natural
physical return after requests; some first-shape observations also included
workspace changes and must not be attributed entirely to KV reclamation.

The manual probe is `tests/ascend/natural_elasticity.py`, with
`--workload realistic --workload-module <frozen realistic_workloads.py>`.
It preserves full input IDs and outputs. Run it under an idle-device/foreign-work
supervisor. Natural EOS means these checks do not establish sustained long-decode
performance, production tail latency, task-answer accuracy, or default reserve
and prefix-cache behavior. Independent single-card runs do not establish TP.

### Optional FIA compatibility

The tested Ascend950/CANN 9.1 stack can fail in its native paged TND FIA
`sparse_mode=3` path for particular mixed prefill/decode shapes. This was also
reproduced with ordinary NPU tensors, without kvcached. Elastic allocation does
not fix that kernel/runtime failure.

For that environment, explicitly set `KVCACHED_ASCEND_FIA_MODE=causal-window`
before starting `vllm serve`. The worker redirects only keyword calls to paged
TND FIA mode 3, using mode 4 with unlimited past and zero future tokens.
It preserves tensor views, the block table, the attention mask and other
arguments. Eager execution is required. Native mode is the default and changes
no attention calls; this opt-in is not a general CANN fix or support for other
devices, kernels or model architectures.

For a matching native-memory reference, set `ENABLE_KVCACHED=false` and
`KVCACHED_AUTOPATCH=true`, keeping the same FIA option. With both features off,
use `ENABLE_KVCACHED=false KVCACHED_AUTOPATCH=false`. Compare identical
checkpoints and cache histories; changing attention paths between the baseline
and elastic run would confound the result.

An independent synthetic probe compared six geometries against a float32
reference (three repetitions each). This is a separate check from capacity
sharing and does not establish accuracy for arbitrary shapes. The normal worker and the experiment
adapter use the same implementation in `ascend_compat.py`.

The standalone probe is `tests/ascend/fia_reference.py`, with synthetic geometry
in `tests/ascend/fixtures/fia_mixed.json`. After selecting an available device:

```bash
ENABLE_KVCACHED=false KVCACHED_AUTOPATCH=false \
python tests/ascend/fia_reference.py tests/ascend/fixtures/fia_mixed.json \
  --sparse-mode 4 --verify --repeat 3
```

For the separately controlled three-arm capacity/latency experiment, see
[the pressure experiment protocol](ascend-pressure.md). Keep preparation and CPU
checks separate from the completed physical-pressure observations below.

Select an available device explicitly before running hardware tests:

```bash
ASCEND_RT_VISIBLE_DEVICES=<available-device> \
KVCACHED_RUN_ASCEND_TESTS=1 ENABLE_KVCACHED=false \
python -m pytest tests/test_ascend_vmm.py -q
```

The native tests exercise split and compound layouts, one and two KV buffers,
partial reclamation, stable addresses, live-data preservation, transaction abort,
physical-budget accounting, and capacity reassignment between two spawned
processes. They do not establish model accuracy, prefix-hit behavior, graph
compatibility, or a serving-throughput improvement.

Native tests have passed with Ascend950DT, CANN 9.1.0, driver 25.1.rc2,
PyTorch 2.10.0, and torch_npu 2.10.0.post3. The initial engine investigation uses
vLLM metadata 0.23.0 and a customized vLLM-Ascend source snapshot. An exact
engine compatibility matrix must be based on completed model tests, not on
package version strings alone.

### Model and runtime-control checks

The manual checks in `tests/ascend/` use a local Qwen3-0.6B checkpoint,
BF16, eager execution, one device, prefix caching, and a 512 MiB configured
KV capacity budget (not the size of the virtual address reservation).
Run them from the repository root, after checking that the selected
device is available. They start private offline engines, not HTTP services.

```bash
export ASCEND_RT_VISIBLE_DEVICES=<available-device>
export PYTHONPATH="$PWD/tests/ascend:$PYTHONPATH"
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export KVCACHED_CONTIGUOUS_LAYOUT=false
export KVCACHED_PAGE_PREALLOC_ENABLED=false
export KVCACHED_MIN_RESERVED_PAGES=0
export KVCACHED_MAX_RESERVED_PAGES=0
export KVCACHED_MAX_CACHED_TOKENS=4096
export KVCACHED_IPC_NAME=ascend-validation
MODEL=/path/to/local/Qwen3-0.6B

timeout --kill-after=20 1200 python tests/ascend/single.py \
  --model "$MODEL" --output native.json
timeout --kill-after=20 1200 python tests/ascend/single.py \
  --elastic --model "$MODEL" --output elastic.json \
  --compare-baseline native.json

timeout --kill-after=20 1200 python tests/ascend/prefix.py \
  --model "$MODEL" --output native-prefix.json
timeout --kill-after=20 1200 python tests/ascend/prefix.py \
  --elastic --model "$MODEL" --output elastic-prefix.json \
  --compare-baseline native-prefix.json
timeout --kill-after=20 1800 python tests/ascend/dual.py \
  --model "$MODEL" --oracle native-prefix.json --output dual.json
```

Use the same checkpoint, engine build, dtype, and generation settings for the
native oracle and elastic checks. The prefix probe covers zero, one, and two
cached blocks, plus a cache reset. The dual-engine check fills A's cache, lowers
its physical limit, waits for idle reclamation, grows B, and checks A's reclaimed
prefix and concurrent requests against the corresponding native execution path.
It records per-request cached-token counts and physical-budget snapshots.

Cold and cached prefill can produce different greedy continuations in the
tested native Ascend stack, even without kvcached. The tests therefore compare
matching cache states and prefix-construction history instead of assuming
that cold and warm outputs must always be identical. This observation is
not a diagnosis of the native stack's
numerical behavior. Exact token agreement on these probes also does not replace
broader model-quality evaluation or establish a throughput improvement.

The tested Qwen3-0.6B checkpoint revision is
`c1899de289a04d12100db370d81485cdf75e47ca`. A four-round, 16-request comparison
matched all 384 output tokens. An additional eight-round prefix comparison
matched all 128 output tokens and all returned top-five log probabilities.
With two live engines, A's physical KV charge fell from 224 MiB to 112 MiB
after a limit update; B subsequently grew from 112 MiB to 224 MiB. A's reclaimed
probe had zero cached tokens and matched native cold execution. Concurrent
cold and warm requests also matched their native reference paths.
Those are correctness observations on the tested stack, not a compatibility
claim for other models or versions.

These original numbers are kvcached ledger charges, not independently measured
device HBM. The device had ample free capacity: B's growth does not demonstrate
that it required A's release. Those early probes did not establish a
memory-pressure benefit, throughput advantage, or equivalence to Prism.

`tests/ascend/physical_probe.py --output physical.json` adds direct CANN
`aclrtGetMemInfo(ACL_HBM_MEM)` readings, retained-handle/abort controls,
an independent PyTorch recipient, and a PyTorch cached-free control. It checks
three bounded 256 MiB cycles while preserving a live guard page and virtual
addresses. Readings are device-wide and must be rejected if other workloads
interfere. The worker probe now includes the same independent readings beside,
not in place of, the ledger. Physical release still does not imply a serving
performance improvement or identify the physical page IDs used by a recipient.

A subsequent Qwen dual-engine run independently observed a 112 MiB increase in
CANN-reported free HBM when A's charge fell from 224 to 112 MiB. Both sampling
windows were stable. A kept the same process identity; its PyTorch reserved
memory stayed at 1246 MiB. The process-memory readings corroborated this delta
while B's reading stayed unchanged. Recomputed and concurrent cold/warm probes
again matched their native reference histories, and the experiment exited zero.
This establishes physical reclamation in that bounded scenario, not that B
needed the returned capacity or that serving performance improved.

The standalone three-cycle audit did not achieve a clean complete pass: two
cycles showed exactly 256 MiB returned to the driver and consumed by an
independent allocator, but external memory noise invalidated the third-cycle
baseline. Other failed attempts exposed test dtype misuse and unreliable
process-counter increments. Keep these failures in the evidence record.
The optional process observer uses `KVCACHED_AUDIT_PHYSICAL_DEVICE` and
`aclrtDeviceGetBareTgid` to identify its own PID; its values require calibration
and must not automatically replace device-wide driver measurements.

The installed engine emitted forced-worker-exit and semaphore-cleanup warnings
during shutdown; the native baseline also emitted a forced-exit warning.
After the dual-engine run, no matching test processes or live ledger entries
remained. This bounded exit check is not a long-running lifecycle stress test.

### Completed pressure comparison

A later controlled run used the same 93,463,773,184-byte AscendCL physical
ballast in all three arms, with three repetitions of twelve requests. Native
fixed pools, elastic held pools, and elastic reclaimed pools respectively met
the predeclared joint TTFT/TPOT/end-to-end targets for 30/36, 23/36, and 35/36
requests. Every request completed; complete token outputs matched the native
reference. Two clean reclamation windows independently measured exactly 336 MiB
returned to the driver while PyTorch reserved memory stayed unchanged.

This is a bounded capacity-under-pressure result on the stated Qwen3-0.6B stack,
not a general throughput ratio. One reclaimed request still missed its TPOT
target. Execution order was not randomized, physical headroom was intentionally
small, and no continuous B peak-allocation trace was collected. Do not interpret
the result as a measured maximum concurrency or a production recommendation.

### Installed HTTP and shutdown checks

`tests/ascend/installed_serve.py` exercises an installed package through the
ordinary CLI and HTTP endpoints, with no test-side attention or engine patch.
Use a dedicated environment and an external idle-device/foreign-work supervisor.
The probe is specific to the Qwen3-0.6B and optional Qwen2.5-0.5B geometries;
its 112/96 MiB minimum limits must not be copied to unrelated models.

The API frontend's exit does not prove that NPU workers and their control leases
have already retired. The tested runtime defaults to immediate abort. Use a
sufficient `--shutdown-timeout` for native resource teardown and verify both
process exit and unavailable control membership before restarting a service.
CANN unmaps each original mapping separately, so teardown can scale with the
reserved KV range. A native stack capture during slow exit reached the driver's
unmap path. A 60-second grace period completed the tested 896 MiB two-model
case without forced-worker termination; this is not a constant bound for larger
KV capacities. Transient "engine exit is unconfirmed" messages do not by
themselves demonstrate a leak.

A final installed-package comparison used Qwen3-0.6B and Qwen2.5-0.5B
(`060db6499f32faf8b98477b0a26969ef7d8b9987`) on one Ascend950, with
336 MiB configured KV capacity per engine. A concurrent eight-request,
1536-input-token batch made B allocate additional pages before reclamation.
A's physical-page ledger charge decreased from 224 to 112 MiB, and B's from
192 to 96 MiB. Both limits were then restored and both models served requests;
B also continued after A exited. All 608 output tokens from 19 ordinary,
streaming, concurrent, and post-reclamation requests matched the corresponding
independent native references under the same explicit FIA compatibility mode.
All four service processes exited zero with the 60-second grace, without forced
worker termination, remaining worker sockets, or live control membership.
The external supervisor observed no foreign NPU processes and no remaining
owned processes. This was an ample-memory functional check, not a pressure or
performance benchmark; the charge values are ledger readings, not new driver
HBM measurements. Earlier independent physical and pressure evidence remains
separate.

### Resize latency is not zero

`tests/ascend/resize_latency.py` measures control acceptance, observed completion,
driver free-memory deltas, and streamed output arrival times. The optional
`vmm_timing_preload.cpp` observer forwards AscendCL allocation, mapping, release,
and synchronization calls while recording their host durations. Load it only
in the experiment's own processes. The script requires an external idle-device
and foreign-work supervisor; it does not reserve a device.

On the tested Qwen3-0.6B stack, five rounds (three traced on one device, two
untraced on another) observed 35–220 ms for idle release of 112/336 MiB.
An active request could take 5.5–5.9 seconds to reach the lower target while
continuing to generate. Its maximum token arrival gap was 79–84 ms, compared
with 28–46 ms without shrinking. The untraced runs reproduced this short stall.
All 19,840 compared output tokens matched the corresponding no-shrink
references. These are bounded functional/timing observations, not production
tail-latency guarantees or a calibrated estimate of instrumentation overhead.

The idle limit watcher currently polls every 200 ms. Increasing a quota does
not eagerly allocate all permitted capacity or restore evicted prefix contents;
allocation and recomputation costs can appear in later requests. Call durations,
control acknowledgement, and end-to-end request latency are different metrics.

The nonblocking control observer can return incomplete membership during a
concurrent accounting update. A single such observation is not sufficient to
declare that a running service exited. Retry within a bounded deadline while
checking process liveness and the original instance generation; never treat an
incomplete snapshot as proof that physical memory has been returned.
