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
  --kv-cache-memory-bytes 536870912
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
that it required A's release. No memory-pressure benefit, throughput advantage,
or equivalence to the complete Prism system has been established.

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
