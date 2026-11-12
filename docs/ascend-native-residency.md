# Native Ascend storage residency (experimental)

This adapter preserves the framework's native KV allocation geometry and
logical block policy. It substitutes the storage backing those native views
with the Ascend VMM backend. It does not normalize every model into one KV
tensor layout or replace the native cache manager with a model-specific pool.

The [2026-11-12 checkpoint](ascend-native-checkpoint-20261112.md) records the
shared-build Qwen/DeepSeek, graph, physical borrowing and failure evidence,
including measured overhead and unresolved boundaries.
The [development policy](ascend-development.md) identifies the single integration
line, frozen prototypes and contract-based experiments.

## Responsibilities

| Layer | Responsibility |
|---|---|
| Native vLLM scheduler and cache managers | Request block assignment, prefix hashes and aliases, state copies, native pins and reuse watchers |
| `native_residency.py`, `native_scheduler.py` | Publish group-qualified ownership with a monotonically increasing execution epoch |
| `native_storage.py` | Substitute the native byte allocations and translate actual tensor offsets, shapes and strides into storage spans |
| `residency.py` | Union every owner's byte ranges at physical-page granularity; preserve pages shared by multiple views/groups |
| `physical_residency.py` | Map missing pages, initialize only new pages, fence execution before retirement, roll back failed preparation |
| Ascend VMM backend | Reserve virtual addresses and create/map/unmap/free CANN physical handles |

The block table continues to contain native cache slots. It does not become a
device page table. Virtual addresses remain stable across physical retirement
and restoration; tensor views and attention kernels retain their native ABI.

Ascend device UUIDs observed on the test A5 host are shared by separately
budgeted devices. The backend key therefore combines the UUID with the
runtime-resolved physical device ID. Using either the UUID alone or a local
visible index would respectively merge different HBM pools or split one pool
across processes. The explicit user-to-physical API is documented by
[CANN](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/920beta1/API/runtimeapi/aclcppdevg_03_2286.html).
The opt-in `test_physical_keys_are_distinct_and_visibility_stable` checks this
contract against the compiled extension on the selected devices.

The earlier native-hybrid prototype supplied the physical-identity fix and
its hardware test. Its native ND storage requirement is implemented directly
in `csrc/ascend_tensor.cpp` using torch_npu's tensor factory; a separate
model-specific storage wrapper is no longer needed. Per-DP control logging,
completed-transition timing and separate PyTorch allocator trimming are also
retained. The old model-specific layout and ownership entry points are not
used by this adapter.

## Lifecycle

1. After profiling, synchronize and return unused PyTorch allocator arenas.
   The independent VMM allocator cannot reuse cached PyTorch allocations.
   Record this startup handoff separately from later KV reclamation.
   Allocate and map the complete native backing during initialization. Native
   view construction, warmup and graph capture run against valid storage.
2. After each native scheduling decision, publish owners for **all** requests,
   not just requests scheduled in that iteration. Cached prefixes with zero
   request references remain owners. Copy and zero endpoints are protected.
3. Before execution, map missing pages needed by that snapshot. Newly mapped
   pages are zeroed without touching still-resident pages.
4. In elastic mode, synchronize device work before unmapping pages with no
   remaining owner. A Python function returning is not a device fence.
5. Idle distributed ranks can execute dummy forwards. Before native dummy
   metadata preparation (which can itself write state), protect its null and
   temporary state slots. The next scheduling step can retire this ownership
   only through the same completion fence.

Reclaiming an unused physical page does not preserve its contents. This adapter
therefore retains prefix-cache owners instead of silently treating every
zero-reference native block as disposable. Physical page sharing and retained
prefixes can limit reclamation even when most request slots are free.

## Selecting the adapter

The entry points are explicit, separate from the legacy automatic patch path:

```sh
export ENABLE_KVCACHED=0 KVCACHED_AUTOPATCH=0 VLLM_USE_V2_MODEL_RUNNER=1
vllm serve MODEL \
  --worker-cls kvcached.integration.vllm.ascend_generic_worker.NativeElasticWorker \
  --scheduler-cls kvcached.integration.vllm.native_scheduler.NativeResidencyScheduler \
  --no-async-scheduling
```

Use a fixed built wheel and the usual model-specific native serving options.
Models still need a working native vLLM-Ascend implementation. This adapter
does not implement missing attention operators or model architectures.

The validated interface currently comes from the tested Ascend MRV2 runtime
with vLLM `0.31.0.dev0+gc4df37df`. A version string outside the legacy autopatch
range does not itself establish incompatibility. Allocation, view, ownership,
executor transport and graph contracts must be checked against the actual
installed source.

`kvcached_native_control` provides `status`, `elastic`, `held`, and
`trim_allocator`. The latter explicitly trims PyTorch's separate caching
allocator; its effects must not be attributed to KV page retirement. The
control RPC is intended for controlled validation, not a public service API.

Physical growth can also be blocked by unused arenas in PyTorch's separate
allocator after a new workload shape. On the backend's explicit
`capacity_exhausted:` rejection, the native adapter waits for a clean local
growth rollback, fences the device, releases only unused PyTorch arenas,
explicitly refreshes the shared physical budget, and retries once. The explicit
refresh avoids using the normal sampler's one-second-old budget immediately
after memory has been returned. It records driver free bytes, torch
allocated/reserved bytes, the shared budget and KV mapped bytes before/after
this handoff. Count these separately from KV page
reclamation. Unknown device errors and poisoned transactions are never retried;
the 95% capacity limit and all live/prefix owners remain unchanged.

A rejected `held` control returns `ok: false` with a rejection reason and keeps
that worker's previous mode. It does not throw an expected capacity exception
through the MP executor's response queues. Callers must check every result;
HTTP 200 alone is not success. This is not an atomic cross-rank policy change:
some workers can succeed while another rejects. The current DP frontend also
returns only its first engine's results, so validation retains all worker
control logs. Normal serving growth still needs admission if one retry cannot
obtain capacity; this change does not implement scheduler waiting.

`KVCACHED_TRACE_GRAPHS=1` instruments actual NPUGraph construction and replay
calls. Compare counters before and after serving requests: capture settings
alone, or startup warmup replay counts, do not prove serving uses graphs.

`KVCACHED_TRACE_RESIDENCY=1` additionally logs each rank's residency, dummy
preparation and execution phases, with a Python stack dump if one worker call
lasts more than 45 seconds. This opt-in diagnostic changes neither the page
policy nor collective synchronization. Disable it for latency comparisons.

## Current boundaries

- Native buffers must expose the validated raw byte-allocation and block-first
  view contracts. Unknown views fail explicitly before elastic use. There is
  no model-name allowlist, but this is not a claim that every layout is tested.
- Synchronous scheduling and MP executor snapshot transport are required.
  Async scheduling, connectors, PP/CP and a second physical allocator owning
  the same tensors require additional lifetime contracts and are rejected.
- The native logical block pool still has its configured virtual capacity.
  This adapter does **not yet** add distributed physical-capacity admission or
  waiting to that scheduler. Returning idle capacity is not a guarantee that
  every future growth request can succeed under arbitrary contention.
- Local growth and retirement preparation have rollback. A partial physical
  retirement commit poisons the worker because released contents cannot be
  recovered by remapping empty pages. Cross-rank recovery is not established.
- Device-wide fencing is conservative and can add visible latency. Stable
  addresses avoid graph rebuilding; they do not make remapping free or permit
  retirement while kernels still access a page.

The dependency-free ownership/layout/failure tests live in
`test_residency_contract.py`, `test_native_residency_snapshot.py`, and
`test_physical_residency.py`. Model acceptance must additionally compare native,
held, elastic and restored outputs, observe driver-level memory, exercise real
graph replay, and prove another process can use returned capacity. An allocator
counter changing alone is insufficient evidence.
