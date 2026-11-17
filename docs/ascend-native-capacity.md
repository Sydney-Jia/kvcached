# Native Ascend capacity closure — implementation candidate

Implementation dated 2026-11-16; TP1/TP2 hardware checkpoint added 2026-11-17.
Development stays on `feat/ascend-generic-kv`. See the
[new hardware evidence and limits](ascend-native-checkpoint-20261117.md) separately
from the [previous fixed-runtime checkpoint](ascend-native-checkpoint-20261112.md).
Only newly created, fixed-commit personal test services were used.

## Capability comparison and evidence

| Capability | Upstream kvcached function | Native NPU implementation in this change | Verification / remaining work |
|---|---|---|---|
| Virtual versus physical capacity | Large virtual pool with lazy physical backing | Opt-in lazy native byte arenas; native tensor layout and block capacity stay fixed | TP1 hybrid startup: 80 MiB mapped within 8176 MiB reserved; whole-device startup peak not measured |
| Online return / regrowth | Map demanded pages and retire disposable pages | Complete owner union, fence, retire before growth; rollback newly mapped pages on clean failure | TP1 candidate real-pressure borrowing, idle convergence and output checks passed |
| Physical-capacity admission | Elastic pool exposes usable capacity to native scheduling | Explicit scheduler/executor admission; prepare candidate slots on all TP workers before native allocation | TP1 physical borrowing and TP2 asymmetric physical rejection / rollback / recovery passed on recorded runtimes |
| Prefix retention and pressure | Bounded prefix retention and quota-triggered eviction | Native eviction API, cold-block limit, pressure retry; preserve active blocks, aliases, reuse watchers | CPU alias/watcher checks; TP1 hit counters, eviction/recompute and latency recorded |
| Idle convergence | Limits can cause reclamation without new demand | Reconcile after output processing, cancellation, and explicit budget updates | TP1 eager/graph: last-request and cancellation cleanup passed without new demand |
| Runtime budget | Revisioned limits, observations, convergence | `kvcached_native_budget` RPC with compare-and-set revision, per-rank results, pending versus converged | TP1/TP2 target changes, waiting/recovery and cancellation passed; upstream kvctl ABI remains absent |
| Hybrid layouts | Multiple supported attention/cache geometries | Same storage/span/ownership contracts; no model-name branch | Candidate: 4-group hybrid and 1-group ordinary layouts; older 13-group DeepSeek result remains separate |
| Graph / parallel execution | Framework-specific graph and distributed integration | Pre-back startup dummy slots; existing stable-address execution; new admission limited to synchronous single-DP TP | TP1/TP2 captured replay and lifecycle passed on recorded runtimes; cross-DP, PP/CP, async, connectors unsupported |

Upstream references: [README](https://github.com/ovg-project/kvcached/blob/main/README.md),
[capacity/pool integration](https://github.com/ovg-project/kvcached/blob/main/kvcached/integration/vllm/patches.py),
[memory control](https://github.com/ovg-project/kvcached/blob/main/examples/02_memory_control/README.md),
[prefix policy](https://github.com/ovg-project/kvcached/blob/main/examples/09_prefix_caching/README.md).
These links describe current upstream; they do not imply this fork automatically
inherits every latest framework integration or a universal multi-model scheduler.

## What changed, and what was already available

The Ascend backend already supplied CANN address reservation, physical handles,
map/unmap/free, granularity queries and a shared physical-capacity guard. Native
vLLM already supplied request allocation, cache hashes, references and preemption.
The existing generic adapter supplied validated native byte layouts and page
ownership. This change connects those components; it does not replace the NPU
driver, attention kernels, native block tables, or per-model cache layouts.

New code boundaries:

- `native_capacity.py`: device-independent budget/revision/convergence semantics.
- `native_admission.py`: native allocation guard and prefix policy. No model or
  cache-spec class names; no replacement of the native scheduler algorithm.
- `native_executor.py`: explicit synchronous RPC bridge in the engine process.
  Its registry is scoped to the actual engine configuration object.
- `native_scheduler.py`: admission, ownership publication, completion/idle cleanup.
- `native_storage.py` and `ascend_generic_worker.py`: optional lazy initialization,
  metadata-only view guards, validated dummy backing, physical preparation and
  fail-closed execution. A reshape that tries to copy/convert unbacked data is
  rejected before launching that operation.
- `physical_residency.py`: retire unowned pages before growth, using existing
  completion fencing and transaction/rollback logic.

## Execution sequence

1. The framework determines logical KV geometry. In capacity mode, native byte
   allocations reserve addresses; physical pages are materialized for validated
   null views and startup dummy accesses, not the whole virtual pool.
2. Before warmup/capture, materialize the native dummy state slots. Do not change
   physical mappings while a graph is being recorded. Startup scratch/weights/
   activations and the dummy floor are **not** bounded by the serving KV limit.
3. The scheduler enters a serialized allocation step. Native managers compute
   block demand, as before. Before native allocation mutates the pool, the guard
   asks every TP worker to back candidate free slots, prefix hits and possible
   native copy sources. Native initialization/zero/copy can address all groups.
4. Workers use complete owner unions. They fence previous device access, release
   unowned pages, then map/zero missing backing. A clean capacity miss is returned
   as data by every rank, not thrown halfway through a TP collective response.
5. A failed grant drops speculative candidate backing on all ranks and returns
   `None` from native `allocate_slots`. The existing scheduler waits/preempts;
   there is no fake inference result or fabricated token progress. Idle retries
   use a small backoff so cancellation/control can still be processed.
6. Successful native scheduling publishes its full request/prefix/pin/watcher/
   copy/zero snapshot. Execution refuses missing backing instead of making an
   uncoordinated late physical allocation. Valid model execution is unchanged.
7. After the output or cancellation is processed, publish current owners again,
   enforce the cold-prefix policy and return idle physical pages. This also runs
   for the last request, without requiring another request to trigger cleanup.

CPU owner removal never proves device completion. The candidate retains the
conservative full-device fence; event-based narrowing remains future work.

## Limits and tradeoffs

- The virtual/logical ceiling remains fixed. Grow/regrow means backing slots
  within that ceiling; increasing `num_blocks` live is not required or implemented.
- `limit_bytes` counts mapped KV pages, not all NPU memory. `None` means no extra
  per-instance cap; the backend's physical guard still applies. Limit zero is
  accepted but cannot eliminate mandatory null/live state: report `pending`.
- Lowering a limit retains existing live state. New physical growth above the
  target is refused. Required working sets may exceed the target; the controller
  must increase it or cancel demand. No deadline/fairness/automatic quota policy
  is promised by this adapter.
- A prefix cap counts **cold native pool blocks**, not tokens or bytes. Active
  blocks and native reuse watchers are excluded. This differs from upstream's
  token cap; the distinction is explicit in the API.
- Hard pressure currently evicts all eligible cold prefixes before one retry.
  Prefix hits used by the allocation, existing live state and watchers survive.
  This is a conservative initial policy, not an optimal cache-hit-rate policy.
- Candidate slots and potential CoW source IDs are backed across all groups.
  This can overestimate physical demand, especially for dissimilar group layouts.
  A future typed per-group allocation/copy preview can reduce that cost without
  model-specific branches. Do not claim current packing efficiency is optimal.
- Each allocation can add synchronous worker RPCs and completion work. Correctness
  and liveness come first; compare overhead before claiming throughput gains.
- TP prepare is not atomic. All results are checked by rank, ticket and revision;
  clean partial grants are reconciled before the scheduler proceeds. Fatal driver,
  transport, quarantine or rollback failures fail the engine rather than pretend
  to preserve unknown state.
- New capacity mode requires synchronous single-DP execution and no batch queue,
  sleep mode, KV connector or PP/CP. Existing legacy-mode DP acceptance does not
  establish support for cross-DP physical admission. Full DeepSeek graph coverage
  remains a separate framework capability.

## Opt-in launch contract

Use a new fixed-commit release and separate controlled service. Do not repoint
an active deployment at the mutable Git checkout. Keep all original model,
dtype, tokenizer, graph, and parallel settings in the experiment manifest.

```bash
export ENABLE_KVCACHED=0
export KVCACHED_AUTOPATCH=0
export KVCACHED_NATIVE_CAPACITY=1
# Examples only: choose budgets after measuring weights / non-KV / dummy floor.
export KVCACHED_NATIVE_LIMIT_BYTES=2147483648
export KVCACHED_NATIVE_PREFIX_BLOCKS=64
# Optional active-step reuse of already mapped, currently unowned pages.
# Default 0; this example permits up to 256 MiB per worker within its KV limit.
export KVCACHED_NATIVE_SPARE_BYTES=268435456

# Append to the pinned native vLLM-Ascend service's existing arguments:
--worker-cls kvcached.integration.vllm.ascend_generic_worker.NativeElasticWorker
--scheduler-cls kvcached.integration.vllm.native_scheduler.NativeResidencyScheduler
--distributed-executor-backend kvcached.integration.vllm.native_executor.NativeCapacityExecutor
--no-async-scheduling
--kv-cache-memory-bytes 8589934592
```

In this mode the last option supplies native **logical geometry**, while the
environment limit bounds serving KV backing. Native startup checks still honor
`gpu_memory_utilization`; choose it for the actual physical startup footprint.
Do not assume two instances with utilization 0.9 will pass native startup merely
because their KV allocations are lazy. Initial profiling uses native temporary
caches and still needs real device capacity. Verify this on hardware.

### Avoiding repeated physical map/unmap work

`KVCACHED_NATIVE_SPARE_BYTES` retains a bounded set of previously mapped pages
while requests remain active. It does not allocate that reserve in advance,
increase logical capacity, or make old KV contents valid. Native zero/copy and
cache-validity rules are unchanged. Spare bytes count toward the same physical
KV limit. Idle cleanup, a changed budget revision, and partial-rank admission
rollback discard spares; a clean local growth rejection drops them before
allocator relief. A peer needing memory must still use the budget protocol;
this is not an automatic cross-engine reclamation scheduler.

The tradeoff is fewer VMM operations for some temporarily retained HBM. Tune
the bound against concurrency and measured headroom, not the model name.
Active state and valid retained prefixes can keep an idle engine above zero.
Physical retirement still uses the existing completion fence.

The validated layout also caches immutable block-to-page geometry, bounded to
65,536 page offsets on the host. It never caches ownership or backing validity.
An unchanged, validated physical target skips redundant map/reclaim walks.
For controlled profiling, `KVCACHED_NATIVE_PROFILE=1` exposes cumulative host
call times in worker status. These nested timings include CPU submission and
driver calls; they are not device-kernel durations and must not be added together.

Budget RPC (the HTTP endpoint must be enabled only for controlled experiments):

```json
{"method":"kvcached_native_budget","kwargs":{"action":"status"},"timeout":120}
```

```json
{"method":"kvcached_native_budget","kwargs":{"action":"set","limit_bytes":1073741824,"prefix_max_blocks":16,"expected_revision":0},"timeout":120}
```

The executor returns a one-element list containing controller status and all
rank observations. `status` is read-only; `reconcile` is an explicit maintenance
action. A changed target's acknowledgement is distinct from `converged`.
Query `kvcached_native_control/status` separately for driver free memory,
allocation ledger and stable addresses. Mapped-byte convergence is not proof
that another process can already allocate the capacity: measure that receiver.
Upstream `kvctl` does not yet control this native scheduler protocol.

## Verification before promotion

Two same-host A5 cards are sufficient: TP1 two-engine sharing on one, then TP2
consistency across both. A four-card DeepSeek run is not needed for the initial
capacity closure. Use available native models as structural coverage; no model
gets its own implementation or permanent branch.

| Check | Required evidence | Status for this candidate |
|---|---|---|
| CPU budget + transactions | Live contents survive rejection; retained-page union; stale revisions rejected | Covered by unit tests |
| CPU allocation boundaries | Native `allocate_slots`/`BlockPool`, prefix aliases, pending copies, partial TP results | CPU contracts and real TP1 native startup passed |
| TP1 lazy startup | Larger logical reservation with smaller mapped footprint; record startup peak, graph/dummy floor, driver HBM | Passed for recorded runtime; not a full startup peak measurement |
| TP1 donor/receiver | Same real pressure and workload; receiver cannot grow in held baseline, can grow after donor release | Passed fixed-physical-pressure held-prefix/release comparison; not a three-implementation performance benchmark |
| Prefix retain/evict | Repeated shared prefixes before/after budget changes; actual hit counters, exact outputs, latency | Passed with real hit counters and exact post-eviction output comparison |
| Active shrink + idle convergence | Long decode and cancellation; preserve outputs; idle mapped bytes decrease without new traffic | TP1 eager/graph and TP2 graph lifecycle passed; zero-target test invokes waiting/preemption, not pause-free shrink |
| Exhaustion / recovery | Both active working sets exceed capacity; no worker crash, no fake tokens, bounded retry; recovery after budget/peer release | TP1 physical refusal/recovery and quota waiting/recovery passed |
| TP2 asymmetric capacity | One rank rejects while another can map; all replies recorded; no partial forward; recovery succeeds | Passed on newer pinned runtime: rank 0 physically rejected, rank 1 granted 288 MiB then rolled back to 48 MiB; token counters did not advance until pressure release; recovered full output matched |
| Graph + hybrid | Same general adapter with actual multiple groups, CoW/state copies, eager and captured paths | TP1/TP2 graph lifecycle and shared-prefix concurrency passed as recorded separate runs; CPU copy/owner contracts pass, but concurrency alone does not prove every native CoW branch executed |
| Matched performance | Native/held/candidate same initial physical budget and traffic; both tenants' full outputs and latency | Initial latency costs measured; matched performance/SLO acceptance remains open |

CPU tests are deliberate layers of evidence: transaction objects, real worker
methods with a CPU page backend, and unmodified Python functions loaded from a
fixed vLLM source tree. They are not a replacement for torch/NPU/graph execution.
Run `tests/test_native_admission_runtime.py` with `KVCACHED_VLLM_SOURCE` set to
the exact vLLM checkout being evaluated. Record source hashes with results.

```bash
python tests/run_native_capacity_cpu.py \
  --vllm-source /path/to/pinned-vllm-source-or-site-packages \
  --out /path/to/new-evidence-directory
python tests/ascend/native_budget_control.py --url http://127.0.0.1:PORT
```

The CPU runner records implementation/source hashes, Git status, individual
results, failures and skipped checks. The HTTP helper requires current rank
observations; it does not turn a pending target into a successful shrink claim.

Remaining parity outside this first closure: upstream CLI/control API integration,
cross-DP admission and liveness, async/connector lifetimes, sleep/weight ownership,
less conservative per-group admission, event fences, and measured optimization.
