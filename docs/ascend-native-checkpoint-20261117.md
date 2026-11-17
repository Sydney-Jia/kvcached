# Native capacity closure: A5 hardware checkpoint

This checkpoint extends, rather than replaces, the [2026-11-12 results](ascend-native-checkpoint-20261112.md).
The implementation stays on `feat/ascend-generic-kv`; no model-specific allocator or branch was added.
The new admission path is opt-in and currently supports synchronous, single-DP TP execution.

## Tested release and evidence

- Capacity implementation: `4a9ec7841920d9c09420bc917225716718f0202d`.
- Bounded per-rank rejection history: `25f8266` (same capacity policy).
- Runtime image: `sha256:8d8e8d7ed793be3dbbca16970aef1814882bbe0e69d289303f79bc250adca224`.
  vLLM `0.31.0.dev0+gc4df37df`, torch / torch_npu 2.10, native Ascend V2 runner.
- Successful TP2 runtime: `sha256:f1943b003d03b2182badbdd58513c229b4bdd711d24fd320ac5b59b5e7eb0778`,
  framework wheel `f36508cea7f46b767f1c966447264b8082397a02`, same vLLM core and
  torch versions, `HCCL_OP_EXPANSION_MODE=AIV`, isolated personal Docker store.
  Its private A5 SDK closure is not a claim about a public CANN release.
- Candidate wheel SHA-256: `0c6c7e5a8dbd36eddfa4f238db1f88c61bb1ef91e11debccf15bb55bd5d50c3a`.
- CPU regression: 70 passed, no failures/errors/skips; seven selected native API checks also passed inside the actual image.
- Raw responses, streaming timestamps, Prometheus counters, per-rank RPC results,
  driver memory, launch arguments and fixed-pressure manifests are retained in the
  project evidence directory `outputs/kvcached-native-capacity-20261117`.
  These are controlled experiments, not a general performance benchmark.
  The downloaded archive has 568 SHA-256-verified recorded files. Test services
  and the new personal runtime daemon were stopped; both assigned cards had no
  remaining processes at the final device check.

## What was connected

| Capability | Existing mechanism | Generic native integration | Evidence / scope |
|---|---|---|---|
| Separate logical and physical capacity | CANN address reservation and physical map/unmap; upstream elastic storage | `NativeStorageArena(lazy=True)` reserves byte arenas without backing the whole native pool; validated null/dummy accesses are backed before use | 27B TP1: 8176 MiB reservation, 80 MiB mapped after startup; 4 groups, 1 arena, 320 spans |
| Admit against real capacity | Native `allocate_slots` computes block demand; VMM guard observes driver headroom | `NativeAdmission` prepares candidate slots on workers before native allocation mutates the block pool; clean misses return to native waiting/preemption | Real pressure produced physical shared-budget refusals and later successful inference |
| Return idle pages and grow again | CANN unmap/free, stable virtual addresses | Complete owner union, completion fence, retirement before growth, rollback of newly mapped pages | Two engines with different layouts exchanged real usable capacity; addresses stayed stable |
| Prefix retention / eviction | Native cache hashes, aliases, reference counts and eviction API | Cold-block cap, pressure retry, post-output and cancellation reconciliation | Repeated input added 896 prefix-hit tokens; clearing prefixes returned 224 MiB; replay recomputed the same answer without further hits |
| Runtime budget | New revisioned native scheduler/executor control, not upstream `kvctl` ABI | `kvcached_native_budget`: set/status/reconcile, all-rank observations, pending vs converged | Live shrink, zero-target waiting, target increase and client cancellation passed on TP1 eager and graph |
| Captured execution | Native attention/state views and NPU graphs | Stable addresses, dummy-state backing before capture, fence before retirement | TP1 and TP2: 4 captures per worker; replay counters advanced through reclaim/regrow phases and full outputs matched; TP2 lifecycle reached 410 replays per rank |
| TP partial grants | Synchronous all-rank RPC and native allocation | Drain every reply, validate tickets/revisions, unwind speculative backing before returning a miss; `last_failed_prepare` retains failure separately from current status | CPU checks and live TP2 asymmetric physical rejection / rollback / recovery passed |

Neither attention kernels nor native block tables are replaced. The logical ceiling
is still fixed by vLLM. Physical growth means backing more slots **within that ceiling**.
Discarded cold prefixes must be recomputed; no old content is invented on remapping.

## Real physical borrowing, not a logical-block counter

Donor: local quantized 27B hybrid checkpoint. Receiver: Qwen3-0.6B, ordinary attention,
1 group and 28 raw allocations. Both used the same generic worker/scheduler path.

The test kept **47,791 MiB of additional live tensors constant**. This consumed real
HBM and left about 4709 MiB driver free, near the allocator's 5% protection margin.
The receiver's instance limit stayed at 8 GiB; this was not a synthetic low-quota test.

| Observation | Donor retains cold prefixes | Donor returns cold prefixes |
|---|---:|---:|
| Donor mapped KV | 4144 MiB | 80 MiB |
| Receiver mapped KV | 56 MiB | 2968 MiB at a post-release sample |
| Four 24,532-token requests | All incomplete after five seconds; capacity refusals recorded | All completed, exact full outputs match no-pressure references |
| Donor inference | Same live engine | Subsequent arithmetic request returned `462` |

The ballast remained live and unchanged through completion. Driver observations and
backend refusal messages were saved independently of logical KV usage. Both engines
returned to their idle floors afterward. The 5% protection margin was kept: this
demonstrates recovery from real usable-capacity exhaustion, not forced driver OOM.

This is a within-run held-prefix/release comparison, **not** an equal-throughput
three-implementation benchmark. The no-pressure reference used the same receiver
candidate; 27B correctness separately used a native engine reference.

## Correctness and measured costs

- Four 27B inputs matched the native engine, its repeat, candidate repeat and
  post-eviction recomputation. Graph initial/evicted/restored phases also matched.
- A 160-token generation matched its reference after lowering the budget to zero
  and restoring it. Zero can leave a mandatory null floor and report `pending`.
  Native preemption/recomputation may occur; this is not pause-free shrink.
- Client cancellation converged without another request; subsequent requests worked.
- Four simultaneous shared-prefix requests matched their sequential references
  on the captured TP1 path. During a separate live 160-token generation, changing
  the target from 8 GiB to 2 GiB and the cold-prefix cap to zero reduced mapped KV
  from 560 MiB at the pre-control observation to 336 MiB at acknowledgement,
  then 80 MiB on completion. No admission denial occurred and full output matched.
  This is a below-target working-set case, distinct from the zero-target test;
  it does not establish an interruption-free guarantee for arbitrary shrink.
- Allocation/free is not free. With warm requests, candidate eager latency was
  visibly above native. Across approximately 400 steps, cumulative mapped/unmapped
  bytes were much larger than the live working set. This indicates substantial
  churn; per-group preparation and bounded spare-page retention need measurement.
- Receiver no-pressure completion latencies: 15.39, 7.93, 11.64, 16.71 seconds.
  Under held-then-released physical pressure: 18.62, 28.39, 38.74, 42.32 seconds.
  These include intentional waiting, prefill and contention; no throughput/SLO gain
  is claimed. Repeated matched runs are required before comparing performance.

## TP2 asymmetric physical pressure

Both ranks started with 48 MiB mapped inside an 8176 MiB logical reservation.
On the successful TP2 eager runtime, the initial, evicted and restored phases
matched all four native TP1 reference outputs. Four shared-prefix concurrent
requests matched their serial references; a live 160-token target reduction
also matched. These are selected deterministic output checks, not an accuracy suite.

Only rank 0's card then received 70,271 MiB of fixed live pressure tensors.
Both engine limits remained 2 GiB, above the requested working set.

| Observation | Rank 0, physically constrained | Rank 1, unconstrained |
|---|---|---|
| Same prepare ticket | Physical shared-capacity rejection; 240 MiB new backing requested | Grant succeeded; mapped KV reached 288 MiB |
| After collective rollback | 48 MiB mapped, no poisoned state | 48 MiB mapped, no poisoned state |
| Request progress while blocked | Native capacity waiting queue: one request | No partially granted inference |
| After releasing pressure | Request resumed and full output matched its no-pressure reference | Participated normally; both ranks returned to 48 MiB idle |

The retained failed ticket and the later successful rollback observation are
separate records. During the blocked interval, prompt tokens stayed at 20,248
and generated tokens at 486; neither advanced. Worker call/step counters did
advance for empty scheduling iterations, so those counters are **not** token
progress. The recorded controller had 21 admission denials. No forced driver OOM
or synthetic low instance quota was needed.

The separate TP2 graph run captured four graphs per worker. All four outputs in
each initial/evicted/restored phase matched the native reference. Live zero-target
waiting/recovery preserved the complete 160-token output; a fresh blocked request
resumed with `462`, and cancellation cleanup was followed by `READY`. Both ranks
reached 410 graph replays and returned to 48 MiB after cancellation. This covers
TP2 graph lifecycle and TP2 eager asymmetric physical pressure as separate tests;
their combination under graph-mode asymmetric physical pressure was not run.

Runtime boundary: the older image failed in native profiling **before elastic
cache initialization**, with `RunAicpuIndOpThreadInit` / `libccl_kernel.so` AICPU
page faults. The unadapted native comparison failed too. Eager and graph both
failed; small AIV all-reduce passed, which was insufficient to establish model
communication support. The newer pinned image and isolated runtime enabled the
TP2 tests without changing the capacity implementation. This does not isolate
the single underlying library/configuration cause; failed launches are retained.

## Boundaries and next priorities

1. Reduce per-step over-preparation, full fences and map/unmap churn, then repeat
   matched physical-pressure and latency tests. Keep correctness and rollback checks.
2. New capacity mode still lacks cross-DP admission, asynchronous queued execution,
   PP/CP, connector completion ownership, sleep/weight ownership and upstream CLI ABI.
3. DeepSeek's older 13-group acceptance is for the prior native-residency mode;
   it is not a new-capacity result. No fresh DeepSeek capacity claim is made here.
4. Hardware startup temporary caches, weights and graph workspace remain outside
   the serving KV budget. The recorded 528 MiB cumulative startup KV backing is
   not a whole-device peak-memory measurement.

Operational correction: on this upgraded host, monitor card IDs and runtime physical
IDs differ. Launchers now read `npu-smi info -m` and record the mapping. The initial
native TP1 baseline ran on an idle monitor card 4 and was stopped; the candidate
ran on monitor card 5. Subsequent tests were confined to assigned monitor cards 5/6.
No other user's process, container or host-wide device setting was changed.
