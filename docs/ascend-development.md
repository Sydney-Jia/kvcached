# Ascend development line and experiments

Updated: 2026-11-12. The single Ascend development line in this fork is
`feat/ascend-generic-kv`. The fork's `main` remains the upstream baseline;
its name does not make it the Ascend deployment source. Use fixed commits for
releases and retain the [acceptance checkpoint](ascend-native-checkpoint-20261112.md).

## One mechanism, explicit contracts

Do not maintain a permanent kvcached branch for each model. A new model that
fits existing contracts adds configuration and acceptance coverage. A new state
or execution semantic may need a new adapter contract, implemented on this
development line after review and validation.

| Difference | Where it belongs |
|---|---|
| Weights, model size, layer count, head geometry, parallelism | Native framework configuration and test fixtures |
| Storage shapes, strides, offsets, aliases, cache groups | Native layout descriptors and the storage adapter |
| Attention blocks, recurrent state, prefix retention, copy/zero or dummy use | Typed ownership/lifetime contracts, not model-name branches |
| CUDA versus Ascend mapping and completion primitives | Backend implementations of the same physical-storage interface |
| vLLM versus another engine, or incompatible engine interfaces | Framework/version adapters with explicit capability checks |
| Eager, captured graphs, asynchronous execution, transfers | Execution-completion contracts and corresponding validation |
| Different reclaim policies | A policy interface/configuration; an experiment branch until the policy is validated |

The core does not need the model name. It does need complete information about
storage, owners and when device accesses finish. A missing contract must fail
explicitly or be implemented; generic design is not evidence that every model
already works. Spec classes alone do not describe all asynchronous lifetimes.

Model-specific kernels and graph wrappers remain in the framework repository.
They may be necessary for the framework to run a model, but should not create a
second kvcached physical allocator or residency implementation for that model.
Legacy entry points still exist in this branch's history/tree; new capabilities
target the explicit native adapter. Their historical tests are not automatically
acceptance for the new adapter, and they are not parallel active product lines.

## Existing branches

| Branch | Role from this checkpoint | Preservation / integration status |
|---|---|---|
| `main` | Upstream baseline | Retained; not the current Ascend build |
| `feat/ascend-generic-kv` | Only active Ascend integration line | Implement, test and document new shared capabilities here |
| `feat/ascend-vmm` | Frozen first implementation | Entire head is an ancestor of generic; archive tag `archive/ascend-vmm-20261029` at `357ed3d` |
| `experiment/ascend-hybrid-kv` | Frozen early hybrid prototype | Entire head is an ancestor of generic; archive tag `archive/ascend-hybrid-kv-20261030` at `abb4112` |
| `feat/ascend-dsv41-hybrid` | Frozen native-layout prototype | Not merged as a branch; selected fixes/contracts reimplemented. Tag `archive/ascend-dsv41-hybrid-20261030` at `1638260`; keep old tests and failure records |
| `experiment/ascend-service-burst` | Frozen workload experiment and migration source | Extends the DeepSeek prototype; not merged into generic. Tag `archive/ascend-service-burst-20261105` at `0fb3930` |

These are lifecycle designations, not GitHub branch protections. Branch pointers
and worktrees remain during this consolidation; archive tags preserve their
exact commits. No branch deletion or deployed-source transition is implied.
Old branches receive no new capabilities. A necessary historical fix must be
explicitly scoped, with an equivalent regression evaluated on the current line.

## What remains to transfer

The native DeepSeek prototype contributed physical device identity, native ND
storage requirements, stable backing, page unions, all-rank measurements and
separation of torch cache trimming from KV reclaim. The generic implementation
now owns these responsibilities; do not merge its model-specific worker on top.
Audit remaining old tests for reusable invariants and keep model/runtime-specific
launch recipes with their original evidence.

The burst branch adds six files relative to its DeepSeek parent: a traffic trace
harness, measurement worker, control tests, documentation and test-index entries.
Its open-loop arrivals, distinct-output correctness checks, queue/latency metrics,
light versus intrusive observations and failure records are reusable. Its old
quota and worker APIs are not the native residency contract. Port the harness
through a measurement/control adapter and rerun it; do not carry its old SLO
results forward as results for generic.

## Bounded research candidates

These are a prioritized backlog, **not pre-created branches**. Open one short-lived
branch from the current development line when work begins. Record its hypothesis,
base commit, acceptance checks and decision to integrate or archive.

| Priority | Research question | Acceptance needed before integration |
|---|---|---|
| 1 | Can physical-capacity admission coordinate all affected ranks before execution? | Insufficient capacity waits/rejects cleanly, cancellation releases ownership, asymmetric rank failures do not deadlock or partially advance execution |
| 2 | Can residency reserves / coalesced operations / narrower fences reduce reclaim overhead? | Same correctness and physical borrowing checks, with measured control latency, token gaps, throughput and capacity tradeoffs |
| 3 | Can queued execution and transfers publish complete owner lifetimes? | Async/graph/connector completion coverage, overlapping execution and cancellation tests; no silent fallback |
| Validation | Can the burst harness evaluate the new native adapter? | Native/held/elastic matched conditions, both tenants' output and latency, fixed traffic and pressure, all offered requests counted |

Correctness defects, including investigation of the earlier MegaMoE timeout,
are normal fixes on this development line, not permanent model-specific research
branches. Short-lived fix branches are appropriate. Release maintenance branches
are justified only by a concrete supported release/runtime commitment.

CaMem is an alternative allocator/backend experiment in **vLLM-Ascend**, not
another kvcached development line. Compare it using the same storage/lifetime requirements
and matched workloads; its older test results do not determine performance on
the new runtime. The decoder-graph overlay is also framework-side integration,
not another kvcached product line.
