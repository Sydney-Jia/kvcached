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

## Active branches and archived history

Only `main` and `feat/ascend-generic-kv` remain as branches locally, in the GitHub
fork and in the server Git exchange. Four old branch pointers were removed after
migration, regression and archive-tag verification:

| Former branch | Preserved tag | Disposition |
|---|---|---|
| `feat/ascend-vmm` | `archive/ascend-vmm-20261029`, `357ed3d` | Entire head already inherited |
| `experiment/ascend-hybrid-kv` | `archive/ascend-hybrid-kv-20261030`, `abb4112` | Entire head already inherited |
| `feat/ascend-dsv41-hybrid` | `archive/ascend-dsv41-hybrid-20261030`, `1638260` | Generic contracts replace old worker; missing invariant tests migrated |
| `experiment/ascend-service-burst` | `archive/ascend-service-burst-20261105`, `0fb3930` | Measurement protocol ported; old launcher and results archived |

The two redundant linked worktrees were removed after checking tracked/untracked
files and backing up ignored cache files. The primary repository stays on `main`;
the generic linked worktree is the development entry. Archive tags and raw local/
remote experiment directories remain. Deployed sources/services are unchanged.

## Completed migration

The [28-commit audit](ascend-branch-migration.md) records every unique DeepSeek/burst
commit and the disposition of its files. Physical device identity, native ND
storage, page unions, all-rank measurements and separate torch cache trimming
already had generic equivalents. The remaining portable assets are now migrated:

- ND/FP8 hardware regression: native metadata, alias lifetime, stable addresses
  and live-page preservation through partial retirement and remapping.
- Ownership regression: block replacement/preemption while preserving unscheduled
  requests and retained prefixes.
- [Configuration-driven burst protocol](ascend-service-burst.md): offered arrivals,
  strict full-response checks, paired references, latency/queue observations,
  explicit held/elastic control, failure evidence and CPU HTTP regression tests.

At test commit `787a7e2`, Linux targeted regression passed 41 tests; single-NPU
storage regression passed 5 tests (2 unrelated tests deselected). Runtime sources
are identical to the `decad6a` wheel used in those NPU tests and prior model
acceptance. Protocol tests do not establish new serving-performance/SLO results.

## Bounded research candidates

These are a prioritized backlog, **not pre-created branches**. Open one short-lived
branch from the current development line when work begins. Record its hypothesis,
base commit, acceptance checks and decision to integrate or archive.

| Priority | Research question | Acceptance needed before integration |
|---|---|---|
| 1 | Can physical-capacity admission coordinate all affected ranks before execution? | Insufficient capacity waits/rejects cleanly, cancellation releases ownership, asymmetric rank failures do not deadlock or partially advance execution |
| 2 | Can residency reserves / coalesced operations / narrower fences reduce reclaim overhead? | Same correctness and physical borrowing checks, with measured control latency, token gaps, throughput and capacity tradeoffs |
| 3 | Can queued execution and transfers publish complete owner lifetimes? | Async/graph/connector completion coverage, overlapping execution and cancellation tests; no silent fallback |
| Validation | Can the migrated burst tool demonstrate benefits under matched serving load? | Native/held/elastic matched conditions, both tenants' output and latency, fixed traffic and pressure, all offered requests counted |

Correctness defects, including investigation of the earlier MegaMoE timeout,
are normal fixes on this development line, not permanent model-specific research
branches. Short-lived fix branches are appropriate. Release maintenance branches
are justified only by a concrete supported release/runtime commitment.

CaMem is an alternative allocator/backend experiment in **vLLM-Ascend**, not
another kvcached development line. Compare it using the same storage/lifetime requirements
and matched workloads; its older test results do not determine performance on
the new runtime. The decoder-graph overlay is also framework-side integration,
not another kvcached product line.
