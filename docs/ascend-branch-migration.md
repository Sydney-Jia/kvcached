# Legacy branch migration audit — 2026-11-12

The active integration line is `feat/ascend-generic-kv`; `main` tracks the upstream
baseline. Archive tags preserve exact historical source, tests and failure records.
This audit covers all 15 DeepSeek-prototype commits after `357ed3d` and all 13
burst commits after `1638260`. VMM (`357ed3d`) and early hybrid (`abb4112`) are
already ancestors of the generic branch. No model-specific worker is merged back.

## DeepSeek prototype: 16 changed files

| Historical commits / assets | Current disposition and supporting code/tests |
|---|---|
| `948e00a`: geometry, native pool, worker; `test_ascend_native_geometry.py`, `test_ascend_native_pool.py` | Replaced by `residency.py`, `physical_residency.py`, and the native storage/snapshot/worker/scheduler adapter. `test_residency_contract.py` covers view ranges, aliases and page unions; `test_physical_residency.py` covers completed retirement, rollback, new-page initialization and poisoned state. `test_native_residency_snapshot.py` covers all groups, unscheduled requests, prefix aliases, pins, watchers, copy/zero and null slots. Added explicit replacement/preemption regression during this migration. The old single-slot/delta-manager assumptions are intentionally not restored. |
| `c0001e2`, `5a8368d`: all-DP control evidence, completed transition time | `ascend_generic_worker.py` emits rank/DP/device identity, ownership and completed control timing. Existing checkpoint includes all-rank evidence. The HTTP frontend's first-engine-only response limitation remains explicit. |
| `4dd921e`, `dfd8d33`, `9f74831`: serialized stream diagnostic and co-resident launch option | Archive only: `ascend_serial_stream_worker.py` and `tools/ascend_native_launch.py` are model/runtime-specific diagnostics, not demonstrated general fixes. Preserve failure records; do not insert startup serialization into the core. Launch configuration stays outside allocator policy. |
| `940829b`, `81196c2`, `86bdea3`, `4049ad0`: optional `_npu_storage` bridge, build wiring, storage tests | Mainline `csrc/ascend_tensor.cpp` creates native ND storage through the tensor factory; `native_storage.py` retains native views. No second extension or private-symbol bridge is needed. Added `test_native_nd_fp8_views_survive_partial_retirement` to `test_ascend_vmm.py`: native ND/FP8 metadata, alias lifetime after dropping the original tensor, live-page preservation, remap and stable addresses. |
| `d64facb`, `4f42b97`: physical device identity and visibility mapping | Already reimplemented in `csrc/inc/ascend_vmm.hpp`; `test_physical_keys_are_distinct_and_visibility_stable` remains in `test_ascend_vmm.py`. Single-device and shared UUID identity must not be conflated. |
| `f31c747`: separate torch allocator trimming | Explicit `trim_allocator` control and bounded capacity-relief path are already present. `test_capacity_relief.py` verifies capacity-only retry and fatal-error handling. Trim remains distinct from KV page reclaim. |
| `3b3c37d`, `1638260`: acceptance, failure boundaries and units | Historical `docs/ascend-native-hybrid.md` stays at `archive/ascend-dsv41-hybrid-20261030`. Current acceptance is `ascend-native-checkpoint-20261112.md`; old results are not relabeled as new-runtime evidence. |

Build-only changes to `MANIFEST.in`/`setup.py`, `csrc_ascend/native_storage.cpp`, and
old test-manifest entries belong to the superseded optional bridge. Restoring them
would create a second storage path. The new regression is classified in the
existing hardware module; generic contract tests remain CPU-only.

## Burst experiment: six changed files

| Historical commits / assets | Current disposition |
|---|---|
| `cda92e9`: traffic harness | Ported open-loop arrivals, full SSE output and per-service/phase measurements into `tests/ascend/service_burst.py`; services and cases are supplied by configuration. No model launch or old quota dependency. |
| `6c2d0a2`, `adabd57`, `930872c`, `c144d9e`: calibration directories, startup memory fraction, tokenizer normalization, initial headroom | Calibration/deployment/pressure remain external and preserved at the archive tag. Fresh output directories prevent overwrite. The new HTTP payload passes through without the old tokenizer dependency; inputs and physical headroom must be matched and recorded explicitly. |
| `bbe241a`, `e41bee0`: distinct output oracles and equal-width counters | Configurable multiple cases with full exact-text or numeric-sequence oracles. Fixture uses equal-width counting ranges; explicit reference mode checks full content/reasoning/token IDs/finish reason. Prompts still require native-repeat validation. |
| `c300360`, `c436c03`: quota versus physical reclaim, light versus intrusive observers | Native residency controls replace quota mutation. HTTP metrics remain light; worker control timing and physical measurements are separate. `service_burst_probe.py` is archived rather than restoring the old quota worker subclass. |
| `3515029`: retry transient quota observations; fail incomplete traces | Old quota-convergence retry is inapplicable to the new synchronous mutation result. New controls reject failed/poisoned/incomplete responses without retry; missing/truncated requests fail the run, and raw evidence survives. |
| `0a77b2e`, `077279d`: protocol, tests and manifest | Rewritten `docs/ascend-service-burst.md`, manual-test index and classified `tests/test_service_burst_control.py`. Tests exercise real local HTTP streaming/control, paired workload fingerprints, strict response oracles, missing IDs, invalid attribution and abort evidence. |
| `0fb3930`: measured outcomes and unresolved regressions | Original document remains under `archive/ascend-service-burst-20261105`; no historical throughput/SLO claim is transferred to generic. |

## Retained history and scope

| Archive tag | Commit |
|---|---|
| `archive/ascend-vmm-20261029` | `357ed3dcbe2c57ce5afbbd0d87de6228136ee505` |
| `archive/ascend-hybrid-kv-20261030` | `abb4112fee99558c75d54d36e89acd40f12bed1d` |
| `archive/ascend-dsv41-hybrid-20261030` | `16382609aa37688259ca019031d7c1249e952eca` |
| `archive/ascend-service-burst-20261105` | `0fb39307d3199c740a2f4c37f9303a8acd696895` |

Use `git rev-parse <tag>^{commit}` to inspect an archived source and
`git worktree add --detach <new-directory> <tag>` to reproduce it without making
another permanent branch. Raw local/remote experiment directories are retained.
CaMem and the decoder-graph framework repositories are separate experiments and
are not deleted or folded into kvcached by this consolidation.

This migration changes tests, measurement tools and documentation only. Runtime
package and C++ sources remain identical to the `decad6a` model-acceptance build.
It does not establish production SLOs, automatic multi-rank capacity admission,
or support for unimplemented async/transfer lifetime contracts.
