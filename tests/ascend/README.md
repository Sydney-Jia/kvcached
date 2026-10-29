# Ascend probe guide

These scripts are manual acceptance/diagnostic probes unless explicitly listed
as CPU checks below. They do not reserve an NPU. Use the
[installation contract](../../docs/ascend.md) and
[current acceptance status](../../docs/ascend-status.md) before selecting a run.

## Select the boundary to verify

| Boundary | Entry points | What the result can establish |
|---|---|---|
| VMM and independent memory observations | `physical_probe.py`, `driver_memory.py` | Physical return, address/data invariants and independent allocator observations |
| Basic model and prefix behavior | `single.py`, `dual.py`, `prefix.py`, `worker_probe.py` | Output equivalence and bounded block/prefix lifecycle behavior |
| Installed service | `installed_serve.py` | CLI/HTTP activation, streaming, cancellation, runtime limits, restart and worker/IPC cleanup |
| Capacity pressure | `three_arm.py`, `pressure_worker.py`, `pressure_protocol.py`, `pressure_stream.py` | Native / held / reclaim comparison with complete outputs and explicit latency targets |
| Natural elasticity and text workloads | `natural_elasticity.py`, `realistic_workloads.py` | Request-driven growth/return without publishing a quota; native output comparison |
| Resize timing | `resize_latency.py`, optional `vmm_timing_preload.cpp` | Control acceptance, observed convergence, physical operations and output gaps as distinct measurements |
| DeepSeek/Qwen co-location | `colocation_driver.py`, `colocation_service.py`, `colocation_http.py`, `colocation_deepseek_profile.json` | Large static service plus a single-card elastic service; not DeepSeek elasticity |
| FIA diagnosis | `fia_reference.py`, `fia_trace.py`, `fia_compatibility.py` | A separate runtime/kernel compatibility check; not a substitute for memory tests |
| Process ownership | `process_identity.py` | Helpers for bounded supervision; not permission to stop unrelated work |

The co-location driver takes external machine bindings using `--config`.
Its profile is an experiment candidate, not a generic deployment recommendation.
Freeze source and configuration for each hardware run; preserve the raw command,
inputs, full outputs, device samples, logs and exit results together.

## CPU-only checks

The following nested modules are independent targeted checks, not a claim that
all hardware probes run in CPU CI:

- `test_colocation_http.py`: stream completion, expected answer and output-length
  contracts. It can run without an accelerator:

  ```bash
  python tests/ascend/test_colocation_http.py
  ```

- `test_ballast_transaction.py`: allocation-pressure helper cleanup behavior.
- `test_fia_compatibility.py`: explicit FIA argument compatibility behavior.

Top-level `tests/test_*.py` files use the exclusive classification described in
[`tests/manifests/README.md`](../manifests/README.md). Check that classification
with `python tools/check_test_classification.py`; do not run all manual NPU probes
as if they were unit tests.

## Evidence rules

1. Verify current device health and process ownership before starting. Invalidate
   comparisons contaminated by foreign device work; stop only the experiment's
   own tasks. A visible device is not a reservation.
2. Keep checkpoint, dtype, attention options, inputs and cache history consistent
   between references and elastic runs. An FIA workaround must match both arms.
3. Record both the allocator ledger and independent HBM readings. A smaller
   logical limit, a returned control call or lower PyTorch reserved memory alone
   does not prove physical reclamation.
4. Compare complete outputs. Separately report structured-answer correctness,
   token equivalence, streaming completion and latency-target attainment.
5. Check actual worker exit, IPC cleanup and control membership after shutdown.
   A frontend exit alone is insufficient; one incomplete observer snapshot is
   not proof of termination or memory return.
