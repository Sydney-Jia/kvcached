# SPDX-License-Identifier: Apache-2.0
"""Run the hardware-free native capacity regression and preserve its evidence."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.vllm_source:
        os.environ["KVCACHED_VLLM_SOURCE"] = str(args.vllm_source.resolve())
    args.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root / "tests"))
    modules = [
        "test_residency_contract",
        "test_physical_residency",
        "test_capacity_relief",
        "test_native_residency_snapshot",
        "test_native_capacity",
        "test_native_admission",
        "test_native_admission_runtime",
        "test_native_capacity_worker",
        "test_native_capacity_executor",
        "test_native_lazy_storage",
        "test_native_budget_control",
    ]
    suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
    with (args.out / "unittest.log").open("w", encoding="utf-8") as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    source_hashes = {}
    if args.vllm_source:
        for name in ("block_pool", "kv_cache_manager", "kv_cache_utils"):
            path = args.vllm_source / f"vllm/v1/core/{name}.py"
            source_hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    implementation = [
        "kvcached/native_capacity.py",
        "kvcached/physical_residency.py",
        "kvcached/residency.py",
        "kvcached/capacity_relief.py",
    ]
    implementation += [
        "kvcached/integration/vllm/" + name + ".py"
        for name in (
            "native_admission",
            "native_executor",
            "native_scheduler",
            "native_storage",
            "native_residency",
            "ascend_generic_worker",
        )
    ]
    summary = dict(
        scope="CPU contracts, real worker methods with CPU storage, extracted native Python functions; NOT NPU/model acceptance",
        tests=result.testsRun,
        failures=len(result.failures),
        errors=len(result.errors),
        skipped=[(str(test), why) for test, why in result.skipped],
        passed=result.wasSuccessful(),
        base_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        working_tree_status=subprocess.check_output(
            ["git", "status", "--short"], cwd=root, text=True
        ),
        implementation_sha256={
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in implementation
        },
        native_python_source_sha256=source_hashes,
    )
    (args.out / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: summary[key] for key in ("tests", "failures", "errors", "skipped", "passed")}
        )
    )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
