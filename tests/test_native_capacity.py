# SPDX-License-Identifier: Apache-2.0
"""Budget semantics without a tensor library or accelerator."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

spec = importlib.util.spec_from_file_location(
    "native_capacity_under_test", Path(__file__).parents[1] / "kvcached/native_capacity.py"
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.storages = {"shared": NS(page_bytes=16), "state": NS(page_bytes=32)}
        self.mapped = {"shared": {0, 16}, "state": {0}}

    def test_counts_physical_union_not_sum_of_logical_group_sizes(self):
        state = module.NativeBudgetState(module.NativeBudget(limit_bytes=48))
        required = {"shared": {0}, "state": {0}}
        self.assertTrue(state.permits(self.storages, self.mapped, required))
        row = state.status(self.storages, self.mapped, required)
        self.assertEqual(
            (row["required_bytes"], row["mapped_bytes"], row["reclaimable_bytes"]), (48, 64, 16)
        )
        self.assertEqual(row["convergence"], "pending")

    def test_lower_limit_preserves_live_decode_but_refuses_new_pages(self):
        state = module.NativeBudgetState(module.NativeBudget(limit_bytes=0))
        self.assertTrue(state.permits(self.storages, self.mapped, self.mapped))
        self.assertFalse(
            state.permits(self.storages, self.mapped, {"shared": {0, 32}, "state": set()})
        )
        empty = {"shared": set(), "state": set()}
        self.assertEqual(state.status(self.storages, empty, empty)["convergence"], "converged")

    def test_revision_is_idempotent_but_conflicts_are_rejected(self):
        state = module.NativeBudgetState()
        new = module.NativeBudget(2, 1024, 4)
        state.apply(new)
        state.apply(new)
        for wrong in (module.NativeBudget(1), module.NativeBudget(2, 2048, 4)):
            with self.assertRaises(ValueError):
                state.apply(wrong)
        self.assertEqual(state.budget, new)

    def test_invalid_limits_fail_before_application(self):
        for kwargs in (
            {"limit_bytes": -1},
            {"limit_bytes": True},
            {"prefix_max_blocks": 1.2},
            {"revision": None},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                module.NativeBudget(**kwargs)

    def test_rank_coverage_and_partial_success(self):
        rows = [{"rank": 0, "ok": True}, {"rank": 1, "ok": False}]
        self.assertFalse(module.all_rank_success(rows, [0, 1]))
        rows[1]["ok"] = True
        self.assertTrue(module.all_rank_success(rows, [0, 1]))
        for malformed in (rows[:1], [rows[0], rows[0]], [{"rank": 0, "ok": 1}], None):
            with self.subTest(rows=malformed), self.assertRaises(RuntimeError):
                module.all_rank_success(malformed, [0, 1])


if __name__ == "__main__":
    unittest.main()
