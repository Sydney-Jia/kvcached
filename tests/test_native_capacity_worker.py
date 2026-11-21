# SPDX-License-Identifier: Apache-2.0
"""Exercise actual worker methods with CPU page storage and a fake NPU base.

Only external device/base-runner operations are replaced. This does not prove
device stream, graph, or model behavior; those require the hardware matrix.
"""

import ast
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

from test_physical_residency import Memory, module as physical
from test_native_admission import load, capacity

ROOT = Path(__file__).parents[1]
residency = load("worker_test_residency", "kvcached/residency.py")
ownership = sys.modules["kvcached.integration.vllm.native_residency"]
relief = load("worker_test_relief", "kvcached/capacity_relief.py")


class BaseWorker:
    def execute_model(self, output):
        self.executions += 1
        return "executed"


namespace = dict(
    NPUWorker=BaseWorker,
    NativeBudget=capacity.NativeBudget,
    NativeBudgetState=capacity.NativeBudgetState,
    is_capacity_rejection=relief.is_capacity_rejection,
    os=os,
    native_gate=NS(installed=lambda: False),
)
source = ROOT / "kvcached/integration/vllm/ascend_generic_worker.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
klass = next(node for node in tree.body if isinstance(node, ast.ClassDef))
exec(compile(ast.Module(body=[klass], type_ignores=[]), str(source), "exec"), namespace)
Worker = namespace["NativeElasticWorker"]


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.memory = Memory()
        layout = residency.Layout(
            [residency.Storage("0", 128, 16)],
            [residency.BlockSpan("0", 0, 0, 16, 8, 8), residency.BlockSpan("0", 1, 8, 16, 8, 8)],
        )
        pages = physical.PhysicalResidency(
            self.memory, self.memory.synchronize, self.memory.initialize
        )
        pages.register(layout.storages["0"], 0)
        pages.ensure({"0": {0}})
        worker = Worker()
        # The arena contract carries spare/keep policy arguments; the CPU fake
        # only needs the physical page transition behind them.
        worker.native_arena = NS(
            residency=pages, ensure=pages.ensure, retire_async=False,
            transition=lambda required, **_: pages.transition(required),
            grow=lambda required, **_: pages.ensure(required),
            defer_retire=lambda required: None,
        )
        worker.native_owners = residency.ResidencyOwners(layout)
        worker.native_owners.replace({"null": ((0, 0), (1, 0))})
        worker.native_capacity_enabled = True
        worker.native_budget = capacity.NativeBudgetState()
        worker.native_capacity_ticket = 0
        worker.native_epoch = 0
        worker.native_elastic = True
        worker.native_mapped_revision = -1
        worker.native_steps = 0
        worker.native_trace_residency = False
        worker.executions = 0
        worker.rank = 0
        self.worker = worker

    def snapshot(self, requests=(), epoch=1, copies=()):
        return ownership.NativeOwnership(
            1,
            epoch,
            8,
            2,
            {
                "null": ((0, 0), (1, 0)),
                "requests": tuple(requests),
                "prefix": (),
                "native-pins": (),
                "native-watchers": (),
                "step-io": tuple(copies),
            },
        )

    def prepare(self, candidates, budget=None, snapshot=None):
        return self.worker.kvcached_native_prepare(
            snapshot or self.snapshot(),
            candidates,
            (budget or capacity.NativeBudget()).as_dict(),
            self.worker.native_capacity_ticket + 1,
        )

    def execute(self, snapshot):
        return self.worker._execute_with_residency(
            NS(total_num_scheduled_tokens=1, kvcached_ownership=snapshot)
        )

    def test_prepared_slots_execute_and_addresses_keep_existing_contents(self):
        self.assertTrue(self.prepare((1,))["ok"])
        self.memory.data[0][16] = "live KV"
        self.assertEqual(self.execute(self.snapshot(((1, 1),))), "executed")
        self.assertEqual(self.memory.data[0][16], "live KV")
        self.assertEqual(self.worker.native_steps, 1)

    def test_transition_is_synchronous_and_pressure_gates_execution(self):
        calls = []
        pages = self.worker.native_arena
        self.worker.native_spare_bytes = 0

        def transition(required, **kwargs):
            calls.append("transition")
            return dict(reclaimed_bytes=0, grown_bytes=0)

        def state(charged):
            return NS(physical_budget_state=lambda: dict(
                budget_bytes=1000, charged_bytes=charged))

        self.worker.native_arena = NS(
            residency=pages.residency, operations=state(100), transition=transition)
        self.worker._transition_native({"0": {0}}, self.snapshot())
        self.assertEqual(calls, ["transition"])
        self.assertFalse(self.worker._budget_pressure())
        self.worker.native_arena.operations = state(900)
        self.assertTrue(self.worker._budget_pressure())

    def test_unprepared_slot_never_launches_native_inference(self):
        with self.assertRaisesRegex(RuntimeError, "absent from native admission"):
            self.execute(self.snapshot(((0, 7),)))
        self.assertEqual(self.worker.executions, 0)
        self.assertEqual(self.worker.native_epoch, 0)
        self.assertEqual(self.memory.data[0], {0: 0})

    def test_budget_rejection_returns_result_and_abort_preserves_live_page(self):
        self.prepare((1,))
        self.memory.data[0][16] = "live"
        self.execute(self.snapshot(((0, 1),)))
        budget = capacity.NativeBudget(1, 32, 0)
        snapshot = self.snapshot(((0, 1),), epoch=2)
        rejected = self.prepare((2,), budget, snapshot)
        self.assertFalse(rejected["ok"])
        self.assertEqual(rejected["rejection"], "instance_limit")
        self.assertIsNone(rejected["fatal"])
        self.assertTrue(self.prepare((), budget, snapshot)["ok"])
        self.assertEqual(self.memory.data[0], {0: 0, 16: "live"})
        self.assertEqual(self.worker.native_steps, 1)

    def test_lower_budget_keeps_live_decode_pending_until_retirement(self):
        self.prepare((1,))
        budget = capacity.NativeBudget(1, 16, None)
        row = self.prepare((), budget, self.snapshot(((0, 1),)))
        self.assertTrue(row["ok"])
        self.assertEqual(row["budget"]["convergence"], "pending")
        self.execute(self.snapshot(((0, 1),)))
        retired = self.prepare((), budget, self.snapshot(epoch=2))
        self.assertTrue(retired["ok"])
        self.assertEqual(retired["budget"]["convergence"], "converged")
        self.assertEqual(self.memory.data[0], {0: 0})

    def test_backend_pressure_is_a_returned_capacity_miss(self):
        original = self.memory.map_to_kv_tensors_with_result

        def bounded(pages, pool):
            if len(self.memory.data[pool]) + len(pages) > 2:
                raise RuntimeError("capacity_exhausted: peer holds remaining pages")
            return original(pages, pool)

        self.memory.map_to_kv_tensors_with_result = bounded
        row = self.prepare((1, 2))
        self.assertFalse(row["ok"])
        self.assertIsNone(row["fatal"])
        self.assertEqual(self.worker.executions, 0)
        self.assertTrue(self.prepare(())["ok"])
        self.assertEqual(self.memory.data[0], {0: 0})

    def test_other_rank_can_receive_fatal_result_before_engine_stops(self):
        self.memory.fail_map = 0
        row = self.prepare((1,))
        self.assertFalse(row["ok"])
        self.assertIn("MemoryError", row["fatal"])
        self.assertEqual(self.worker.executions, 0)

    def test_legacy_elastic_path_also_returns_old_capacity_before_growth(self):
        self.worker.native_capacity_enabled = False
        self.worker.native_arena.ensure({"0": {0, 16}})
        original = self.memory.map_to_kv_tensors_with_result

        def bounded(pages, pool):
            if len(self.memory.data[pool]) + len(pages) > 2:
                raise RuntimeError("capacity_exhausted: two-page cap")
            return original(pages, pool)

        self.memory.map_to_kv_tensors_with_result = bounded
        self.execute(self.snapshot(((0, 2),)))
        self.assertEqual(self.memory.data[0], {0: 0, 32: 0})
        self.assertEqual(self.worker.executions, 1)


if __name__ == "__main__":
    unittest.main()
