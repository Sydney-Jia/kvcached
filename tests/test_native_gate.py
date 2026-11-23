# SPDX-License-Identifier: Apache-2.0
"""Gate registration and scheduling decisions without vLLM or a device.

These pin three failure modes found in review of the elastic gate: a gate
registered before the owner table exists, cold prefixes evicted before any rank
reports pressure, and a busy heartbeat that stops probing once a sample came
back low.
"""

import ast
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import time
import unittest

ROOT = Path(__file__).parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


gate = load("native_gate_under_test", "kvcached/integration/vllm/native_gate.py")


class StubScheduler:
    def schedule(self, output):
        return output


namespace = dict(
    Scheduler=StubScheduler,
    get_group_id=lambda *args, **kwargs: 0,
    native_gate=gate,
    snapshot_ownership=lambda *args, **kwargs: None,
    os=os,
    time=time,
)
source = ROOT / "kvcached/integration/vllm/native_scheduler.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
klass = next(node for node in tree.body
             if isinstance(node, ast.ClassDef)
             and node.name == "NativeResidencyScheduler")
exec(compile(ast.Module(body=[klass], type_ignores=[]), str(source), "exec"),
     namespace)
Scheduler = namespace["NativeResidencyScheduler"]


class FakeOwners:
    def __init__(self, pages):
        self.pages = pages

    def required_pages(self):
        return self.pages


class FakeArena:
    def __init__(self, budget=100, charged=100):
        self.budget, self.charged = budget, charged
        self.transitions = []
        self.operations = NS(
            physical_budget_state=lambda: dict(
                budget_bytes=self.budget, charged_bytes=self.charged))

    def transition(self, required):
        self.transitions.append(required)
        return dict(reclaimed_bytes=48, grown_bytes=0)


class Rpc:
    """Records actions; the same pressure answer for every rank."""

    def __init__(self, pressure, reclaimed=0):
        self.pressure, self.reclaimed = pressure, reclaimed
        self.actions = []

    def __call__(self, method, kwargs=None):
        action = (kwargs or {}).get('action')
        self.actions.append(action)
        pressure = self.pressure() if callable(self.pressure) else self.pressure
        return [dict(rank=rank, action=action, ok=True, pressure=pressure,
                     reclaimed=self.reclaimed if pressure else 0)
                for rank in range(2)]


class RegistrationTests(unittest.TestCase):
    def tearDown(self):
        gate.clear()

    def test_install_rejects_a_missing_owner_table(self):
        # The worker used to register before reshape published owners; a None
        # captured here made every idle reclaim return 0 while reporting ok.
        with self.assertRaises(ValueError):
            gate.install(FakeArena(), None)
        self.assertFalse(gate.installed())

    def test_reclaim_uses_the_installed_owner_table(self):
        arena = FakeArena()
        gate.install(arena, FakeOwners({"0": {0}}))
        self.assertTrue(gate.installed())
        self.assertEqual(gate.reclaim_idle(), 48)
        self.assertEqual(arena.transitions, [{"0": {0}}])

    def test_worker_registers_after_the_owner_table_exists(self):
        worker = ROOT / "kvcached/integration/vllm/ascend_generic_worker.py"
        module = ast.parse(worker.read_text(encoding="utf-8"))
        function = next(node for node in ast.walk(module)
                        if isinstance(node, ast.FunctionDef)
                        and node.name == "initialize_from_config")
        install = [
            node.lineno for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "install"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "native_gate"
        ]
        validation = [
            node.lineno for node in ast.walk(function)
            if isinstance(node, ast.Raise) and node.exc is not None
            and "did not materialize a validated layout" in ast.dump(node.exc)
        ]
        self.assertEqual(len(install), 1)
        self.assertTrue(validation)
        self.assertGreater(min(install), max(validation))


class ScheduleTests(unittest.TestCase):
    def scheduler(self, rpc, epoch=1, pressure=False, probes=16):
        value = object.__new__(Scheduler)
        value._native_admission = None
        value._gate_rpc = rpc
        value._gate_pressure = pressure
        value._kvcached_epoch = epoch
        value._gate_probe_epochs = probes
        value.kv_cache_manager = NS(coordinator=None)
        value.actions = []
        value._release_cold_prefix = lambda: value.actions.append("evict")
        return value

    def run_schedule(self, value, tokens):
        output = NS(total_num_scheduled_tokens=tokens)
        return value.schedule(output)

    def test_idle_without_pressure_neither_evicts_nor_retires(self):
        rpc = Rpc(False)
        value = self.scheduler(rpc, epoch=3)
        self.run_schedule(value, 0)
        # Retiring on an empty step is cheap, but nothing is retired when no
        # rank reports pressure, and no prefix is sacrificed.
        self.assertEqual(rpc.actions, ["idle"])
        self.assertEqual(value.actions, [])

    def test_idle_evicts_only_after_a_rank_reports_pressure(self):
        rpc = Rpc(True, reclaimed=48)
        value = self.scheduler(rpc, epoch=3)
        self.run_schedule(value, 0)
        self.assertEqual(rpc.actions, ["idle"])
        self.assertEqual(value.actions, ["evict"])

    def test_busy_probe_surveys_pressure_even_after_a_low_sample(self):
        rpc = Rpc(True)
        value = self.scheduler(rpc, epoch=16, pressure=False)
        self.run_schedule(value, 7)
        # The old heartbeat required _gate_pressure, so this call never ran.
        self.assertEqual(rpc.actions, ["status"])
        self.assertEqual(value.actions, ["evict"])

    def test_busy_probe_between_periods_is_silent(self):
        rpc = Rpc(True)
        value = self.scheduler(rpc, epoch=5, pressure=False)
        self.run_schedule(value, 7)
        self.assertEqual(rpc.actions, [])
        self.assertEqual(value.actions, [])

    def test_busy_probe_reports_no_pressure_without_evicting(self):
        rpc = Rpc(False)
        value = self.scheduler(rpc, epoch=32, pressure=False)
        self.run_schedule(value, 7)
        self.assertEqual(rpc.actions, ["status"])
        self.assertEqual(value.actions, [])

    def test_in_process_gate_retires_before_evicting(self):
        arena = FakeArena(budget=100, charged=100)
        gate.install(arena, FakeOwners({"0": {0}}))
        try:
            value = self.scheduler(None, epoch=4)
            self.run_schedule(value, 0)
            self.assertEqual(len(arena.transitions), 1)
            self.assertEqual(value.actions, ["evict"])
        finally:
            gate.clear()


if __name__ == "__main__":
    unittest.main()
