# SPDX-License-Identifier: Apache-2.0
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


relief = load('capacity_relief_test_module', ROOT / 'kvcached/capacity_relief.py')
fixtures = load('capacity_relief_fixtures', Path(__file__).with_name('test_physical_residency.py'))


class CapacityReliefTest(unittest.TestCase):
    def test_native_handoff_refreshes_budget_before_immediate_retry(self):
        tree = ast.parse((ROOT / 'kvcached/integration/vllm/native_storage.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == 'NativeStorageArena')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                      and n.name == '_release_unused_allocator')
        accounting = {'free': 0, 'budget': 0}
        events = []

        def empty_cache():
            accounting['free'] = 64
            events.append('release')

        def recalibrate():
            accounting['budget'] = accounting['free']
            events.append('refresh')

        def ensure(desired):
            events.append('ensure')
            if accounting['budget'] < 64:
                raise RuntimeError('capacity_exhausted: stale budget')
            return 64

        namespace = {'torch': NS(npu=NS(empty_cache=empty_cache))}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<actual-native-handoff>', 'exec'), namespace)
        arena = NS(operations=NS(recalibrate_physical_budget=recalibrate))
        state = NS(poisoned=False, ensure=ensure, synchronize=lambda: events.append('fence'))
        result = relief.ensure_with_allocator_relief(
            state, {}, lambda: namespace['_release_unused_allocator'](arena),
            lambda: dict(accounting), lambda event: None)
        self.assertEqual(result, 64)
        self.assertEqual(events, ['ensure', 'fence', 'release', 'refresh', 'ensure'])
        self.assertFalse(state.poisoned)

    def test_partial_map_rollback_precedes_allocator_handoff(self):
        memory = fixtures.Memory()
        state = fixtures.module.PhysicalResidency(memory, memory.synchronize, memory.initialize)
        for index in range(2):
            state.register(NS(key=str(index), reservation_bytes=48, size_bytes=41, page_bytes=16), index)
        state.ensure({'0': {0}, '1': {0}})
        memory.data[0][0] = 'live prefix'
        original = memory.map_to_kv_tensors_with_result
        blocked = [True]

        def mapping(pages, pool):
            if pool == 1 and blocked[0]:
                raise RuntimeError('capacity_exhausted: shared KV budget would be exceeded')
            return original(pages, pool)

        memory.map_to_kv_tensors_with_result = mapping

        def release():
            self.assertEqual(state.mapped, {'0': {0}, '1': {0}})
            self.assertEqual(memory.data, {0: {0: 'live prefix'}, 1: {0: 0}})
            self.assertEqual(memory.log[-1], 'fence')
            blocked[0] = False

        events = []
        grown = relief.ensure_with_allocator_relief(
            state, state.full(), release, state.status, events.append)
        self.assertEqual(grown, 64)
        self.assertEqual(memory.data[0][0], 'live prefix')
        self.assertEqual(events[0]['before']['mapped_bytes'], 32)
        self.assertEqual(events[0]['after']['mapped_bytes'], 32)
        self.assertFalse(state.poisoned)

    def test_persistent_capacity_rejection_retries_only_once(self):
        state = NS(poisoned=False, ensure=Mock(side_effect=RuntimeError('capacity_exhausted: full')),
                   synchronize=Mock())
        release, observe, report = Mock(), Mock(return_value={}), Mock()
        with self.assertRaisesRegex(RuntimeError, 'capacity_exhausted'):
            relief.ensure_with_allocator_relief(state, {}, release, observe, report)
        self.assertEqual(state.ensure.call_count, 2)
        release.assert_called_once()
        self.assertFalse(state.poisoned)

    def test_success_does_not_flush_allocator(self):
        state = NS(poisoned=False, ensure=Mock(return_value=64), synchronize=Mock())
        release, observe, report = Mock(), Mock(), Mock()
        self.assertEqual(relief.ensure_with_allocator_relief(state, {}, release, observe, report), 64)
        release.assert_not_called()
        observe.assert_not_called()
        state.synchronize.assert_not_called()

    def test_unknown_or_poisoned_failures_never_retry(self):
        for error, poisoned in [(RuntimeError('driver failure'), False),
                                (MemoryError('host allocation'), False),
                                (RuntimeError('capacity_exhausted: quarantined'), True)]:
            with self.subTest(error=str(error), poisoned=poisoned):
                state = NS(poisoned=poisoned, ensure=Mock(side_effect=error), synchronize=Mock())
                release = Mock()
                with self.assertRaises(type(error)):
                    relief.ensure_with_allocator_relief(state, {}, release, Mock(), Mock())
                self.assertEqual(state.ensure.call_count, 1)
                release.assert_not_called()

    def test_failed_handoff_poisoned_without_retry(self):
        for failing in ['fence', 'release']:
            with self.subTest(failing=failing):
                state = NS(poisoned=False, ensure=Mock(side_effect=RuntimeError('capacity_exhausted: full')),
                           synchronize=Mock(side_effect=RuntimeError('device fault') if failing == 'fence' else None))
                release = Mock(side_effect=RuntimeError('device fault') if failing == 'release' else None)
                with self.assertRaisesRegex(RuntimeError, 'device fault'):
                    relief.ensure_with_allocator_relief(state, {}, release, Mock(), Mock())
                self.assertTrue(state.poisoned)
                self.assertEqual(state.ensure.call_count, 1)


class ControlResponseTest(unittest.TestCase):
    def setUp(self):
        # Execute the actual worker method without importing torch/CANN. This
        # isolates the RPC contract from hardware setup, not a copied method.
        tree = ast.parse((ROOT / 'kvcached/integration/vllm/ascend_generic_worker.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'kvcached_native_control')
        runtime = NS(mem_get_info=Mock(return_value=(1024, 4096)), synchronize=Mock(), empty_cache=Mock())
        namespace = dict(torch=NS(npu=runtime), time=NS(monotonic=lambda: 1), json=json,
                         is_capacity_rejection=relief.is_capacity_rejection, print=Mock())
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<actual-worker-control>', 'exec'), namespace)
        self.call = namespace['kvcached_native_control']
        self.arena = NS(residency=NS(poisoned=False, full=lambda: {'0': {0}}, reclaim=Mock(return_value=0)),
                        ensure=Mock(), capacity_relief_attempts=1, last_capacity_relief={})
        self.worker = NS(native_arena=self.arena, native_elastic=True,
                         _native_status=lambda: dict(poisoned=self.arena.residency.poisoned))

    def test_rejected_restore_is_normal_reply_and_next_status_is_readable(self):
        self.arena.ensure.side_effect = RuntimeError('capacity_exhausted: full')
        response = self.call(self.worker, 'held')
        self.assertFalse(response['ok'])
        self.assertEqual(response['action'], 'held')
        self.assertTrue(self.worker.native_elastic)
        response = self.call(self.worker, 'status')
        self.assertTrue(response['ok'])
        self.assertEqual(response['action'], 'status')

    def test_success_commits_policy_and_fatal_failure_is_not_success(self):
        self.assertTrue(self.call(self.worker, 'held')['ok'])
        self.assertFalse(self.worker.native_elastic)
        self.arena.residency.poisoned = True
        self.arena.ensure.side_effect = RuntimeError('capacity_exhausted: quarantined')
        with self.assertRaises(RuntimeError):
            self.call(self.worker, 'held')


if __name__ == '__main__':
    unittest.main()
