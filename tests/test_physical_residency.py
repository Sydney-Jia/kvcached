# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

spec = importlib.util.spec_from_file_location(
    "physical_residency_test_module", Path(__file__).parents[1] / "kvcached/physical_residency.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Memory:
    def __init__(self):
        self.data = {0: {}, 1: {}}
        self.pending = {}
        self.fail_map = None
        self.fail_prepare = None
        self.fence_ok = True
        self.log = []

    def synchronize(self):
        self.log.append('fence')
        if not self.fence_ok:
            raise RuntimeError('device error')

    def map_to_kv_tensors_with_result(self, pages, pool):
        if pool == self.fail_map:
            return False, []
        self.data[pool].update({page: 'uninitialized' for page in pages})
        return True, pages

    def initialize(self, key, pages):
        self.data[int(key)].update({page: 0 for page in pages})

    def unmap_from_kv_tensors(self, pages, pool):
        for page in pages:
            del self.data[pool][page]
        return True

    def prepare_unmap_from_kv_tensors(self, pages, transaction, pool):
        self.log.append(('prepare', pool))
        if pool == self.fail_prepare:
            return False
        self.pending[transaction, pool] = {p: self.data[pool].pop(p) for p in pages}
        return True

    def abort_unmap_from_kv_tensors(self, transaction, pool):
        self.data[pool].update(self.pending.pop((transaction, pool)))
        return True

    def commit_unmap_from_kv_tensors(self, transaction, pool):
        self.pending.pop((transaction, pool))
        return True


class PhysicalTest(unittest.TestCase):
    def setUp(self):
        self.memory = Memory()
        self.pool = module.PhysicalResidency(self.memory, self.memory.synchronize,
                                             self.memory.initialize)
        for index in range(2):
            self.pool.register(NS(key=str(index), reservation_bytes=48,
                                  size_bytes=41, page_bytes=16), index)

    def test_reclaim_and_regrow_do_not_overwrite_live_shared_page(self):
        self.pool.ensure(self.pool.full())
        self.memory.data[0][16] = 'live state'
        self.assertEqual(self.pool.reclaim({'0': {16}, '1': {0}}), 64)
        self.assertEqual(self.memory.log[0], 'fence')
        self.pool.ensure({'0': {0, 16}, '1': {0}})
        self.assertEqual(self.memory.data[0], {0: 0, 16: 'live state'})
        self.assertEqual(self.pool.status()['mapped_bytes'], 48)

    def test_partial_growth_failure_rolls_back_only_new_pages(self):
        self.pool.ensure({'0': {0}, '1': {0}})
        self.memory.data[0][0] = 'live'
        self.memory.fail_map = 1
        with self.assertRaises(MemoryError):
            self.pool.ensure(self.pool.full())
        self.assertEqual(self.memory.data, {0: {0: 'live'}, 1: {0: 0}})
        self.assertEqual(self.pool.mapped, {'0': {0}, '1': {0}})
        self.assertFalse(self.pool.poisoned)

    def test_retirement_rollback_preserves_data(self):
        self.pool.ensure(self.pool.full())
        self.memory.data[0][32] = 'survive failed transaction'
        self.memory.fail_prepare = 1
        with self.assertRaises(RuntimeError):
            self.pool.reclaim({'0': {0}, '1': {0}})
        self.assertEqual(self.memory.data[0][32], 'survive failed transaction')
        self.assertEqual(self.pool.mapped, self.pool.full())
        self.assertFalse(self.pool.poisoned)

    def test_completion_failure_prevents_unmapping(self):
        self.pool.ensure(self.pool.full())
        self.memory.fence_ok = False
        with self.assertRaises(RuntimeError):
            self.pool.reclaim({'0': {0}, '1': {0}})
        self.assertEqual(self.pool.mapped, self.pool.full())
        self.assertFalse(self.memory.pending)

    def test_invalid_snapshot_is_rejected_before_any_mapping(self):
        for pages in ({'0': {0}}, {'0': {49}, '1': {0}}, {'0': {1}, '1': {0}}):
            with self.assertRaises(ValueError):
                self.pool.ensure(pages)
        self.assertEqual(self.memory.data, {0: {}, 1: {}})


if __name__ == '__main__':
    unittest.main()
