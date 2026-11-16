# SPDX-License-Identifier: Apache-2.0
"""Native allocation/RPC boundary tests; not NPU or real-vLLM acceptance."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


capacity = load("kvcached.native_capacity", "kvcached/native_capacity.py")
load("kvcached.integration.vllm.native_residency", "kvcached/integration/vllm/native_residency.py")
admission = load("native_admission_under_test", "kvcached/integration/vllm/native_admission.py")


class Queue:
    def __init__(self, blocks):
        self.blocks = list(blocks)

    def get_all_free_blocks(self):
        return list(self.blocks)


class Pool:
    def __init__(self, size=12):
        self.blocks = [
            NS(block_id=i, is_null=i == 0, ref_cnt=0, block_hash=None) for i in range(size)
        ]
        self.free_block_queue = Queue(self.blocks[1:])
        self.cached_block_hashes_by_block = {}
        self._reuse_watchers = {}
        self.evictions = []

    def get_num_free_blocks(self):
        return len(self.free_block_queue.blocks)

    def get_new_blocks(self, count):
        if count > self.get_num_free_blocks():
            raise ValueError("logical exhaustion")
        blocks = self.free_block_queue.blocks[:count]
        del self.free_block_queue.blocks[:count]
        for block in blocks:
            block.ref_cnt += 1
            self.evict_blocks({block.block_id})
            self._reuse_watchers.pop(block.block_id, None)
        return blocks

    def evict_blocks(self, ids):
        for bid in ids:
            self.blocks[bid].block_hash = None
            self.cached_block_hashes_by_block.pop(bid, None)
            self.evictions.append(bid)

    def touch(self, blocks):
        for block in blocks:
            if block.ref_cnt == 0 and not block.is_null:
                self.free_block_queue.blocks.remove(block)
            block.ref_cnt += 1


class Manager:
    def __init__(self):
        self.pool = Pool()
        self.coordinator = NS(
            block_pool=self.pool,
            single_type_managers=[NS(block_pool=self.pool, req_to_blocks={}) for _ in range(3)],
            get_num_blocks_to_allocate=lambda count: count,
        )
        self.watermark_blocks = 0

    def allocate_slots(
        self,
        request,
        num_new_tokens,
        new_computed_blocks=None,
        reserved_blocks=0,
        has_scheduled_reqs=True,
    ):
        # A small protocol fixture: authoritative demand precedes the capacity
        # query; prefix touch precedes new allocations, like the pinned engine.
        needed = self.coordinator.get_num_blocks_to_allocate(num_new_tokens)
        if needed > self.pool.get_num_free_blocks() - reserved_blocks:
            return None
        if new_computed_blocks:
            for group in new_computed_blocks.blocks:
                self.pool.touch(group)
        got = self.pool.get_new_blocks(needed)
        self.coordinator.single_type_managers[0].req_to_blocks.setdefault(
            request.request_id, []
        ).extend(got)
        return got


class Rpc:
    def __init__(self, ranks=(0, 1)):
        self.ranks = ranks
        self.calls = []
        self.reject = lambda snapshot, candidates, rank: False
        self.rows_hook = lambda rows: rows
        self.mapped = {rank: set() for rank in ranks}

    def __call__(self, method, *, args):
        snapshot, candidates, budget, ticket = args
        self.calls.append((snapshot, set(candidates), budget, ticket))
        rows = []
        for rank in self.ranks:
            ok = not self.reject(snapshot, candidates, rank)
            if ok:
                self.mapped[rank] = set(candidates)
            rows.append(
                dict(
                    rank=rank,
                    ok=ok,
                    fatal=None,
                    ticket=ticket,
                    budget=dict(**budget, convergence="converged"),
                )
            )
        return self.rows_hook(rows)


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.manager = Manager()
        self.pool = self.manager.pool
        self.rpc = Rpc()
        self.guard = admission.NativeAdmission(
            self.manager, lambda key: key[0], self.rpc, (0, 1), capacity.NativeBudget()
        )

    def start(self):
        self.guard.begin_step(1)

    def test_all_ranks_prepare_before_native_pool_changes(self):
        before = tuple(self.pool.free_block_queue.blocks)
        self.rpc.reject = lambda snap, ids, rank: (
            self.assertEqual(tuple(self.pool.free_block_queue.blocks), before) or False
        )
        self.start()
        got = self.manager.allocate_slots(NS(request_id="a"), 2)
        self.assertEqual([b.block_id for b in got], [1, 2])
        self.assertEqual(self.rpc.mapped, {0: {1, 2}, 1: {1, 2}})

    def test_partial_rank_failure_unwinds_without_allocating_request(self):
        self.rpc.reject = lambda snap, ids, rank: rank == 1 and bool(ids)
        self.start()
        self.assertIsNone(self.manager.allocate_slots(NS(request_id="a"), 2))
        self.assertEqual(self.pool.get_num_free_blocks(), 11)
        self.assertEqual(self.rpc.mapped, {0: set(), 1: set()})
        self.assertEqual(self.guard.denied_allocations, 1)
        self.assertFalse(self.manager.coordinator.single_type_managers[0].req_to_blocks)

    def test_zero_new_block_demand_can_still_be_rejected_cleanly(self):
        hit = self.pool.blocks[3]
        hit.block_hash = (0, "prefix")
        self.rpc.reject = lambda snap, ids, rank: bool(ids)
        self.start()
        result = self.manager.allocate_slots(
            NS(request_id="a"), 0, new_computed_blocks=NS(blocks=((hit,),))
        )
        self.assertIsNone(result)
        self.assertEqual(hit.ref_cnt, 0)
        self.assertIsNotNone(hit.block_hash)

    def test_prefix_touch_changes_queue_but_actual_slots_are_prepared(self):
        hit = self.pool.blocks[1]
        hit.block_hash = (1, "shared")
        self.start()
        got = self.manager.allocate_slots(
            NS(request_id="a"), 2, new_computed_blocks=NS(blocks=((hit,),))
        )
        self.assertEqual([b.block_id for b in got], [2, 3])
        self.assertTrue({1, 2, 3} <= self.guard.step_candidates)

    def test_copy_source_of_running_request_is_prepared_in_all_groups(self):
        source = self.pool.blocks[6]
        self.pool.touch([source])
        self.manager.coordinator.single_type_managers[2].req_to_blocks["a"] = [source]
        self.start()
        self.manager.allocate_slots(NS(request_id="a"), 1)
        self.assertIn(6, self.guard.step_candidates)

    def test_bounded_prefix_eviction_preserves_active_aliases_and_watchers(self):
        for i in range(1, 6):
            self.pool.blocks[i].block_hash = (0, str(i))
            self.pool.cached_block_hashes_by_block[i] = {(2, "alias")}
        self.pool.touch([self.pool.blocks[2]])
        self.pool._reuse_watchers[3] = object()
        self.guard.budget = capacity.NativeBudget(prefix_max_blocks=1)
        self.guard.reconcile()
        self.assertIsNone(self.pool.blocks[1].block_hash)
        self.assertIsNone(self.pool.blocks[4].block_hash)
        self.assertIsNotNone(self.pool.blocks[2].block_hash)
        self.assertIsNotNone(self.pool.blocks[3].block_hash)
        self.assertIsNotNone(self.pool.blocks[5].block_hash)
        self.assertNotIn(1, self.pool.cached_block_hashes_by_block)

    def test_pressure_retries_after_cold_prefix_eviction(self):
        self.pool.blocks[8].block_hash = (0, "cold")
        self.rpc.reject = lambda snap, ids, rank: bool(ids) and bool(snap.owners["prefix"])
        self.start()
        self.assertIsNotNone(self.manager.allocate_slots(NS(request_id="a"), 1))
        self.assertIsNone(self.pool.blocks[8].block_hash)
        self.assertEqual(self.guard.evicted_prefix_blocks, 1)

    def test_idle_reconcile_drops_candidates_without_another_request(self):
        self.start()
        self.manager.allocate_slots(NS(request_id="a"), 2)
        self.guard.end_schedule()
        self.guard.reconcile()
        snapshot, ids, _, _ = self.rpc.calls[-1]
        self.assertFalse(ids)
        self.assertEqual(snapshot.owners["requests"], ((0, 1), (0, 2)))

    def test_step_base_preserves_queued_copy_sources(self):
        source = self.pool.blocks[7]
        self.pool.touch([source])
        native = self.manager.coordinator.single_type_managers[1]
        native.req_to_blocks["old"] = [source]
        self.start()
        del native.req_to_blocks["old"]
        source.ref_cnt = 0
        self.manager.allocate_slots(NS(request_id="a"), 1)
        self.assertIn((1, 7), self.rpc.calls[-1][0].owners["requests"])

    def test_revision_conflict_and_increase_do_not_evict_cache(self):
        self.guard.budget = capacity.NativeBudget(2, 100, None)
        self.pool.blocks[8].block_hash = (0, "cold")
        with self.assertRaises(ValueError):
            self.guard.set_budget(limit_bytes=200, prefix_max_blocks=None, expected_revision=1)
        result = self.guard.set_budget(limit_bytes=200, prefix_max_blocks=None, expected_revision=2)
        self.assertEqual(result["revision"], 3)
        self.assertEqual(result["convergence"], "converged")
        self.assertIsNotNone(self.pool.blocks[8].block_hash)

    def test_stats_queries_cannot_allocate_physical_pages(self):
        self.assertEqual(self.pool.get_num_free_blocks(), 11)
        self.assertEqual(self.rpc.calls, [])

    def test_incomplete_rpc_response_is_not_a_grant(self):
        self.rpc.rows_hook = lambda rows: rows[:1]
        self.start()
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            self.manager.allocate_slots(NS(request_id="a"), 1)
        self.assertEqual(len(self.pool.free_block_queue.blocks), 11)

    def test_native_unannounced_allocation_fails_before_mutation(self):
        with self.assertRaisesRegex(RuntimeError, "bypassed"):
            self.pool.get_new_blocks(1)
        self.start()
        with self.assertRaisesRegex(RuntimeError, "bypassed"):
            self.pool.get_new_blocks(1)
        self.assertEqual(len(self.pool.free_block_queue.blocks), 11)

    def test_old_rpc_result_cannot_authorize_a_new_allocation(self):
        self.rpc.rows_hook = lambda rows: [dict(row, ticket=0) for row in rows]
        self.start()
        with self.assertRaisesRegex(RuntimeError, "Stale"):
            self.manager.allocate_slots(NS(request_id="a"), 1)
        self.assertEqual(len(self.pool.free_block_queue.blocks), 11)


if __name__ == "__main__":
    unittest.main()
