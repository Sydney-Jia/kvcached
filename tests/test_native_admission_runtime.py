# SPDX-License-Identifier: Apache-2.0
"""Run unmodified vLLM Python allocation functions without importing torch.

Set KVCACHED_VLLM_SOURCE to a vLLM checkout. Classes/functions are loaded by AST
from that checkout; coordinator/executor are fixtures. This checks real pool
and allocate_slots semantics, not a complete framework startup or device run.
"""

import ast
from dataclasses import dataclass, field
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest

from test_native_admission import admission, capacity, Rpc


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise ValueError(f"Native source contract missing in {path}")
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


@unittest.skipUnless(
    os.getenv("KVCACHED_VLLM_SOURCE"), "Set KVCACHED_VLLM_SOURCE for real native Python interfaces"
)
class NativePythonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(os.environ["KVCACHED_VLLM_SOURCE"]) / "vllm/v1/core"
        module = ModuleType("_real_native_capacity_test")
        sys.modules[module.__name__] = module
        cls.ns = module.__dict__
        cls.ns.update(dataclass=dataclass, field=field, MEDIUM_GPU="GPU")
        definitions(root / "kv_cache_utils.py", ["KVCacheBlock", "FreeKVCacheBlockQueue"], cls.ns)
        definitions(root / "block_pool.py", ["BlockHashToBlockMap", "BlockPool"], cls.ns)
        tree = ast.parse((root / "kv_cache_manager.py").read_text(encoding="utf-8"))
        manager = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "KVCacheManager"
        )
        function = next(
            n for n in manager.body if isinstance(n, ast.FunctionDef) and n.name == "allocate_slots"
        )
        selected = ast.Module(
            body=[
                ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                function,
            ],
            type_ignores=[],
        )
        cls.ns.update(
            RequestStatus=NS(WAITING="waiting", PREEMPTED="preempted"),
            cdiv=lambda n, d: (n + d - 1) // d,
        )
        exec(
            compile(ast.fix_missing_locations(selected), str(root / "kv_cache_manager.py"), "exec"),
            cls.ns,
        )

    def setUp(self):
        pool = self.ns["BlockPool"](20, True, 16)
        groups = [NS(block_pool=pool, req_to_blocks={}) for _ in range(2)]

        def demand(**kwargs):
            existing = groups[0].req_to_blocks.get(kwargs["request_id"], ())
            return max(0, (kwargs["num_tokens"] + 15) // 16 - len(existing))

        def allocate(request_id, num_tokens, *_):
            existing = groups[0].req_to_blocks.setdefault(request_id, [])
            count = max(0, (num_tokens + 15) // 16 - len(existing))
            added = pool.get_new_blocks(count)
            existing.extend(added)
            return (added, [])

        coordinator = NS(
            block_pool=pool,
            single_type_managers=groups,
            get_num_blocks_to_allocate=demand,
            remove_skipped_blocks=lambda *a, **k: None,
            allocate_new_blocks=allocate,
            cache_blocks=lambda *a, **k: None,
        )
        klass = type(
            "RealAllocationWithFixtureCoordinator",
            (),
            {"allocate_slots": self.ns["allocate_slots"]},
        )
        manager = klass()
        manager.coordinator = coordinator
        manager.block_pool = pool
        manager.max_model_len = 4096
        manager.retained_hit_group_ids = ()
        manager.empty_kv_cache_blocks = NS(blocks=([], []))
        manager.watermark_blocks = 0
        manager.create_kv_cache_blocks = lambda blocks: blocks
        self.pool, self.manager = pool, manager
        self.rpc = Rpc()
        self.guard = admission.NativeAdmission(
            manager, lambda key: key[0], self.rpc, (0, 1), capacity.NativeBudget()
        )
        self.request = NS(
            request_id="a",
            num_computed_tokens=0,
            num_tokens=32,
            num_prompt_tokens=32,
            num_in_flight_tokens=0,
            status="waiting",
        )

    def test_real_allocate_and_pool_accept_prepared_ids(self):
        self.guard.begin_step(1)
        result = self.manager.allocate_slots(self.request, 32, full_sequence_must_fit=True)
        self.assertEqual([b.block_id for b in result[0]], [1, 2])
        self.assertEqual(self.rpc.mapped, {0: {1, 2}, 1: {1, 2}})

    def test_real_allocate_returns_none_on_physical_rejection(self):
        self.rpc.reject = lambda snapshot, ids, rank: rank == 1 and bool(ids)
        self.guard.begin_step(1)
        result = self.manager.allocate_slots(self.request, 32)
        self.assertIsNone(result)
        self.assertEqual(self.pool.get_num_free_blocks(), 19)
        self.assertEqual(self.pool.blocks[1].ref_cnt, 0)
        self.assertFalse(self.manager.coordinator.single_type_managers[0].req_to_blocks)

    def test_real_prefix_alias_removal_and_watcher_protection(self):
        for bid in (3, 4):
            self.pool._insert_block_hash((0, "hash" + str(bid)), self.pool.blocks[bid], 16)
            self.pool._insert_block_hash((1, "alias" + str(bid)), self.pool.blocks[bid], 16)
        self.pool.touch([self.pool.blocks[4]])
        self.pool.unpin_blocks([self.pool.blocks[4]], lambda block: None)
        self.guard.budget = capacity.NativeBudget(prefix_max_blocks=0)
        self.guard.reconcile()
        self.assertIsNone(self.pool.blocks[3].block_hash)
        self.assertNotIn(3, self.pool.cached_block_hashes_by_block)
        self.assertFalse(self.pool.cached_block_hash_to_block.contain((1, "alias3"), 3))
        self.assertIsNotNone(self.pool.blocks[4].block_hash)
        self.assertIn(4, self.pool._reuse_watchers)


if __name__ == "__main__":
    unittest.main()
