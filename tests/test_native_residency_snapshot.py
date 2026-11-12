# SPDX-License-Identifier: Apache-2.0
"""Ownership invariants independent of framework/device installation."""
import importlib.util
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace as NS
import unittest

spec = importlib.util.spec_from_file_location(
    "native_residency_under_test", Path(__file__).parents[1] /
    "kvcached/integration/vllm/native_residency.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class OwnershipTest(unittest.TestCase):
    def fixture(self):
        blocks = [NS(block_id=i, ref_cnt=0, block_hash=None, is_null=i == 0)
                  for i in range(8)]
        pool = NS(blocks=blocks, cached_block_hashes_by_block={}, _reuse_watchers={})
        managers = [NS(block_pool=pool, req_to_blocks={}) for _ in range(3)]
        output = NS(new_block_ids_to_zero=[], kv_cache_block_copies=[])
        return NS(block_pool=pool, single_type_managers=managers), output

    def test_prefix_alias_survives_request_free_and_pickle(self):
        coordinator, output = self.fixture()
        pool = coordinator.block_pool
        pool.blocks[2].block_hash = (1, "hash")
        pool.cached_block_hashes_by_block[2] = {(2, "alias")}
        snap = module.snapshot_ownership(coordinator, output, epoch=1,
                                         decode_group=lambda key: key[0])
        self.assertEqual(snap.owners["prefix"], ((1, 2), (2, 2)))
        self.assertFalse(snap.owners["requests"])
        output.kvcached_ownership = snap
        self.assertEqual(pickle.loads(pickle.dumps(output)).kvcached_ownership, snap)

    def test_unscheduled_request_watchers_pins_and_copy(self):
        coordinator, output = self.fixture()
        pool = coordinator.block_pool
        coordinator.single_type_managers[1].req_to_blocks["not-scheduled"] = [pool.blocks[3]]
        pool.blocks[3].ref_cnt = 1
        pool.blocks[4].ref_cnt = 1
        pool._reuse_watchers[5] = object()
        output.kv_cache_block_copies = [NS(src_block_id=6, dst_block_id=7)]
        snap = module.snapshot_ownership(coordinator, output, epoch=7, decode_group=lambda k: k[0])
        self.assertEqual(snap.owners["requests"], ((1, 3),))
        for group in range(3):
            self.assertIn((group, 4), snap.owners["native-pins"])
            self.assertNotIn((group, 3), snap.owners["native-pins"])
            self.assertIn((group, 5), snap.owners["native-watchers"])
            self.assertIn((group, 6), snap.owners["step-io"])
            self.assertIn((group, 7), snap.owners["step-io"])
            self.assertIn((group, 0), snap.owners["null"])

    def test_reject_invalid_native_identity(self):
        coordinator, output = self.fixture()
        coordinator.block_pool.blocks[2].block_hash = (3, "invalid-group")
        with self.assertRaises(ValueError):
            module.snapshot_ownership(coordinator, output, epoch=1, decode_group=lambda k: k[0])
        coordinator.single_type_managers[1].block_pool = object()
        with self.assertRaises(ValueError):
            module.snapshot_ownership(coordinator, output, epoch=1, decode_group=lambda k: k[0])


if __name__ == "__main__":
    unittest.main()
