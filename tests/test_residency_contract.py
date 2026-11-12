# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Dependency-free contract tests; no model name or accelerator required."""

import importlib.util
import itertools
import sys
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "residency_contract_under_test", Path(__file__).parents[1] / "kvcached/residency.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
Storage, BlockSpan = module.Storage, module.BlockSpan
Layout, Owners = module.Layout, module.ResidencyOwners


class ResidencyContractTests(unittest.TestCase):
    def layout(self):
        return Layout([Storage("shared", 120, 16), Storage("state", 48, 16)], [
            BlockSpan("shared", 0, 0, 12, 12, 10),
            BlockSpan("shared", 1, 3, 12, 9, 10),
            BlockSpan("state", 1, 0, 4, 4, 10),
        ])

    def test_crossing_pages_and_group_aliases(self):
        layout = self.layout()
        owners = Owners(layout)
        owners.replace({"request:a": [(0, 1)], "request:b": [(1, 1)]})
        self.assertEqual(owners.required_pages(), {"shared": {0, 16}, "state": {0}})
        owners.replace({}, ["request:a"])
        self.assertEqual(owners.required_pages(), {"shared": {0, 16}, "state": {0}})
        owners.replace({}, ["request:b"])
        self.assertEqual(owners.required_pages(), {"shared": set(), "state": set()})

    def test_request_end_does_not_release_prefix_copy_or_inflight_graph(self):
        owners = Owners(self.layout())
        owners.replace({"request": [(0, 2)], "prefix": [(0, 2)], "copy": [(1, 5)]})
        ready = [False]
        owners.begin_execution("graph:7", [(0, 2)], lambda: ready[0])
        before = owners.required_pages()
        owners.replace({}, ["request"])
        self.assertEqual(owners.required_pages(), before)
        owners.replace({}, ["prefix", "copy"])
        self.assertEqual(owners.required_pages()["shared"], {16, 32})
        self.assertEqual(owners.collect_completed(), 0)
        ready[0] = True
        self.assertEqual(owners.collect_completed(), 1)
        self.assertFalse(any(owners.required_pages().values()))

    def test_invalid_update_is_atomic(self):
        owners = Owners(self.layout())
        owners.replace({"kept": [(0, 1)]})
        before = owners.snapshot()
        with self.assertRaises(ValueError):
            owners.replace({"bad": [(1, 100)]}, ["kept"])
        self.assertEqual(before, owners.snapshot())
        with self.assertRaises(ValueError):
            owners.replace({"new": [(0, 2)]}, ["unknown"])
        self.assertEqual(before, owners.snapshot())

    def test_fence_failure_retains_all_executions(self):
        owners = Owners(self.layout())
        owners.begin_execution("finished", [(0, 1)], lambda: True)
        def fail():
            raise RuntimeError("device error")
        owners.begin_execution("failed", [(1, 2)], fail)
        before = owners.snapshot()
        with self.assertRaises(RuntimeError):
            owners.collect_completed()
        self.assertEqual(before, owners.snapshot())

    def test_layout_bounds_and_ambiguous_identity_rejected(self):
        with self.assertRaises(ValueError):
            Layout([Storage("x", 9, 4)], [BlockSpan("x", 0, 0, 4, 4, 3)])
        with self.assertRaises(ValueError):
            Layout([Storage("x", 9, 4), Storage("x", 9, 4)], [])
        with self.assertRaises(ValueError):
            Layout([Storage("x", 9, 4)], [])

    def test_native_tensor_views_cover_actual_elements(self):
        # Blocks-first contiguous, head-major, packed layer gaps, split virtual
        # blocks and component-major recurrent state, using one shared contract.
        geometries = [
            ((8, 2, 3), (6, 3, 1), 1, 8),
            ((8, 2, 3), (3, 24, 1), 2, 8),
            ((8, 2, 3), (18, 3, 1), 2, 8),
            ((16, 2, 3), (6, 3, 1), 1, 8),
            ((8, 2, 3), (10, 3, 1), 4, 8),
        ]
        for shape, strides, itemsize, count in geometries:
            with self.subTest(shape=shape, strides=strides, itemsize=itemsize):
                offset = 7
                spans = module.block_first_spans(storage="x", group_id=0, offset=offset,
                    shape=shape, strides=strides, itemsize=itemsize, num_blocks=count)
                nbytes = offset + itemsize * (1 + sum((d-1)*s for d,s in zip(shape,strides)))
                layout = Layout([Storage("x", nbytes, 16)], spans)
                ratio = shape[0] // count
                for block in range(count):
                    actual = set()
                    for row in range(block * ratio, (block+1) * ratio):
                        for index in itertools.product(*(range(d) for d in shape[1:])):
                            address = offset + itemsize*(row*strides[0] + sum(i*s for i,s in zip(index,strides[1:])))
                            actual.update((address+b)//16*16 for b in range(itemsize))
                    self.assertTrue(actual <= layout.pages_for([(0,block)])["x"])

    def test_group_count_is_not_model_specific(self):
        for count in (1, 2, 4, 7):
            layout = Layout([Storage("x", 64, 16)], [
                BlockSpan("x", group, 0, 8, 8, 8) for group in range(count)])
            owners = Owners(layout)
            owners.replace({str(group): [(group, group)] for group in range(count)})
            expected = {group*8//16*16 for group in range(count)}
            self.assertEqual(owners.required_pages()["x"], expected)


if __name__ == "__main__":
    unittest.main()
