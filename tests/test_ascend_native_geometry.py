# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


path = Path(__file__).parents[1] / "kvcached/integration/vllm/ascend_native_geometry.py"
spec = importlib.util.spec_from_file_location("ascend_native_geometry_under_test", path)
geometry = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = geometry
spec.loader.exec_module(geometry)


def descriptor(stride=3, num_blocks=8, layers=("long_kv", "index", "swa", "ring"), **updates):
    values = dict(size=num_blocks * stride, block_stride=stride, offset=0,
                  layer_stride=0, host_resident=False, layers=list(layers))
    values.update(updates)
    return NS(**values)


def output(new=(), ids=(), additions=(), finished=(), preempted=(), replacements=None, zero=(), copies=()):
    return NS(scheduled_new_reqs=list(new), finished_req_ids=set(finished),
              preempted_req_ids=set(preempted), block_table_updates=replacements,
              scheduled_cached_reqs=NS(req_ids=list(ids), new_block_ids=list(additions), resumed_req_ids=set()),
              new_block_ids_to_zero=list(zero), kv_cache_block_copies=list(copies))


def test_aliases_are_one_backing_and_strides_stay_distinct():
    config = NS(num_blocks=8, kv_cache_tensors=[descriptor(), descriptor(5, layers=("other",))])
    a, b = geometry.native_backings(config)
    assert a.layers == ("long_kv", "index", "swa", "ring")
    assert (a.block_stride, b.block_stride) == (3, 5)
    assert a.size_bytes == 24


def test_page_intersections_preserve_partially_used_pages():
    backing = geometry.NativeBacking(0, 30, 3, 10, ("cache",), page_bytes=8)
    assert backing.page_offsets([2]) == {0, 8}
    assert backing.page_offsets([3, 4]) == {8}
    assert backing.all_page_offsets() - backing.page_offsets([2, 3]) == {16, 24}
    assert backing.reservation_bytes == 32


@pytest.mark.parametrize("change", [dict(offset=1), dict(layer_stride=1), dict(host_resident=True),
                                    dict(size=23), dict(layers=[]), dict(block_stride=0)])
def test_reject_unproven_allocation_contracts(change):
    with pytest.raises(ValueError):
        geometry.native_backings(NS(num_blocks=8, kv_cache_tensors=[descriptor(**change)]))


def test_unscheduled_requests_and_group_alias_owners_remain_live():
    live = geometry.NativeLiveBlocks(2, 100)
    new = [NS(req_id="a", block_ids=([1, 2], [3])), NS(req_id="b", block_ids=([4], [5]))]
    assert live.update(output(new=new)) == {0, 1, 2, 3, 4, 5}
    assert live.update(output(ids=["a"], additions=[([6], [])])) == {0, 1, 2, 3, 4, 5, 6}
    assert live.update(output(finished=["a"])) == {0, 4, 5}


def test_replacement_preempts_append_and_preemption_retires():
    live = geometry.NativeLiveBlocks(2, 100)
    live.update(output(new=[NS(req_id="a", block_ids=([1], [2]))]))
    assert live.update(output(ids=["a"], additions=[([99], [])], replacements={"a": ([3], [4])})) == {0, 3, 4}
    assert live.update(output(preempted=["a"])) == {0}


def test_zero_and_copy_are_protected_until_device_completion():
    live = geometry.NativeLiveBlocks(2, 100)
    assert live.update(output(zero=[8], copies=[NS(src_block_id=9, dst_block_id=10)])) == {0, 8, 9, 10}
    assert live.live_blocks() == {0}


def test_invalid_update_cannot_drop_previous_owners():
    live = geometry.NativeLiveBlocks(2, 100)
    live.update(output(new=[NS(req_id="a", block_ids=([1], [2]))]))
    with pytest.raises(ValueError):
        live.update(output(finished=["a"], new=[NS(req_id="b", block_ids=([100], [3]))]))
    assert live.live_blocks() == {0, 1, 2}
