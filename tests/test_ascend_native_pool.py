# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


@pytest.fixture
def pool_class(monkeypatch):
    root = Path(__file__).parents[1] / 'kvcached/integration/vllm'
    result = None
    for stem in ('ascend_native_geometry', 'ascend_native_pool'):
        name = 'kvcached.integration.vllm.' + stem
        spec = importlib.util.spec_from_file_location(name, root / (stem + '.py'))
        result = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, result)
        spec.loader.exec_module(result)
    return result.NativeKVPool


class Tensor:
    def __init__(self, events, group, region=None):
        self.events, self.group, self.region = events, group, region

    def __getitem__(self, region):
        return Tensor(self.events, self.group, region)

    def zero_(self):
        self.events.append(('zero', self.group, self.region.start, self.region.stop))


class Operations:
    def __init__(self):
        self.mapped, self.pending, self.events = {}, {}, []
        self.fail_map = None
        self.fail_prepare = None

    def init_kvcached(self, *args):
        pass

    def create_kv_tensors(self, size, dtype, device, layers, buffers, group, unified, page):
        self.mapped[group] = set()
        return [Tensor(self.events, group)]

    def map_to_kv_tensors_with_result(self, offsets, group):
        if group == self.fail_map:
            raise MemoryError('injected physical allocation failure')
        assert not self.mapped[group].intersection(offsets)
        self.mapped[group].update(offsets)
        return True, offsets

    def unmap_from_kv_tensors(self, offsets, group):
        self.mapped[group].difference_update(offsets)
        return True

    def prepare_unmap_from_kv_tensors(self, offsets, tx, group):
        self.events.append(('prepare', group))
        if group == self.fail_prepare:
            raise RuntimeError('injected unmap failure')
        self.pending[(tx, group)] = offsets
        return True

    def abort_unmap_from_kv_tensors(self, tx, group):
        self.pending.pop((tx, group))
        return True

    def commit_unmap_from_kv_tensors(self, tx, group):
        self.mapped[group].difference_update(self.pending.pop((tx, group)))
        return True


def make_pool(pool_class):
    ops = Operations()
    page = 2 * 1024**2
    descriptors = [NS(size=8 * page, block_stride=page // 2, offset=0, layer_stride=0,
                      layers=[f'long{i}', f'ring{i}'], host_resident=False) for i in range(2)]
    pool = pool_class(NS(num_blocks=16, kv_cache_tensors=descriptors), 'npu:0', ops,
                      lambda: ops.events.append(('sync',)), 'unit-test')
    return pool, ops, page


def test_native_aliases_and_live_shared_pages_survive(pool_class):
    pool, ops, page = make_pool(pool_class)
    views = pool.views_by_layer()
    assert views['long0'] is views['ring0']
    pool.ensure(full=True)
    ops.events.clear()
    assert pool.reclaim({2, 3}) == 12 * page
    assert ops.events[0] == ('sync',)
    assert ops.mapped == {0: {0, page}, 1: {0, page}}
    pool.ensure({2, 3, 5})
    assert ops.mapped == {0: {0, page, 2 * page}, 1: {0, page, 2 * page}}
    zeroed = [event for event in ops.events if event[0] == 'zero']
    assert zeroed == [('zero', 0, 2 * page, 3 * page), ('zero', 1, 2 * page, 3 * page)]


def test_grow_failure_rolls_back_only_new_pages(pool_class):
    pool, ops, page = make_pool(pool_class)
    pool.ensure({2})
    previous = {group: pages.copy() for group, pages in ops.mapped.items()}
    ops.fail_map = 1
    with pytest.raises(MemoryError):
        pool.ensure(full=True)
    assert ops.mapped == previous == pool.mapped
    assert not pool.poisoned
    ops.fail_map = None
    pool.ensure(full=True)
    assert pool.status()['mapped_bytes'] == 16 * page


def test_retirement_prepare_failure_restores_prior_groups(pool_class):
    pool, ops, page = make_pool(pool_class)
    pool.ensure(full=True)
    ops.fail_prepare = 1
    with pytest.raises(RuntimeError, match='injected unmap'):
        pool.reclaim({0})
    assert not ops.pending
    assert pool.status()['mapped_bytes'] == 16 * page
    assert not pool.poisoned


def test_native_wrapper_runs_before_views_and_initialization(pool_class):
    ops = Operations()
    page = 2 * 1024**2
    descriptor = NS(size=2 * page, block_stride=page, offset=0, layer_stride=0,
                    layers=['long', 'ring'], host_resident=False)
    wrapped = []

    def wrap(tensor):
        assert tensor.region is None
        assert not ops.events
        wrapped.append(tensor)
        return tensor

    pool = pool_class(NS(num_blocks=2, kv_cache_tensors=[descriptor]), 'npu:0', ops,
                      lambda: None, 'wrapper-test', wrap_storage=wrap)
    assert len(wrapped) == 1
    assert pool.views_by_layer()['long'] is pool.views_by_layer()['ring']
    pool.ensure(full=True)
    assert ops.events == [('zero', 0, 0, 2 * page)]
