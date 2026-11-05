# SPDX-License-Identifier: Apache-2.0
"""The observational control API can miss a concurrent ledger write."""
import importlib.util
from pathlib import Path
import sys
import types

import pytest


@pytest.fixture
def burst(monkeypatch):
    path = Path(__file__).parent / 'ascend' / 'service_burst.py'
    spec = importlib.util.spec_from_file_location('burst_control_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    control = types.ModuleType('kvcached.control')
    control.set_instance_memory_limit = lambda *a, **k: {'accepted': True}
    monkeypatch.setitem(sys.modules, 'kvcached.control', control)
    monkeypatch.setattr(module.time, 'sleep', lambda _: None)
    return module


def test_transient_observation_miss_does_not_cancel_published_resize(burst, monkeypatch):
    states = iter([{'status': 'unavailable', 'converged': False},
                   {'status': 'converged', 'converged': True}])
    monkeypatch.setattr(burst, 'limits', lambda _: next(states))
    result = burst.resize({'name': 'test'}, 1024)
    assert len(result['observations']) == 2
    assert result['observations'][-1]['state']['converged']


def test_real_control_failure_is_not_retried(burst, monkeypatch):
    monkeypatch.setattr(burst, 'limits', lambda _: {'status': 'failed', 'converged': False})
    with pytest.raises(RuntimeError, match='failed'):
        burst.resize({'name': 'test'}, 1024)


def test_zero_deadline_is_a_timeout_not_an_index_error(burst):
    with pytest.raises(TimeoutError):
        burst.resize({'name': 'test'}, 1024, timeout=0)
