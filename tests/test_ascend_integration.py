# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from kvcached.integration.vllm.ascend import NPUWorkerPatch, validate_ascend_config
from kvcached.utils import KVCachedConfigError


def config():
    return SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=True, enable_sleep_mode=False),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1),
        cache_config=SimpleNamespace(cache_dtype="auto", kv_cache_memory_bytes=1024),
        use_v2_model_runner=False, kv_transfer_config=None,
    )


@pytest.mark.parametrize("owner,field,value", [
    ("model_config", "enable_sleep_mode", True),
    ("model_config", "enforce_eager", False),
    ("parallel_config", "tensor_parallel_size", 2),
    ("cache_config", "cache_dtype", "fp8"),
    ("cache_config", "kv_cache_memory_bytes", None),
])
def test_unsupported_configuration_fails_before_worker_initialization(monkeypatch, owner, field, value):
    cfg = config()
    setattr(getattr(cfg, owner), field, value)
    initialized = []

    class Worker:
        vllm_config = cfg

        def init_device(self):
            initialized.append(True)

        def shutdown(self):
            pass

    monkeypatch.setenv("ENABLE_KVCACHED", "true")
    assert NPUWorkerPatch().apply(SimpleNamespace(NPUWorker=Worker))
    with pytest.raises(KVCachedConfigError):
        Worker().init_device()
    assert initialized == []
    # Disabling the feature must preserve native behavior, even when that
    # native configuration is outside the elastic integration's current scope.
    monkeypatch.setenv("ENABLE_KVCACHED", "false")
    Worker().init_device()
    assert initialized == [True]


def test_single_device_explicit_budget_is_accepted():
    validate_ascend_config(config())
