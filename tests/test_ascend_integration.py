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


def test_ascend_classes_are_patched_only_after_native_import(monkeypatch):
    import sys

    from kvcached.integration.vllm import ascend

    callbacks, applied = {}, []

    def register(name):
        return lambda callback: callbacks.setdefault(name, callback)

    class FakePatch:
        patch_name = "test"

        def __init__(self, name):
            self.target_module = name

        def apply(self, module):
            applied.append(module)
            return True

    monkeypatch.setitem(sys.modules, "wrapt.importer", SimpleNamespace(when_imported=register))
    monkeypatch.setattr(ascend, "is_integration_version_supported", lambda *_: True)
    monkeypatch.setattr(ascend, "NPUModelRunnerPatch", lambda: FakePatch("test.runner"))
    monkeypatch.setattr(ascend, "NPUWorkerPatch", lambda: FakePatch("test.worker"))
    ascend.register_ascend_patches()
    assert applied == []
    assert set(callbacks) == {"test.runner", "test.worker"}
    module = SimpleNamespace()
    callbacks["test.worker"](module)
    assert applied == [module]


def test_sglang_npu_rejected_before_applying_framework_patches(monkeypatch):
    monkeypatch.setenv("KVCACHED_AUTOPATCH", "false")
    from kvcached.integration.sglang import autopatch

    monkeypatch.setenv("KVCACHED_AUTOPATCH", "true")
    monkeypatch.setattr(autopatch, "get_device_type", lambda: "npu")
    with pytest.raises(KVCachedConfigError, match="vLLM-Ascend only"):
        autopatch._patch_sglang(SimpleNamespace())
