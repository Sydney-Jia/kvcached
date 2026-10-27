# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Ascend V1 integration using native attention kernels and elastic KV storage.

The initial contract is deliberately narrow: single-device eager MHA/GQA with
unquantized KV and an explicit virtual KV budget. Unsupported configurations
fail before allocating KV instead of silently serving with a static pool.
"""

from functools import wraps
from importlib.util import find_spec
from pathlib import Path

from kvcached.integration.patch_base import enable_kvcached
from kvcached.integration.vllm.patches import (
    GPUModelRunnerPatch,
    GPUWorkerPatch,
    _infer_attention_type,
)
from kvcached.utils import KVCachedConfigError


def validate_ascend_config(config):
    if not config.model_config.enforce_eager:
        raise KVCachedConfigError("Ascend elastic KV currently requires --enforce-eager")
    if getattr(config.model_config, "enable_sleep_mode", False):
        raise KVCachedConfigError("Ascend elastic KV cannot also be owned by camem sleep mode")
    parallel = config.parallel_config
    if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
        raise KVCachedConfigError("Ascend elastic KV currently requires TP=PP=1")
    if getattr(config, "use_v2_model_runner", False):
        raise KVCachedConfigError("Ascend elastic KV currently requires the V1 model runner")
    if getattr(config, "kv_transfer_config", None) is not None:
        raise KVCachedConfigError("Ascend KV transfer connectors are not yet supported")
    if config.cache_config.cache_dtype != "auto":
        raise KVCachedConfigError("Ascend elastic KV currently requires unquantized KV")
    if not getattr(config.cache_config, "kv_cache_memory_bytes", None):
        raise KVCachedConfigError(
            "Set --kv-cache-memory-bytes to the desired virtual KV capacity; "
            "physical usage will grow with live KV pages"
        )


class NPUModelRunnerPatch(GPUModelRunnerPatch):
    target_class = "NPUModelRunner"
    patch_name = "npu_model_runner"

    def __init__(self, module=None):
        super().__init__()
        if module is None:
            spec = find_spec("vllm_ascend")
            if spec is None or spec.origin is None:
                raise KVCachedConfigError("vllm_ascend is not installed")
            root = Path(spec.origin).parent
            module = ("vllm_ascend.worker.v1.model_runner"
                      if (root / "worker/v1/model_runner.py").is_file()
                      else "vllm_ascend.worker.model_runner_v1")
        self.target_module = module

    def apply(self, module):
        if not self.initialize_version_info():
            return False
        runner = self._get_target_class(module)
        if runner is None:
            return False
        if self._is_already_patched(runner, "__kvcached_ascend__"):
            return True

        original_init = runner.__init__

        @wraps(original_init)
        def initialize(instance, *args, **kwargs):
            original_init(instance, *args, **kwargs)
            if enable_kvcached():
                validate_ascend_config(instance.vllm_config)
                from kvcached.integration.vllm import interfaces

                if interfaces._contiguous_layout:
                    raise KVCachedConfigError(
                        "Ascend attention requires KVCACHED_CONTIGUOUS_LAYOUT=false")
                interfaces.init_kvcached(is_worker=True, device=str(instance.device))

        runner.__init__ = initialize
        self.add_kvcache_allocator(runner)
        self.add_reshape_methods(runner)
        generic_allocate = runner._allocate_kv_cache_from_kvcached
        generic_reshape = runner._reshape_kv_cache_tensors_from_kvcached

        def allocate(instance, config):
            if len(config.kv_cache_groups) != 1 or _infer_attention_type(config) not in ("MHA", "GQA"):
                raise KVCachedConfigError("Ascend elastic KV currently supports one MHA/GQA group")
            spec = config.kv_cache_groups[0].kv_cache_spec
            if getattr(spec, "head_size_v", spec.head_size) != spec.head_size:
                raise KVCachedConfigError("Ascend K and V head dimensions must match")
            if getattr(instance, "use_sparse", False) or getattr(instance, "use_compress", False):
                raise KVCachedConfigError("Ascend compressed/sparse KV is not yet supported")
            instance.hybrid_with_attn_and_mamba = False
            return generic_allocate(instance, config)

        def reshape(instance, config, raw_tensors, *args, **kwargs):
            views = generic_reshape(instance, config, raw_tensors, *args, **kwargs)
            # Ascend's attention ABI takes separate K/V views. Keep its native
            # block table and kernel implementation; no storage copy occurs.
            return {name: (tensor[0], tensor[1]) for name, tensor in views.items()}

        runner._allocate_kv_cache_from_kvcached = allocate
        runner._reshape_kv_cache_tensors_from_kvcached = reshape
        self.patch_allocation_methods(runner)
        self.patch_reshape_methods(runner)
        self._mark_as_patched(runner, "__kvcached_ascend__")
        return True


class NPUWorkerPatch(GPUWorkerPatch):
    target_module = "vllm_ascend.worker.worker"
    target_class = "NPUWorker"
    patch_name = "npu_worker"

    def apply(self, module):
        worker = self._get_target_class(module)
        if worker is None:
            return False
        if self._is_already_patched(worker.init_device, "__kvcached_ascend__"):
            return True
        original_init = worker.init_device

        @wraps(original_init)
        def initialize(instance, *args, **kwargs):
            if enable_kvcached():
                validate_ascend_config(instance.vllm_config)
            return original_init(instance, *args, **kwargs)

        self._mark_as_patched(initialize, "__kvcached_ascend__")
        worker.init_device = initialize
        # Keep Ascend's profile/graph/communication setup intact. Explicit KV
        # bytes selects its native profile path without device-wide subtraction.
        return self.patch_worker_shutdown(module)
