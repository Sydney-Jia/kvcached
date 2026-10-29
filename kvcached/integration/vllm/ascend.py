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

from kvcached.integration.patch_base import BasePatch, enable_kvcached, is_integration_version_supported
from kvcached.integration.vllm.patches import (
    GPUModelRunnerPatch,
    GPUWorkerPatch,
    _infer_attention_type,
)
from kvcached.utils import KVCachedConfigError


def register_ascend_patches():
    """Patch native classes after Ascend completes their ordinary imports.

    Importing the runner eagerly from vLLM's root hook interrupts Ascend's
    device-operator initialization and can create a circular import.
    """
    from wrapt.importer import when_imported

    if not is_integration_version_supported("vllm", ">=0.23.0,<0.31.0"):
        raise KVCachedConfigError("Unsupported vLLM version for Ascend elastic KV")
    for patch in (NPUModelRunnerPatch(), NPUWorkerPatch(), NPUEngineCorePatch()):
        def apply_after_import(module, patch=patch):
            if not patch.apply(module):
                raise KVCachedConfigError(f"Could not apply Ascend patch {patch.patch_name}")

        when_imported(patch.target_module)(apply_after_import)


def validate_ascend_config(config):
    from kvcached.integration.vllm.ascend_compat import configured_mode

    configured_mode()
    if not config.model_config.enforce_eager:
        raise KVCachedConfigError("Ascend elastic KV currently requires --enforce-eager")
    if getattr(config.model_config, "enable_sleep_mode", False):
        raise KVCachedConfigError("Ascend elastic KV cannot also be owned by camem sleep mode")
    parallel = config.parallel_config
    parallel_axes = (
        "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
        "decode_context_parallel_size", "prefill_context_parallel_size",
    )
    if any(getattr(parallel, axis, 1) != 1 for axis in parallel_axes):
        raise KVCachedConfigError("Ascend elastic KV currently requires single-device execution")
    if getattr(config, "use_v2_model_runner", False):
        raise KVCachedConfigError("Ascend elastic KV currently requires the V1 model runner")
    if getattr(config, "kv_transfer_config", None) is not None:
        raise KVCachedConfigError("Ascend KV transfer connectors are not yet supported")
    if getattr(config, "speculative_config", None) is not None:
        raise KVCachedConfigError("Ascend speculative decoding is not yet supported")
    if config.cache_config.cache_dtype != "auto":
        raise KVCachedConfigError("Ascend elastic KV currently requires unquantized KV")
    if not getattr(config.cache_config, "kv_cache_memory_bytes", None):
        raise KVCachedConfigError(
            "Set --kv-cache-memory-bytes to the desired virtual KV capacity; "
            "physical usage will grow with live KV pages"
        )


class NPUEngineCorePatch(BasePatch):
    """Apply controls between synchronous batches even without a new KV block.

    Large hybrid blocks can cover an entire request. The generic idle watcher
    and allocation hooks alone then delay a live shrink until that request ends.
    The synchronous step has completed its previous future/output processing;
    the allocator's native unmap synchronization remains the device fence.
    Queued/async execution keeps its separate existing lifetime protocol.
    """
    library = "vllm"
    target_module = "vllm.v1.engine.core"
    target_class = "EngineCore"
    patch_name = "ascend_sync_control"

    def apply(self, module):
        core = self._get_target_class(module)
        if core is None or not callable(getattr(core, "step", None)):
            return False
        original = core.step
        if self._is_already_patched(original, self.patch_name):
            return True

        @wraps(original)
        def step(instance, *args, **kwargs):
            if (enable_kvcached() and getattr(instance, "batch_queue", None) is None
                    and not getattr(instance, "async_scheduling", False)):
                for manager in getattr(instance, "_kvcached_control_managers", ()):
                    manager.apply_resize_target()
                    manager.reclaim_for_shrink()
            return original(instance, *args, **kwargs)

        self._mark_as_patched(step, self.patch_name)
        core.step = step
        return True


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
        original_specs = runner.get_kv_cache_spec

        @wraps(original_specs)
        def cache_specs(instance, *args, **kwargs):
            specs = original_specs(instance, *args, **kwargs)
            if enable_kvcached() and instance.vllm_config.additional_config.get("kvcached_ascend_hybrid", False):
                from kvcached.integration.vllm.ascend_hybrid import normalize_hybrid_specs

                return normalize_hybrid_specs(specs)
            return specs

        runner.get_kv_cache_spec = cache_specs
        self.add_kvcache_allocator(runner)
        self.add_reshape_methods(runner)
        generic_allocate = runner._allocate_kv_cache_from_kvcached
        generic_reshape = runner._reshape_kv_cache_tensors_from_kvcached

        def allocate(instance, config):
            attention_type = _infer_attention_type(config)
            hybrid = (instance.vllm_config.additional_config.get("kvcached_ascend_hybrid", False)
                      and attention_type == "HYBRID_LINEAR")
            if not hybrid and (len(config.kv_cache_groups) != 1 or attention_type not in ("MHA", "GQA")):
                raise KVCachedConfigError("Ascend elastic KV currently supports one MHA/GQA group")
            if hybrid:
                from kvcached.integration.vllm.ascend_hybrid import install_state_bridge, validate_hybrid_model

                validate_hybrid_model(instance.vllm_config.model_config)
                if instance.vllm_config.cache_config.enable_prefix_caching:
                    raise KVCachedConfigError("Ascend hybrid experiment excludes prefix caching")
                for group in config.kv_cache_groups:
                    candidate = group.kv_cache_spec
                    if type(candidate).__name__ == "FullAttentionSpec":
                        continue
                    if (type(candidate).__name__ == "MambaSpec"
                            and candidate.mamba_cache_mode == "none"
                            and candidate.num_speculative_blocks == 0):
                        continue
                    raise KVCachedConfigError("Unsupported Ascend hybrid state lifetime")
                install_state_bridge()
            spec = (next(group.kv_cache_spec for group in config.kv_cache_groups
                         if type(group.kv_cache_spec).__name__ == "FullAttentionSpec")
                    if hybrid else config.kv_cache_groups[0].kv_cache_spec)
            if getattr(spec, "head_size_v", spec.head_size) != spec.head_size:
                raise KVCachedConfigError("Ascend K and V head dimensions must match")
            if getattr(instance, "use_sparse", False) or getattr(instance, "use_compress", False):
                raise KVCachedConfigError("Ascend compressed/sparse KV is not yet supported")
            instance.hybrid_with_attn_and_mamba = hybrid
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
            from kvcached.integration.vllm.ascend_compat import configure_worker, validate_worker

            validate_worker(instance.vllm_config)
            if enable_kvcached():
                validate_ascend_config(instance.vllm_config)
            result = original_init(instance, *args, **kwargs)
            configure_worker(instance.vllm_config)
            return result

        self._mark_as_patched(initialize, "__kvcached_ascend__")
        worker.init_device = initialize
        # Keep Ascend's profile/graph/communication setup intact. Explicit KV
        # bytes selects its native profile path without device-wide subtraction.
        return self.patch_worker_shutdown(module)
