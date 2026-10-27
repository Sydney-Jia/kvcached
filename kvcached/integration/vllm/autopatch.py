# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import os
import types

from wrapt.importer import when_imported

from kvcached.integration.patch_base import (
    PatchManager,
    is_integration_version_supported,
    log_patch_results,
)
from kvcached.integration.vllm.model_runner_v2 import KVLayoutV2Patch, ModelRunnerV2Patch
from kvcached.integration.vllm.model_runner_v2_028 import (
    VLLM_028_MRV2_RANGE,
    GPUModelRunnerV2Patch,
)
from kvcached.integration.vllm.nixl_compat import NixlConnectorPatch
from kvcached.integration.vllm.patches import (
    VLLM_ALL_RANGE,
    VLLM_MRV2_RANGE,
    VLLM_V8_RANGE,
    VLLM_V9_PLUS_RANGE,
    CoreEngineProcManagerPatch,
    ElasticBlockPoolPatch,
    EngineCorePatch,
    GPUModelRunnerPatch,
    GPUWorkerPatch,
    HybridBlockSizeAlignPatch,
    KVCacheCoordinatorPatch,
    KVCacheManagerAllocateSlotsPatch,
    KVCacheManagerPatch,
    MambaPartialTailPatch,
    MPClientPatch,
    TritonAttentionPatch,
)
from kvcached.utils import get_kvcached_logger

logger = get_kvcached_logger()


def _env_enabled() -> bool:
    return os.getenv("KVCACHED_AUTOPATCH", "false").lower() in ("true", "1")


def _target_device_type() -> str:
    # Spawned workers can import vLLM before torch_npu registers torch.npu.
    # The selected vLLM platform is authoritative at this engine boundary.
    from vllm.platforms import current_platform

    return current_platform.device_type


@when_imported("vllm")
def _patch_vllm(_vllm: types.ModuleType) -> None:
    if not _env_enabled():
        logger.debug("Disabled by KVCACHED_AUTOPATCH")
        return

    if not is_integration_version_supported("vllm", VLLM_ALL_RANGE):
        return

    # Create patch manager and register version-specific vLLM patches
    patch_manager = PatchManager("vllm")

    patches = [
            (NixlConnectorPatch(), VLLM_ALL_RANGE),
            (ElasticBlockPoolPatch(), VLLM_ALL_RANGE),
            (EngineCorePatch(), VLLM_ALL_RANGE),
            (MPClientPatch(), VLLM_ALL_RANGE),
            # CoreEngineProcManager lives in vllm.v1.engine.utils from 0.10
            (CoreEngineProcManagerPatch(), ">=0.10.0"),
            (GPUModelRunnerPatch(), ">=0.8.4,<0.29.0"),
            (GPUModelRunnerV2Patch(), VLLM_028_MRV2_RANGE),
            (ModelRunnerV2Patch(), VLLM_MRV2_RANGE),
            (KVLayoutV2Patch(), VLLM_MRV2_RANGE),
            (GPUWorkerPatch(), VLLM_ALL_RANGE),
            (HybridBlockSizeAlignPatch(), VLLM_ALL_RANGE),
            (KVCacheCoordinatorPatch(), VLLM_V9_PLUS_RANGE),
            (KVCacheManagerPatch(), VLLM_V8_RANGE),
            (KVCacheManagerAllocateSlotsPatch(), VLLM_ALL_RANGE),
            (MambaPartialTailPatch(), ">=0.28.0,<0.31.0"),
            (TritonAttentionPatch(), ">=0.9.0,<0.29.0"),
        ]
    if _target_device_type() == "npu":
        from kvcached.integration.vllm.ascend import register_ascend_patches

        gpu_only = (
            GPUModelRunnerPatch, GPUModelRunnerV2Patch, ModelRunnerV2Patch,
            KVLayoutV2Patch, GPUWorkerPatch, TritonAttentionPatch, NixlConnectorPatch,
        )
        patches = [(patch, versions) for patch, versions in patches
                   if not isinstance(patch, gpu_only)]
        register_ascend_patches()
    patch_manager.register_patches_with_versions(patches)

    # Apply all patches
    results = patch_manager.apply_all_patches()

    # Log results
    log_patch_results("vllm", results)
