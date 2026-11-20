# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Explicit single-DP native capacity executor; no global vLLM autopatch."""

from weakref import WeakValueDictionary

from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.executor.uniproc_executor import UniProcExecutor


_executors = WeakValueDictionary()

_BUDGET_UNHANDLED = object()


def executor_for(config):
    executor = _executors.get(id(config))
    if executor is None or executor.vllm_config is not config:
        raise ValueError("Capacity mode requires NativeCapacityExecutor in the same engine")
    return executor


def _validate_serialized_single_dp(vllm_config):
    parallel = vllm_config.parallel_config
    if (
        parallel.data_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
        or vllm_config.scheduler_config.async_scheduling
        or vllm_config.max_concurrent_batches != 1
    ):
        raise ValueError(
            "Native capacity admission currently requires synchronous single-DP TP execution"
        )


def _budget_rpc(admission, method, args, kwargs, non_block, unique_reply_rank, extra):
    """Serve the budget-control method shared by both executors."""
    if method != "kvcached_native_budget":
        return _BUDGET_UNHANDLED
    if admission is None:
        raise RuntimeError("Native budget scheduler is not attached")
    if args or non_block or unique_reply_rank is not None or extra:
        raise ValueError(
            "Budget control requires synchronous keyword arguments and all ranks"
        )
    options = dict(kwargs or {})
    action = options.pop("action", "status")
    if action == "status" and not options:
        return [admission.status()]
    if action == "reconcile" and not options:
        return [admission.reconcile()]
    if action == "set":
        return [admission.set_budget(**options)]
    raise ValueError(f"Unknown native budget action {action!r}")


class NativeCapacityExecutor(MultiprocExecutor):
    in_process = False

    def __init__(self, vllm_config, *args, **kwargs):
        _validate_serialized_single_dp(vllm_config)
        self.native_admission = None
        super().__init__(vllm_config, *args, **kwargs)
        _executors[id(vllm_config)] = self

    def collective_rpc(
        self,
        method,
        timeout=None,
        args=(),
        kwargs=None,
        non_block=False,
        unique_reply_rank=None,
        **extra,
    ):
        handled = _budget_rpc(self.native_admission, method, args, kwargs,
                              non_block, unique_reply_rank, extra)
        if handled is not _BUDGET_UNHANDLED:
            return handled
        return super().collective_rpc(
            method,
            timeout,
            args,
            kwargs,
            non_block=non_block,
            unique_reply_rank=unique_reply_rank,
            **extra,
        )


class NativeCapacityUniProcExecutor(UniProcExecutor):
    """Same admission protocol with the worker in-process (single rank).

    The zero-trust contract is unchanged: the scheduler still prepares every
    candidate and only then lets the allocator take slots. With one rank the
    worker lives in the engine process, so ``collective_rpc`` is a local call
    and the per-step IPC of ``MultiprocExecutor`` disappears.
    """

    world_size = 1
    in_process = True

    def __init__(self, vllm_config, *args, **kwargs):
        _validate_serialized_single_dp(vllm_config)
        if vllm_config.parallel_config.world_size != 1:
            raise ValueError(
                "NativeCapacityUniProcExecutor requires tensor-parallel size 1"
            )
        self.native_admission = None
        super().__init__(vllm_config, *args, **kwargs)
        _executors[id(vllm_config)] = self

    def collective_rpc(
        self,
        method,
        timeout=None,
        args=(),
        kwargs=None,
        non_block=False,
        single_value=False,
        unique_reply_rank=None,
        **extra,
    ):
        handled = _budget_rpc(self.native_admission, method, args, kwargs,
                              non_block, unique_reply_rank, extra)
        if handled is not _BUDGET_UNHANDLED:
            return handled
        if unique_reply_rank is not None or extra:
            raise ValueError(
                "In-process capacity executor serves all ranks in one call"
            )
        return super().collective_rpc(
            method,
            timeout=timeout,
            args=args,
            kwargs=kwargs,
            non_block=non_block,
            single_value=single_value,
        )
