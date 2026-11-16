# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Explicit single-DP native capacity executor; no global vLLM autopatch."""

from weakref import WeakValueDictionary

from vllm.v1.executor.multiproc_executor import MultiprocExecutor


_executors = WeakValueDictionary()


def executor_for(config):
    executor = _executors.get(id(config))
    if executor is None or executor.vllm_config is not config:
        raise ValueError("Capacity mode requires NativeCapacityExecutor in the same engine")
    return executor


class NativeCapacityExecutor(MultiprocExecutor):
    def __init__(self, vllm_config, *args, **kwargs):
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
        if method == "kvcached_native_budget":
            if self.native_admission is None:
                raise RuntimeError("Native budget scheduler is not attached")
            if args or non_block or unique_reply_rank is not None or extra:
                raise ValueError(
                    "Budget control requires synchronous keyword arguments and all ranks"
                )
            options = dict(kwargs or {})
            action = options.pop("action", "status")
            if action == "status" and not options:
                return [self.native_admission.status()]
            if action == "reconcile" and not options:
                return [self.native_admission.reconcile()]
            if action == "set":
                return [self.native_admission.set_budget(**options)]
            raise ValueError(f"Unknown native budget action {action!r}")
        return super().collective_rpc(
            method,
            timeout,
            args,
            kwargs,
            non_block=non_block,
            unique_reply_rank=unique_reply_rank,
            **extra,
        )
