# SPDX-License-Identifier: Apache-2.0
"""Real executor override with a CPU base: verify the EngineCore RPC ABI."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from weakref import WeakValueDictionary


class BaseExecutor:
    def __init__(self, config, *args, **kwargs):
        self.vllm_config = config
        self.calls = []

    def collective_rpc(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return ["native"]


class UniProcBase(BaseExecutor):
    world_size = 1


namespace = dict(MultiprocExecutor=BaseExecutor, UniProcExecutor=UniProcBase,
                 _executors=WeakValueDictionary(),
                 WeakValueDictionary=WeakValueDictionary)
source = Path(__file__).parents[1] / "kvcached/integration/vllm/native_executor.py"
nodes = [
    node
    for node in ast.parse(source.read_text(encoding="utf-8")).body
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.Assign))
]
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)


class ExecutorTests(unittest.TestCase):
    def config(self):
        return NS(
            parallel_config=NS(
                data_parallel_size=1,
                pipeline_parallel_size=1,
                prefill_context_parallel_size=1,
                decode_context_parallel_size=1,
            ),
            scheduler_config=NS(async_scheduling=False),
            max_concurrent_batches=1,
        )

    def config_uni(self):
        config = self.config()
        config.parallel_config.world_size = 1
        return config

    def test_engine_core_positional_rpc_reaches_budget_controller(self):
        config = self.config()
        executor = namespace["NativeCapacityExecutor"](config)
        self.assertIs(namespace["executor_for"](config), executor)
        executor.native_admission = NS(
            status=lambda: {"revision": 3},
            reconcile=lambda: {"revision": 3},
            set_budget=lambda **kwargs: kwargs,
        )
        # Exact call form used by EngineCore.collective_rpc, not just our test
        # helper's keyword call form.
        self.assertEqual(
            executor.collective_rpc("kvcached_native_budget", None, (), None), [{"revision": 3}]
        )
        answer = executor.collective_rpc(
            "kvcached_native_budget",
            30,
            (),
            dict(action="set", limit_bytes=512, prefix_max_blocks=4, expected_revision=3),
        )
        self.assertEqual(answer[0]["expected_revision"], 3)
        self.assertFalse(executor.calls)

    def test_normal_nonblocking_execution_keeps_original_dispatch(self):
        executor = namespace["NativeCapacityExecutor"](self.config())
        self.assertEqual(
            executor.collective_rpc(
                "execute_model", args=("output",), non_block=True, unique_reply_rank=1
            ),
            ["native"],
        )
        args, kwargs = executor.calls[-1]
        self.assertEqual(args, ("execute_model", None, ("output",), None))
        self.assertTrue(kwargs["non_block"])
        self.assertEqual(kwargs["unique_reply_rank"], 1)

    def test_unsupported_execution_contracts_fail_before_starting_workers(self):
        for axis in (
            "data_parallel_size",
            "pipeline_parallel_size",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
        ):
            config = self.config()
            setattr(config.parallel_config, axis, 2)
            with self.subTest(axis=axis), self.assertRaises(ValueError):
                namespace["NativeCapacityExecutor"](config)
        config = self.config()
        config.scheduler_config.async_scheduling = True
        with self.assertRaises(ValueError):
            namespace["NativeCapacityExecutor"](config)


    def test_in_process_executor_forwards_single_value_and_rejects_rank_replies(self):
        config = self.config_uni()
        executor = namespace["NativeCapacityUniProcExecutor"](config)
        self.assertTrue(executor.in_process)
        # EngineCore passes single_value positionally; the override must not
        # swallow it into **extra or the engine dies on its first step.
        self.assertEqual(
            executor.collective_rpc("execute_model", timeout=5, args=("o",),
                                    single_value=True),
            ["native"])
        _, kwargs = executor.calls[-1]
        self.assertTrue(kwargs["single_value"])
        with self.assertRaises(ValueError):
            executor.collective_rpc("execute_model", unique_reply_rank=0)

    def test_in_process_executor_requires_single_rank(self):
        config = self.config_uni()
        config.parallel_config.world_size = 2
        with self.assertRaises(ValueError):
            namespace["NativeCapacityUniProcExecutor"](config)


if __name__ == "__main__":
    unittest.main()
