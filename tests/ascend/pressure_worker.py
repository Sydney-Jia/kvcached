# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""One private engine for the three-arm experiment; no network service."""
import os
import traceback

from pressure_protocol import MIB
from pressure_stream import measure_requests


def run_worker(connection, name, model, arm, index, log):
    os.setsid()
    stream = open(log, "w", buffering=1)
    os.dup2(stream.fileno(), 1)
    os.dup2(stream.fileno(), 2)
    elastic = arm != "native"
    os.environ["ENABLE_KVCACHED"] = "true" if elastic else "false"
    os.environ["KVCACHED_AUTOPATCH"] = "true" if elastic else "false"
    os.environ["KVCACHED_IPC_NAME"] = name
    os.environ["KVCACHED_MAX_CACHED_TOKENS"] = "-1"
    engine = None
    try:
        import torch_npu  # noqa: F401
        if elastic:
            import kvcached.integration.vllm.autopatch  # noqa: F401
        from vllm import LLM, SamplingParams
        from vllm.platforms import current_platform
        from vllm.plugins import load_plugins_by_group

        from kvcached.control import get_instance_memory_limit, set_instance_memory_limit

        runtime_identity = {
            "platform": type(current_platform).__module__ + "." + type(current_platform).__name__,
            "plugins": {group: {name: function.__module__ + ":" + function.__name__
                                for name, function in load_plugins_by_group(group).items()}
                        for group in ("vllm.platform_plugins", "vllm.general_plugins")},
        }
        if (runtime_identity["platform"] != "vllm_ascend.platform.NPUPlatform"
                or any(not value.startswith("vllm_ascend:")
                       for group in runtime_identity["plugins"].values()
                       for value in group.values())):
            raise RuntimeError(f"Uncontrolled platform/plugin selection: {runtime_identity}")
        engine = LLM(
            model=model, dtype="bfloat16", enforce_eager=True, max_model_len=1024,
            max_num_seqs=12, max_num_batched_tokens=2048,
            kv_cache_memory_bytes=(896 if elastic else 448) * MIB,
            gpu_memory_utilization=0.10, enable_prefix_caching=(index == 0),
            enable_chunked_prefill=True, seed=42,
            worker_extension_cls="worker_probe.PhysicalBudgetProbe")

        def snapshot():
            if elastic:
                return {"memory": engine.collective_rpc("kvcached_physical_snapshot")[0],
                        "control": get_instance_memory_limit(name)}
            return {"memory": {"driver": engine.collective_rpc("driver_memory_snapshot")[0]}}

        def generate(prompts, output_tokens, offered_at=None, phase="generation"):
            params = SamplingParams(temperature=0, max_tokens=output_tokens, ignore_eos=True)
            return measure_requests(
                engine.llm_engine, prompts, params, offered_at,
                progress=lambda records: connection.send(
                    {"progress": {"phase": phase, "requests": records}}))

        connection.send({"ready": name, "pid": os.getpid(),
                         "runtime_identity": runtime_identity, **snapshot()})
        while True:
            command, value = connection.recv()
            if command == "stop":
                break
            if command == "snapshot":
                result = snapshot()
            elif command == "reset":
                result = engine.reset_prefix_cache()
            elif command == "limit":
                if not elastic:
                    raise RuntimeError("Static control has no runtime resize")
                result = set_instance_memory_limit(value, instance_name=name)
            elif command == "generate":
                result = generate(**value)
            elif command == "ballast":
                result = engine.collective_rpc("audit_allocate_ballast", args=(value,))[0]
            elif command == "release_ballast":
                result = engine.collective_rpc("audit_release_ballast")[0]
            else:
                raise ValueError(command)
            connection.send({"result": result})
    except BaseException:
        traceback.print_exc()
        try:
            connection.send({"error": traceback.format_exc()})
        except (BrokenPipeError, OSError):
            pass
        raise
    finally:
        if engine is not None:
            engine.llm_engine.engine_core.shutdown()
        connection.close()
