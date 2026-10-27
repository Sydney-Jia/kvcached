# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""One private engine for the three-arm experiment; no network service."""
import os
import time
import traceback

from pressure_protocol import MIB


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

        from kvcached.control import get_instance_memory_limit, set_instance_memory_limit

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

        def generate(prompts, output_tokens, offered_at=None):
            core = engine.llm_engine
            params = SamplingParams(temperature=0, max_tokens=output_tokens, ignore_eos=True)
            started = time.monotonic() if offered_at is None else offered_at
            records = {}
            for i, prompt in enumerate(prompts):
                request_id = f"{time.monotonic_ns()}-{i}"
                records[request_id] = {"index": i, "first": None, "finished": False,
                                       "token_ids": [], "cached_tokens": None}
                core.add_request(request_id, {"prompt_token_ids": prompt}, params)
            while core.has_unfinished_requests():
                outputs = core.step()
                arrived = time.monotonic()
                for output in outputs:
                    record = records[output.request_id]
                    tokens = list(output.outputs[0].token_ids)
                    if tokens and record["first"] is None:
                        record["first"] = arrived
                    record["token_ids"] = tokens
                    record["cached_tokens"] = output.num_cached_tokens
                    if output.finished:
                        count = len(tokens)
                        first = record["first"]
                        if first is None:
                            raise RuntimeError("Finished request has no output token")
                        record.update(finished=True, ttft_seconds=first - started,
                                      latency_seconds=arrived - started,
                                      tpot_seconds=(arrived - first) / max(1, count - 1))
            return sorted(records.values(), key=lambda r: r["index"])

        connection.send({"ready": name, "pid": os.getpid(), **snapshot()})
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
