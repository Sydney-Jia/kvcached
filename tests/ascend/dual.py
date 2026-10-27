# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Two live engines: reclaim A's cached backing, grow B, and re-run A."""
import argparse
import json
import multiprocessing
import os
import time
import traceback
from pathlib import Path


def engine_worker(connection, name, model, log):
    stream = open(log, "w", buffering=1)
    os.dup2(stream.fileno(), 1)
    os.dup2(stream.fileno(), 2)
    os.environ["KVCACHED_IPC_NAME"] = name
    os.environ["KVCACHED_MAX_CACHED_TOKENS"] = "4096"
    engine = None
    try:
        import torch_npu  # noqa: F401
        from vllm import LLM, SamplingParams

        import kvcached.integration.vllm.autopatch  # noqa: F401
        from kvcached.control import get_instance_memory_limit, set_instance_memory_limit

        engine = LLM(model=model, dtype="bfloat16", enforce_eager=True,
                     max_model_len=1024, max_num_seqs=4, max_num_batched_tokens=1024,
                     kv_cache_memory_bytes=512 * 1024 * 1024, gpu_memory_utilization=0.15,
                     enable_prefix_caching=True, enable_chunked_prefill=True,
                     disable_log_stats=False, seed=42,
                     worker_extension_cls="worker_probe.PhysicalBudgetProbe")

        def snapshot():
            metrics = {metric.name: getattr(metric, "value", repr(metric))
                       for metric in engine.get_metrics() if "prefix_cache" in metric.name}
            return {"physical": engine.collective_rpc("kvcached_physical_snapshot")[0],
                    "control": get_instance_memory_limit(name), "prefix_metrics": metrics}

        connection.send({"ready": name, **snapshot()})
        while True:
            command, arg = connection.recv()
            if command == "stop":
                break
            if command == "generate":
                params = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)
                results = engine.generate(arg, params, use_tqdm=False)
                connection.send({"tokens": [list(result.outputs[0].token_ids) for result in results],
                                 **snapshot()})
            elif command == "limit":
                connection.send(set_instance_memory_limit(arg, instance_name=name))
            elif command == "snapshot":
                connection.send(snapshot())
            else:
                raise ValueError(command)
    except BaseException:
        traceback.print_exc()
        connection.send({"error": traceback.format_exc()})
        raise
    finally:
        if engine is not None:
            engine.llm_engine.engine_core.shutdown()
        connection.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not os.getenv("ASCEND_RT_VISIBLE_DEVICES"):
        parser.error("Select an available NPU with ASCEND_RT_VISIBLE_DEVICES")
    if not Path(args.model).is_dir():
        parser.error("--model must name a downloaded local Qwen3-0.6B directory")
    os.environ["ENABLE_KVCACHED"] = "true"
    os.environ["KVCACHED_AUTOPATCH"] = "true"
    output = Path(args.output)
    context = multiprocessing.get_context("spawn")
    workers, connections, events = [], [], []

    def record(label, value):
        events.append({"label": label, "value": value})
        output.write_text(json.dumps(events, indent=2, default=str) + "\n")
        print(label, json.dumps(value, default=str)[:800], flush=True)
        return value

    def receive(index, timeout=900):
        assert connections[index].poll(timeout), f"engine {index} timed out"
        value = connections[index].recv()
        assert "error" not in value, value
        return value

    def call(index, command, arg=None):
        connections[index].send((command, arg))
        return receive(index)

    try:
        for index in range(2):
            parent, child = context.Pipe()
            process = context.Process(target=engine_worker, args=(
                child, f"ascend-check-{os.getpid()}-{index}", args.model,
                str(output.with_suffix(f".engine-{index}.log"))))
            process.start()
            child.close()
            workers.append(process)
            connections.append(parent)
            # Serialize initialization, while keeping the first engine alive.
            record(f"ready-{index}", receive(index))

        paragraph = "A memory manager maps virtual blocks to physical pages and preserves live data. " * 24
        prompts = [f"Document {index}. {paragraph}\nAnswer only OK." for index in range(4)]
        reference = record("A-reference", call(0, "generate", prompts[:1]))
        repeated = record("A-prefix-hit", call(0, "generate", prompts[:1]))
        assert repeated["tokens"] == reference["tokens"]
        primed = record("A-prime", call(0, "generate", prompts))
        sources = primed["control"]["gpus"]
        assert len(sources) == 1 and primed["control"]["complete_membership"], sources
        minimum = primed["physical"]["pool_control"]["minimum_bytes"]
        assert 0 < minimum < primed["physical"]["instance_charged_bytes"], primed
        assert record("A-limit", call(0, "limit", minimum))["accepted"]
        deadline = time.monotonic() + 60
        while True:
            shrunk = call(0, "snapshot")
            if shrunk["control"]["converged"]:
                break
            assert time.monotonic() < deadline, shrunk
            time.sleep(0.2)
        record("A-shrunk", shrunk)
        assert shrunk["physical"]["instance_charged_bytes"] <= minimum
        grown = record("B-grow", call(1, "generate", prompts))
        assert grown["physical"]["instance_charged_bytes"] > minimum
        recovered = record("A-recomputed", call(0, "generate", prompts[:1]))
        assert recovered["tokens"] == reference["tokens"]
        replayed = record("A-replay-all", call(0, "generate", prompts))
        assert replayed["tokens"] == primed["tokens"]
        for connection in connections:
            connection.send(("generate", prompts[:1]))
        concurrent = [record(f"concurrent-{index}", receive(index)) for index in range(2)]
        assert all(value["tokens"] == reference["tokens"] for value in concurrent)
        record("PASS", {"A_limit": minimum, "B_bytes": grown["physical"]["instance_charged_bytes"],
                        "same_tokens_after_reclaim": True})
    finally:
        for connection, process in zip(connections, workers):
            if process.is_alive():
                try:
                    connection.send(("stop", None))
                except (EOFError, BrokenPipeError):
                    pass
            process.join(30)
            if process.is_alive():
                process.terminate()
                process.join(15)
            connection.close()


if __name__ == "__main__":
    main()
