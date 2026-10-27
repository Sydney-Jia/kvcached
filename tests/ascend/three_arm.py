# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""One arm of a fixed-budget, paired-workload experiment.

Run native/held/reclaim with the same trace and fixed physical ballast. A logical
budget alone is never labelled physical pressure. Pressure requires a separate
exclusive reservation record and a host supervisor watching foreign NPU jobs.
"""
import argparse
import json
import multiprocessing
import os
import signal
import time
from pathlib import Path

from pressure_protocol import ARMS, MIB, check_lease, summarize, trace_hash
from pressure_worker import run_worker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--regime", choices=("ample", "pressure"), default="ample")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--abort-file", type=Path, required=True)
    parser.add_argument("--lease-file", type=Path)
    parser.add_argument("--node", required=True)
    parser.add_argument("--physical-device", type=int, required=True)
    parser.add_argument("--ballast-bytes", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if not os.getenv("ASCEND_RT_VISIBLE_DEVICES"):
        parser.error("Explicitly select a device")
    if args.regime == "pressure" and (not args.lease_file or args.ballast_bytes <= 0):
        parser.error("Pressure requires an exclusive reservation and fixed physical ballast")
    if args.regime == "ample" and args.ballast_bytes:
        parser.error("Ample control must not allocate pressure ballast")
    trace = json.loads(args.trace.read_text())
    assert len(trace["donor_prompts"]) == 4 and len(trace["burst_prompts"]) == 12
    assert all(len(p) == 800 for p in trace["donor_prompts"])
    assert all(len(p) == 640 for p in trace["burst_prompts"])
    result = {"arm": args.arm, "regime": args.regime, "trace_sha256": trace_hash(trace),
              "ballast_bytes": args.ballast_bytes, "rounds": [], "events": [],
              "status": "running", "model": str(args.model),
              "comparison": "static 448+448 MiB versus two elastic 896 MiB logical ceilings; pressure uses common fixed ballast; ample is an unconstrained negative control"}
    workers, connections = [], []
    latest_progress = {}
    context = multiprocessing.get_context("spawn")

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    def guard():
        if args.abort_file.exists():
            raise RuntimeError("Host supervisor detected foreign work or an expired reservation")
        if args.regime == "pressure":
            check_lease(args.lease_file, args.node, args.physical_device)

    def receive(index, timeout=900):
        deadline = time.monotonic() + timeout
        while True:
            guard()
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Engine {index} exceeded its phase deadline")
            if not connections[index].poll(0.2):
                if not workers[index].is_alive():
                    raise RuntimeError(f"Engine {index} exited")
                continue
            response = connections[index].recv()
            if "progress" in response:
                latest_progress[index] = response["progress"]
                continue
            if "error" in response:
                raise RuntimeError(response["error"])
            return response.get("result", response)

    def call(index, command, value=None, timeout=900):
        guard()
        connections[index].send((command, value))
        return receive(index, timeout)

    def generate(index, prompts, tokens, offered_at=None):
        return call(index, "generate", {"prompts": prompts, "output_tokens": tokens,
                                         "offered_at": offered_at}, timeout=120)

    save()
    def interrupted(signum, frame):
        raise RuntimeError(f"Experiment received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        guard()
        for index in range(2):
            parent, child = context.Pipe()
            process = context.Process(target=run_worker, args=(
                child, f"syd-three-arm-{os.getpid()}-{index}", str(args.model),
                args.arm, index, str(args.output.with_suffix(f".engine-{index}.log"))))
            process.start()
            child.close()
            connections.append(parent)
            workers.append(process)
            result["events"].append(receive(index))
            save()
        # Warm model workspace at the intended workload before allocating
        # ballast. B has prefix caching disabled in every arm.
        result["warmup"] = generate(1, trace["burst_prompts"], trace["output_tokens"])
        for iteration in range(args.rounds):
            if args.arm != "native":
                assert call(0, "limit", 896 * MIB)["accepted"]
            assert call(0, "reset")
            donor_cold = [generate(0, [prompt], 16)[0] for prompt in trace["donor_prompts"]]
            donor_warm = generate(0, [trace["donor_prompts"][-1]], 16)[0]
            before = [call(i, "snapshot") for i in range(2)]
            if args.arm != "native":
                donor_bytes = before[0]["memory"]["instance_charged_bytes"]
                if donor_bytes != 448 * MIB:
                    raise RuntimeError(f"Trace did not produce the planned donor footprint: {donor_bytes}")
            if iteration == 0 and args.regime == "pressure":
                guard()
                result["ballast_allocation"] = call(0, "ballast", args.ballast_bytes)
            round_result = {"iteration": iteration, "before": before,
                            "before_burst": [call(i, "snapshot") for i in range(2)],
                            "donor_cold": donor_cold, "donor_warm": donor_warm}
            result["rounds"].append(round_result)
            save()
            offered_at = time.monotonic()
            connections[1].send(("generate", {"prompts": trace["burst_prompts"],
                                               "output_tokens": trace["output_tokens"],
                                               "offered_at": offered_at,
                                               "phase": f"burst-{iteration}"}))
            if args.arm == "reclaim":
                minimum = before[0]["memory"]["pool_control"]["minimum_bytes"]
                round_result["control"] = call(0, "limit", minimum)
                assert round_result["control"]["accepted"]
                round_result["control_accept_seconds"] = time.monotonic() - offered_at
            round_result["requests"] = receive(1, timeout=120)
            round_result["summary"] = summarize(round_result["requests"], 12, trace["slo"])
            round_result["after_burst"] = [call(i, "snapshot") for i in range(2)]
            round_result["donor_return"] = generate(0, [trace["donor_prompts"][-1]], 16)[0]
            save()
        result["status"] = "completed"
    except BaseException as error:
        result["status"] = "failed_or_invalid"
        result["error"] = repr(error)
        result["last_generation_progress"] = latest_progress
        if result["rounds"]:
            current = result["rounds"][-1]
            partial = latest_progress.get(1, {})
            if ("requests" not in current
                    and partial.get("phase") == f"burst-{current['iteration']}"):
                current["partial_requests"] = partial["requests"]
                current["partial_summary"] = summarize(partial["requests"], 12, trace["slo"])
                current["partial_result_warning"] = "Incomplete/invalid run; not a valid performance comparison"
        raise
    finally:
        # Release pressure before process teardown, including when the host
        # reports another job. Do not route cleanup through the rejecting guard.
        if connections and workers[0].is_alive():
            try:
                connections[0].send(("release_ballast", None))
                if connections[0].poll(5):
                    result["ballast_cleanup"] = connections[0].recv()
            except (OSError, EOFError):
                pass
        for connection, process in zip(connections, workers):
            if process.is_alive():
                try:
                    connection.send(("stop", None))
                except (OSError, EOFError):
                    pass
            process.join(20)
            if process.is_alive() and os.getpgid(process.pid) == process.pid:
                os.killpg(process.pid, signal.SIGTERM)
                process.join(5)
                if process.is_alive():
                    os.killpg(process.pid, signal.SIGKILL)
                    process.join(5)
            connection.close()
        result["worker_exit_codes"] = [p.exitcode for p in workers]
        save()


if __name__ == "__main__":
    main()
