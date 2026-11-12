# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Manual native/held/elastic/restored smoke probe against owned services.

This changes the candidate's residency mode. It neither starts services nor
reserves devices. Run on idle, controlled services with matching model options.
It is not a pressure, quality, distributed atomicity or performance benchmark.
"""
import argparse
import json
import time
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-url", required=True)
    parser.add_argument("--candidate-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=96)
    args = parser.parse_args()
    if args.reference_url.rstrip("/") == args.candidate_url.rstrip("/"):
        parser.error("reference and candidate must be different services")
    if not 1 <= args.max_tokens <= 512:
        parser.error("max-tokens must be in [1, 512]")
    args.out.mkdir(parents=True, exist_ok=False)
    opener = build_opener(ProxyHandler({}))

    def save(name, value):
        (args.out / (name + ".json")).write_text(
            json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")

    def post(url, path, body):
        request = Request(url.rstrip("/") + path,
                          data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"})
        with opener.open(request, timeout=300) as response:
            return json.load(response)

    def control(action, label):
        reply = post(args.candidate_url, "/collective_rpc", {
            "method": "kvcached_native_control", "args": [action], "timeout": 120})
        save(label, reply)
        results = reply.get("results", [])
        if not results or not all(row.get("ok") is True
                                  and row.get("action") == action
                                  and row.get("poisoned") is False
                                  for row in results):
            raise RuntimeError(f"Control did not complete: {label}")
        return results

    prompts = ["Reply with exactly READY.",
               "What is 173 plus 289? Reply with the integer only.",
               "List three prime numbers greater than 20.",
               "Print integers from 1 through 40 separated by spaces."]

    def run(url, label):
        rows = []
        for prompt in prompts:
            start = time.monotonic()
            result = post(url, "/v1/chat/completions", {
                "model": args.model, "messages": [{"role": "user", "content": prompt}],
                "temperature": 0, "seed": 42, "max_tokens": args.max_tokens,
                "chat_template_kwargs": {"enable_thinking": False}})
            if not result.get("choices") or result.get("error"):
                raise RuntimeError(f"Incomplete response in {label}")
            rows.append({"prompt": prompt, "elapsed_s": time.monotonic() - start,
                         "response": result})
            save(label, rows)
        return rows

    def output(row):
        message = row["response"]["choices"][0]["message"]
        return (message.get("content"), message.get("reasoning_content"),
                message.get("reasoning"), message.get("tool_calls"))

    try:
        reference = run(args.reference_url, "native")
        observations = {"native-repeat": run(args.reference_url, "native-repeat")}
        control("status", "initial-control")
        for label, action in [("held", "held"), ("elastic", "elastic"),
                              ("restored", "held")]:
            control(action, label + "-control")
            observations[label] = run(args.candidate_url, label)
            control("status", label + "-after")
        checks = {label: [output(a) == output(b) for a, b in zip(reference, rows)]
                  for label, rows in observations.items()}
        save("comparison", checks)
        if not all(all(matches) for matches in checks.values()):
            raise RuntimeError("Output mismatch; inspect native repeat before attributing cause")
        save("summary", {"completed": True, "comparisons": checks,
                         "scope": "Serial smoke only; collect every DP/TP worker log separately"})
    except BaseException as error:
        save("failure", {"error": repr(error)})
        # Do not force full restoration on failure: another engine may own the
        # returned capacity. Preserve the observed state for the operator.
        raise


if __name__ == "__main__":
    main()
