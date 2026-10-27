# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Isolated, bounded Qwen baseline/elastic comparison (no network service)."""
import argparse
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--elastic", action="store_true")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compare-baseline", type=Path)
    args = parser.parse_args()
    if not os.getenv("ASCEND_RT_VISIBLE_DEVICES"):
        parser.error("Select an available NPU with ASCEND_RT_VISIBLE_DEVICES")
    if not Path(args.model).is_dir():
        parser.error("--model must name a downloaded local Qwen3-0.6B directory")
    os.environ["ENABLE_KVCACHED"] = "true" if args.elastic else "false"
    os.environ["KVCACHED_AUTOPATCH"] = "true" if args.elastic else "false"
    # Register PrivateUse1 before backend selection.
    import torch_npu  # noqa: F401
    if args.elastic:
        import kvcached.integration.vllm.autopatch  # noqa: F401
    from vllm import LLM, SamplingParams

    started = time.monotonic()
    engine = LLM(
        model=args.model, dtype="bfloat16", enforce_eager=True,
        max_model_len=1024, max_num_seqs=4, max_num_batched_tokens=1024,
        kv_cache_memory_bytes=512 * 1024 * 1024, gpu_memory_utilization=0.15,
        enable_prefix_caching=True, enable_chunked_prefill=True,
        disable_log_stats=False, seed=42,
        worker_extension_cls="worker_probe.PhysicalBudgetProbe" if args.elastic else "",
    )
    initialized = time.monotonic()
    prompts = [
        ("A memory manager maps virtual blocks to physical pages and preserves live data. " * 24
         + "\nSummarize the paragraph in one sentence:"),
        "The capital of France is",
        "def add(a, b):\n    return",
        "A cache is useful because",
    ]
    params = SamplingParams(temperature=0, max_tokens=24, ignore_eos=True)
    lengths = [len(engine.get_tokenizer().encode(prompt)) for prompt in prompts]
    assert 256 <= lengths[0] < 900, lengths
    print("prompt token lengths", lengths, flush=True)
    records = []
    physical = [engine.collective_rpc("kvcached_physical_snapshot")] if args.elastic else []
    # Identical prompts on round two exercise cache reuse; later rounds vary
    # the prefix while keeping the first request reproducible.
    for iteration in range(4):
        batch = prompts if iteration < 2 else [prompts[0]] + [
            f"Round {iteration}. {prompt}" for prompt in prompts[1:]
        ]
        outputs = engine.generate(batch, params, use_tqdm=False)
        records.append([{"prompt": output.prompt,
                         "token_ids": list(output.outputs[0].token_ids),
                         "text": output.outputs[0].text}
                        for output in outputs])
        print("finished round", iteration, flush=True)
        if args.elastic:
            physical.append(engine.collective_rpc("kvcached_physical_snapshot"))
            print("physical budget", physical[-1], flush=True)
    result = {"elastic": args.elastic, "init_seconds": initialized - started,
              "generation_seconds": time.monotonic() - initialized,
              "prompt_token_lengths": lengths, "rounds": records,
              "physical_snapshots": physical}
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("RESULT", args.output, flush=True)
    engine.llm_engine.engine_core.shutdown()
    if args.compare_baseline:
        baseline = json.loads(args.compare_baseline.read_text())
        assert not baseline["elastic"], "Comparison input must use the unmodified engine"
        assert result["prompt_token_lengths"] == baseline["prompt_token_lengths"]
        assert result["rounds"] == baseline["rounds"], "Output differs from the native baseline"
        print("PASS: every prompt, output token and decoded response matches", flush=True)


if __name__ == "__main__":
    main()
