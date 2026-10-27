# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Compare cold/warm/reset prefix paths with the same single request."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--elastic", action="store_true")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not os.getenv("ASCEND_RT_VISIBLE_DEVICES"):
        parser.error("Select an available NPU with ASCEND_RT_VISIBLE_DEVICES")
    if not Path(args.model).is_dir():
        parser.error("--model must name a downloaded local Qwen3-0.6B directory")
    os.environ["ENABLE_KVCACHED"] = str(args.elastic).lower()
    os.environ["KVCACHED_AUTOPATCH"] = str(args.elastic).lower()
    import torch_npu  # noqa: F401
    if args.elastic:
        import kvcached.integration.vllm.autopatch  # noqa: F401
    from vllm import LLM, SamplingParams
    engine = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
                 max_model_len=1024, max_num_seqs=4, max_num_batched_tokens=1024,
                 kv_cache_memory_bytes=512 * 1024 * 1024, gpu_memory_utilization=0.15,
                 enable_prefix_caching=True, enable_chunked_prefill=True,
                 disable_log_stats=False, seed=42)
    paragraph = "A memory manager maps virtual blocks to physical pages and preserves live data. " * 24
    prompt = f"Document 0. {paragraph}\nAnswer only OK."
    records = []
    try:
        for index in range(8):
            if index == 3:
                assert engine.reset_prefix_cache()
            if index == 6:
                assert engine.reset_prefix_cache()
                tokens = engine.get_tokenizer().encode(prompt)
                engine.generate([{"prompt_token_ids": tokens[:128]}],
                                SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
                                use_tqdm=False)
            result = engine.generate([prompt], SamplingParams(temperature=0, max_tokens=16,
                                     ignore_eos=True, logprobs=5), use_tqdm=False)[0]
            output = result.outputs[0]
            metrics = {m.name: getattr(m, "value", repr(m)) for m in engine.get_metrics()
                       if "prefix_cache" in m.name}
            record = {"round": index, "tokens": list(output.token_ids), "text": output.text,
                      "cached_tokens": result.num_cached_tokens,
                      "logprobs": [{str(k): v.logprob for k, v in lp.items()} for lp in output.logprobs],
                      "metrics": metrics}
            records.append(record)
            Path(args.output).write_text(json.dumps(records, indent=2) + "\n")
            print(index, record["tokens"], metrics, flush=True)
    finally:
        engine.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
