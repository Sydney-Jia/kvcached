# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Opt-in, slow diagnostic of the last FIA call; never a performance run."""
import json
import os
from pathlib import Path


def install(path):
    import torch
    import torch_npu

    original = torch_npu.npu_fused_infer_attention_score
    target = Path(path)
    sequence = 0

    def encode(name, value):
        if isinstance(value, torch.Tensor):
            result = {"shape": list(value.shape), "stride": list(value.stride()),
                      "dtype": str(value.dtype), "device": str(value.device),
                      "contiguous": value.is_contiguous()}
            if name in ("block_table", "actual_seq_lengths", "actual_seq_lengths_kv"):
                result["values"] = value.detach().cpu().tolist()
            return result
        if value is None or isinstance(value, (int, float, bool, str, list, tuple)):
            return value
        return repr(value)

    def traced(*args, **kwargs):
        nonlocal sequence
        sequence += 1
        record = {"pid": os.getpid(), "sequence": sequence, "state": "entered",
                  "args": [encode(str(i), v) for i, v in enumerate(args)],
                  "kwargs": {k: encode(k, v) for k, v in kwargs.items()}}
        target.write_text(json.dumps(record, indent=2) + "\n")
        result = original(*args, **kwargs)
        record["state"] = "returned"
        target.write_text(json.dumps(record, indent=2) + "\n")
        return result

    torch_npu.npu_fused_infer_attention_score = traced
    return {"diagnostic": str(target)}
