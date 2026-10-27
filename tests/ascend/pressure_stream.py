# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Stream request measurements and preserve partial results on engine failure."""
import time


def measure_requests(core, prompts, params, offered_at=None, progress=None,
                     clock=time.monotonic):
    started = clock() if offered_at is None else offered_at
    generation_id = time.monotonic_ns()
    records = {
        f"{generation_id}-{i}": {"index": i, "first": None, "finished": False,
                                "token_ids": [], "cached_tokens": None}
        for i in range(len(prompts))
    }

    def snapshot():
        return [dict(record, token_ids=list(record["token_ids"]))
                for record in records.values()]

    last_publish = started
    last_completed = 0
    try:
        if progress:
            progress(snapshot())
        for request_id, prompt in zip(records, prompts):
            core.add_request(request_id, {"prompt_token_ids": prompt}, params)
        while core.has_unfinished_requests():
            outputs = core.step()
            arrived = clock()
            for output in outputs:
                record = records[output.request_id]
                if not output.outputs:
                    continue
                tokens = list(output.outputs[0].token_ids)
                if tokens and record["first"] is None:
                    record["first"] = arrived
                record["token_ids"] = tokens
                record["cached_tokens"] = output.num_cached_tokens
                if output.finished:
                    first = record["first"]
                    if first is None:
                        raise RuntimeError("Finished request has no output token")
                    record.update(finished=True, ttft_seconds=first - started,
                                  latency_seconds=arrived - started,
                                  tpot_seconds=(arrived - first) / max(1, len(tokens) - 1))
            completed = sum(record["finished"] for record in records.values())
            if progress and (completed != last_completed or arrived - last_publish >= 0.5):
                progress(snapshot())
                last_completed, last_publish = completed, arrived
        return snapshot()
    finally:
        # A worker exception must not erase requests already completed, nor
        # remove pending requests from the offered-load denominator.
        if progress:
            progress(snapshot())
