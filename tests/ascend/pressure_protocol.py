# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""CPU-only contract and analysis for the three-arm Ascend experiment."""
import hashlib
import json
import math
import time
from pathlib import Path

ARMS = ("native", "held", "reclaim")
MIB = 1024 * 1024


def check_lease(path, node, device, now=None):
    """An operator reservation is required, not inferred from an idle snapshot.

    This validates a reservation record; it cannot itself enforce exclusivity.
    The host supervisor must also detect foreign jobs and abort our experiment.
    """
    now = time.time() if now is None else now
    record = json.loads(Path(path).read_text())
    if (record.get("node") != node or record.get("device") != device
            or record.get("exclusive") is not True
            or not record.get("reservation_reference")
            or not record.get("starts_at", math.inf) <= now
            or record.get("expires_at", 0) - now < 120):
        raise RuntimeError("No matching, current exclusive reservation with 120 seconds remaining")
    return record


def percentile(values, quantile):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * quantile) - 1)]


def summarize(records, offered, slo):
    finished = [r for r in records if r.get("finished")]
    completed_within_slo = sum(
        r["ttft_seconds"] <= slo["ttft_seconds"]
        and r["tpot_seconds"] <= slo["tpot_seconds"]
        and r["latency_seconds"] <= slo["latency_seconds"]
        for r in finished)
    return {
        "offered": offered, "completed": len(finished),
        "completed_within_slo": completed_within_slo,
        "slo_attainment": completed_within_slo / offered if offered else None,
        "ttft_p50_seconds": percentile([r["ttft_seconds"] for r in finished], 0.5),
        "ttft_p95_seconds": percentile([r["ttft_seconds"] for r in finished], 0.95),
        "tpot_p95_seconds": percentile([r["tpot_seconds"] for r in finished], 0.95),
        "latency_p95_seconds": percentile([r["latency_seconds"] for r in finished], 0.95),
        "latency_semantics": "client-observed streamed outputs; includes queue/control delay",
        "warning": "latency percentiles exclude unfinished requests; denominator does not",
    }


def trace_hash(trace):
    return hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest()
