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
    """Validate explicit pressure authorization, never infer it from idleness.

    A reservation and a user's explicit instruction to pressure-test an idle
    device are distinct records. Neither enforces isolation. The latter must
    require the host supervisor to abort our run when foreign work appears.
    """
    now = time.time() if now is None else now
    record = json.loads(Path(path).read_text())
    idle_device_authorization = (
        record.get("authorization_kind") == "user_directed_idle_device"
        and record.get("pressure_authorized") is True
        and bool(record.get("authorization_reference"))
        and record.get("exclusive") is False
        and record.get("external_reservation") is False
        and record.get("foreign_process_policy") == "abort_own_run")
    reservation = (record.get("authorization_kind", "exclusive_reservation")
                   == "exclusive_reservation"
                   and record.get("exclusive") is True
                   and bool(record.get("reservation_reference")))
    if (record.get("node") != node or record.get("device") != device
            or not (idle_device_authorization or reservation)
            or not record.get("starts_at", math.inf) <= now
            or record.get("expires_at", 0) - now < 120):
        raise RuntimeError("No matching explicit pressure authorization with 120 seconds remaining")
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
