# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Opt-in timing probes for the native residency control path.

Enable with KVCACHED_NATIVE_PROFILE=1.  Probes are intentionally small and
only aggregate at 256-event boundaries so they do not perturb the decode path
when disabled.
"""

import json
import os
import threading
import time
from contextlib import contextmanager

_ENABLED = os.getenv("KVCACHED_NATIVE_PROFILE", "0").strip() == "1"
# Any single probe above this threshold is printed immediately, with a wall
# clock timestamp, instead of waiting for the 256-event aggregate.  A probe
# that never returns still needs the stall watchdog; this covers slow-but-OK.
_SLOW_S = float(os.getenv("KVCACHED_NATIVE_SLOW_S", "1.0") or 0)
_LOCK = threading.Lock()
_METRICS = {}


def enabled():
    return _ENABLED


def _emit(metric):
    values = metric["samples"]
    if not values:
        return
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    print("KVCACHED_PROFILE " + json.dumps({
        "name": metric["name"],
        "count": metric["count"],
        "mean_ms": round(metric["total_s"] * 1000 / metric["count"], 3),
        "p95_ms": round(p95 * 1000, 3),
        "max_ms": round(metric["max_s"] * 1000, 3),
        "total_s": round(metric["total_s"], 3),
        "bytes": metric["bytes"],
        "last_n": metric["last_n"],
    }, sort_keys=True), flush=True)


def record(name, seconds, nbytes=0, last_n=0):
    if not _ENABLED:
        return
    if _SLOW_S > 0 and seconds >= _SLOW_S:
        print("KVCACHED_SLOW " + json.dumps({
            "name": name,
            "seconds": round(seconds, 3),
            "ts": round(time.time(), 3),
            "thread": threading.current_thread().name,
            "nbytes": int(nbytes),
            "last_n": int(last_n),
        }, sort_keys=True), flush=True)
    with _LOCK:
        metric = _METRICS.get(name)
        if metric is None:
            metric = _METRICS[name] = {
                "name": name, "count": 0, "total_s": 0.0, "max_s": 0.0,
                "bytes": 0, "last_n": 0, "samples": [],
            }
        metric["count"] += 1
        metric["total_s"] += seconds
        metric["max_s"] = max(metric["max_s"], seconds)
        metric["bytes"] += int(nbytes)
        metric["last_n"] = int(last_n)
        samples = metric["samples"]
        samples.append(seconds)
        if len(samples) > 4096:
            del samples[:2048]
        if metric["count"] % 256 == 0:
            _emit(metric)


@contextmanager
def timer(name, nbytes=0):
    if not _ENABLED:
        yield
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        record(name, time.perf_counter() - started, nbytes=nbytes)


def report():
    if not _ENABLED:
        return
    with _LOCK:
        for metric in _METRICS.values():
            _emit(metric)
