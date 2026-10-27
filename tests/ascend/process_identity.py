# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Identify private experiment processes even when their title rewrites environ.

Callers supply exact, run-unique log paths, never a shared directory prefix.
This is an ownership check, not a reservation or an isolation mechanism.
Signals must still use pidfds to avoid races with PID reuse.
"""
import os
from pathlib import Path


def experiment_identity(process_dir, token, log_paths):
    if not token:
        raise ValueError("An experiment token is required")
    process_dir = Path(process_dir)
    try:
        environment = (process_dir / "environ").read_bytes().split(b"\0")
        if ("KVCACHED_EXPERIMENT_TOKEN=" + token).encode() in environment:
            return "environment"
    except OSError:
        pass
    # A process-title library may overwrite the original environment memory
    # exposed by /proc even though Python's os.environ still has the marker.
    # Inherited stdout/stderr descriptors survive this and reparenting.
    for descriptor in ("1", "2"):
        try:
            if os.readlink(process_dir / "fd" / descriptor) in log_paths:
                return "experiment_log"
        except OSError:
            pass
    return None
