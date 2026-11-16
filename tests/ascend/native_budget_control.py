# SPDX-License-Identifier: Apache-2.0
"""Inspect/change the native capacity candidate on an owned test service.

Does not start a service or reserve a card. Use --action set only on the service
under test. Results include every rank; convergence refers to mapped KV bytes,
not receiver success or whole-device free memory.
"""

import argparse
import json
from urllib.request import ProxyHandler, Request, build_opener


def validate_status(reply):
    results = reply.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError("Expected one single-DP controller result")
    status = results[0]
    expected = status.get("expected_ranks", [])
    ranks = status.get("ranks", [])
    if (
        not expected
        or len(ranks) != len(expected)
        or {row.get("rank") for row in ranks} != set(expected)
        or len({row.get("rank") for row in ranks}) != len(ranks)
    ):
        raise ValueError("Incomplete or duplicate rank observations")
    if any(
        row.get("ok") is not True
        or row.get("fatal")
        or row.get("ticket") != status.get("observed_ticket")
        or row.get("budget", {}).get("revision") != status.get("revision")
        for row in ranks
    ):
        raise ValueError("A rank has failed or has stale budget state")
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--action", choices=("status", "set", "reconcile"), default="status")
    parser.add_argument("--limit-bytes", type=int)
    parser.add_argument("--prefix-max-blocks", type=int)
    parser.add_argument("--expected-revision", type=int)
    args = parser.parse_args()
    options = dict(action=args.action)
    values = (args.limit_bytes, args.prefix_max_blocks, args.expected_revision)
    if any(value is not None and value < 0 for value in values):
        parser.error("budgets and revisions must be nonnegative")
    if args.action == "set":
        if args.expected_revision is None:
            parser.error("set requires --expected-revision from a previous status")
        options.update(
            limit_bytes=args.limit_bytes,
            prefix_max_blocks=args.prefix_max_blocks,
            expected_revision=args.expected_revision,
        )
    elif any(value is not None for value in values):
        parser.error("budget options are only valid with --action set")
    request = Request(
        args.url.rstrip("/") + "/collective_rpc",
        data=json.dumps(
            dict(method="kvcached_native_budget", kwargs=options, timeout=120)
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with build_opener(ProxyHandler({})).open(request, timeout=130) as response:
        reply = json.load(response)
    # Emit the original observation even if validation fails, preserving evidence.
    print(json.dumps(reply, indent=2, ensure_ascii=False))
    validate_status(reply)


if __name__ == "__main__":
    main()
