# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Check reservation rejection and failure-inclusive benchmark accounting."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "pressure_protocol", Path(__file__).parent / "ascend/pressure_protocol.py")
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


def test_unfinished_requests_stay_in_slo_denominator():
    records = [{"finished": True, "ttft_seconds": 1, "tpot_seconds": 0.01,
                "latency_seconds": 2}, {"finished": False}]
    result = protocol.summarize(records, 4, {"ttft_seconds": 5,
                                           "tpot_seconds": 0.1, "latency_seconds": 30})
    assert result["completed_within_slo"] == 1
    assert result["slo_attainment"] == 0.25


@pytest.mark.parametrize("change", [
    {"exclusive": False}, {"node": "another-node"}, {"device": 3},
    {"reservation_reference": ""}, {"starts_at": 1001}, {"expires_at": 1119},
])
def test_pressure_rejects_missing_or_wrong_reservation(tmp_path, change):
    record = {"node": "test-node", "device": 5, "exclusive": True,
              "reservation_reference": "unit-test-only", "starts_at": 900, "expires_at": 1500}
    record.update(change)
    path = tmp_path / "lease.json"
    path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError):
        protocol.check_lease(path, "test-node", 5, now=1000)


def test_trace_fingerprint_changes_with_workload():
    assert protocol.trace_hash({"prompts": [1], "output_tokens": 64}) != protocol.trace_hash(
        {"prompts": [1], "output_tokens": 128})
