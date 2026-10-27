# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Verify timing and retained partial results without a device or vLLM."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "pressure_stream", Path(__file__).parent / "ascend/pressure_stream.py")
stream = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stream)


class FakeEngine:
    def __init__(self, frames):
        self.frames = list(frames)
        self.ids = []

    def add_request(self, request_id, prompt, params):
        self.ids.append(request_id)

    def has_unfinished_requests(self):
        return bool(self.frames)

    def step(self):
        frame = self.frames.pop(0)
        if isinstance(frame, Exception):
            raise frame
        return [SimpleNamespace(request_id=self.ids[index],
                                outputs=[SimpleNamespace(token_ids=tokens)],
                                finished=finished, num_cached_tokens=0)
                for index, tokens, finished in frame]


def test_per_request_latency_includes_offered_queue_delay():
    core = FakeEngine([[(0, [10], False), (1, [20], False)],
                       [(0, [10, 11], True)], [(1, [20, 21], True)]])
    clock = iter([11.0, 12.0, 14.0])
    result = stream.measure_requests(core, [[1], [2]], None, offered_at=10.0,
                                     clock=lambda: next(clock))
    assert [r["ttft_seconds"] for r in result] == [1.0, 1.0]
    assert [r["latency_seconds"] for r in result] == [2.0, 4.0]
    assert [r["tpot_seconds"] for r in result] == [1.0, 3.0]
    assert [r["token_ids"] for r in result] == [[10, 11], [20, 21]]


def test_failure_preserves_completed_and_pending_requests():
    core = FakeEngine([[(0, [10], True), (1, [20], False)], RuntimeError("engine failed")])
    snapshots = []
    with pytest.raises(RuntimeError, match="engine failed"):
        stream.measure_requests(core, [[1], [2]], None, offered_at=0,
                                progress=snapshots.append, clock=lambda: 1)
    assert len(snapshots[-1]) == 2
    assert [r["finished"] for r in snapshots[-1]] == [True, False]
    assert [r["token_ids"] for r in snapshots[-1]] == [[10], [20]]
    assert all(not r["finished"] for r in snapshots[0])


def test_request_with_no_output_cannot_be_reported_successful():
    core = FakeEngine([[(0, [], True)]])
    snapshots = []
    with pytest.raises(RuntimeError, match="no output token"):
        stream.measure_requests(core, [[1]], None, offered_at=0,
                                progress=snapshots.append, clock=lambda: 1)
    assert not snapshots[-1][0]["finished"]
