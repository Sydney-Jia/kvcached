# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for supervision after an engine changes process title."""
import importlib.util
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "process_identity", Path(__file__).parent / "ascend/process_identity.py")
identity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(identity)


def test_exact_environment_marker(tmp_path):
    (tmp_path / "environ").write_bytes(b"X=1\0KVCACHED_EXPERIMENT_TOKEN=private-run\0")
    assert identity.experiment_identity(tmp_path, "private-run", set()) == "environment"


@pytest.mark.parametrize("descriptor", ["1", "2"])
def test_inherited_log_survives_missing_environment_marker(tmp_path, descriptor):
    (tmp_path / "environ").write_bytes(b"VLLM::EngineCore\0OTHER=value\0")
    (tmp_path / "fd").mkdir()
    (tmp_path / "fd" / descriptor).symlink_to("/private/run-uuid.engine-0.log")
    assert identity.experiment_identity(
        tmp_path, "private-run", {"/private/run-uuid.engine-0.log"}) == "experiment_log"


@pytest.mark.parametrize("log_path", [
    "/private/another-run.engine-0.log", "/private/run-uuid.engine-0.log.other", "pipe:[123]",
])
def test_foreign_or_similarly_named_log_is_not_owned(tmp_path, log_path):
    (tmp_path / "environ").write_bytes(b"KVCACHED_EXPERIMENT_TOKEN=private-run-other\0")
    (tmp_path / "fd").mkdir()
    (tmp_path / "fd" / "1").symlink_to(log_path)
    assert identity.experiment_identity(
        tmp_path, "private-run", {"/private/run-uuid.engine-0.log"}) is None


def test_exited_process_is_not_owned(tmp_path):
    assert identity.experiment_identity(tmp_path / "absent", "private-run", set()) is None


def test_empty_token_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        identity.experiment_identity(tmp_path, "", set())


@pytest.mark.skipif(sys.platform != "linux" or not hasattr(os, "pidfd_open"),
                    reason="Real /proc and pidfd supervision requires Linux")
def test_renamed_process_cleanup_leaves_another_process_alive(tmp_path):
    pytest.importorskip("setproctitle")
    code = ("import setproctitle,time; "
            "setproctitle.setproctitle('VLLM::EngineCore'); "
            "print('ready',flush=True); time.sleep(60)")
    token = "cpu-supervisor-integration"
    owned_log, other_log = tmp_path / "owned.log", tmp_path / "other.log"
    children = []
    descriptors = []
    try:
        for path, owned in ((owned_log, True), (other_log, False)):
            environment = dict(os.environ)
            environment.pop("KVCACHED_EXPERIMENT_TOKEN", None)
            if owned:
                environment["KVCACHED_EXPERIMENT_TOKEN"] = token
            with path.open("w") as output:
                child = subprocess.Popen([sys.executable, "-c", code], env=environment,
                                         stdout=output, stderr=subprocess.STDOUT)
            children.append(child)
            descriptors.append(os.pidfd_open(child.pid))
        deadline = time.monotonic() + 10
        while not all("ready" in p.read_text() for p in (owned_log, other_log)):
            assert time.monotonic() < deadline, "CPU fixture did not become ready"
            assert all(p.poll() is None for p in children)
            time.sleep(0.02)
        owned_dir, other_dir = [Path(f"/proc/{p.pid}") for p in children]
        assert ("KVCACHED_EXPERIMENT_TOKEN=" + token).encode() not in (
            owned_dir / "environ").read_bytes().split(b"\0")
        assert identity.experiment_identity(owned_dir, token, {str(owned_log)}) == "experiment_log"
        assert identity.experiment_identity(other_dir, token, {str(owned_log)}) is None
        signal.pidfd_send_signal(descriptors[0], signal.SIGTERM)
        assert children[0].wait(timeout=5) == -signal.SIGTERM
        assert children[1].poll() is None
    finally:
        # Both children are owned test fixtures, and their pidfds cannot refer
        # to any later process that reuses the numerical PID.
        for child, descriptor in zip(children, descriptors):
            try:
                if child.poll() is None:
                    signal.pidfd_send_signal(descriptor, signal.SIGKILL)
                    child.wait(timeout=5)
            finally:
                os.close(descriptor)
