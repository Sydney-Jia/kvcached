# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for supervision after an engine changes process title."""
import importlib.util
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
