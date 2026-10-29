# SPDX-License-Identifier: Apache-2.0
"""Diagnostic startup variant for a driver SQ/CQ allocation race hypothesis.

This is not part of the cache algorithm. It serializes first creation of the
Torch NPU stream pool after native distributed initialization. A test that uses
it must disclose this startup variation; no device or driver setting changes.
"""

import fcntl
import os

import torch

from kvcached.integration.vllm.ascend_native_worker import NativeElasticWorker


class SerialStreamNativeWorker(NativeElasticWorker):
    def _init_device(self):
        device = super()._init_device()
        path = os.environ.get('KVCACHED_NATIVE_STREAM_INIT_LOCK')
        if not path:
            raise ValueError('Diagnostic startup requires an explicit shared lock file')
        # All distributed collectives above complete outside this lock.
        # Only local stream-pool materialization is serialized across workers.
        with open(path, 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            stream = torch.npu.Stream(device=device)
            stream.synchronize()
            fcntl.flock(lock, fcntl.LOCK_UN)
        print(f'KVCACHED_STREAM_INIT rank={self.rank} device={device}', flush=True)
        return device
