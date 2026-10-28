# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Manual installed-package coverage, with no runtime quota changes.

Compare native and elastic services under identical no-prefix-cache workloads.
The parent only observes the control ledger and AscendCL free memory. This is a
bounded correctness/physical-allocation test, not a production latency benchmark.
Use the external idle-device and foreign-process supervisor.
"""
import argparse
import concurrent.futures
import ctypes
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--dtype', choices=('bfloat16', 'float16'), required=True)
p.add_argument('--kv-mib', type=int, required=True)
p.add_argument('--cycles', type=int, default=4)
p.add_argument('--include-default-reserve', action='store_true')
p.add_argument('--output', type=Path, required=True)
p.add_argument('--abort-file', type=Path, required=True)
a = p.parse_args()
assert os.getenv('ASCEND_RT_VISIBLE_DEVICES')
import kvcached
from kvcached.control import get_instance_memory_limit
from kvcached.utils import get_tp_socket_dir

assert '/site-packages/' in kvcached.__file__, kvcached.__file__
MiB = 1024**2
result = dict(status='running', package=kvcached.__file__, python=sys.executable,
              config={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
              services=[], quota_changes=0, measurements=[],
              caveats=['No prefix cache: default-reserve arm tests allocator defaults only',
                       'Driver free memory also includes framework workspaces',
                       'Polling may miss a transient peak; no performance claims'])
services = []


def save():
    a.output.write_text(json.dumps(result, indent=2)+'\n')


def guard():
    if a.abort_file.exists():
        raise RuntimeError(a.abort_file.read_text())


def http(s, path, payload=None, timeout=240):
    guard()
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(f'http://127.0.0.1:{s["port"]}{path}', data=data,
                                     headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(request, timeout=timeout)


def start(label):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    name = 'syd-natural-'+a.output.stem[-20:]+'-'+label
    log = a.output.with_suffix('.'+label+'.log')
    env = dict(os.environ, ENABLE_KVCACHED=str(label != 'native').lower(),
               KVCACHED_AUTOPATCH='true', KVCACHED_IPC_NAME=name,
               KVCACHED_ASCEND_FIA_MODE='causal-window')
    # Unset inherited overrides to exercise the package's actual defaults.
    for key in ('KVCACHED_PAGE_PREALLOC_ENABLED', 'KVCACHED_MIN_RESERVED_PAGES',
                'KVCACHED_MAX_RESERVED_PAGES', 'KVCACHED_MAX_CACHED_TOKENS'):
        env.pop(key, None)
    if label == 'zero':
        env.update(KVCACHED_PAGE_PREALLOC_ENABLED='false',
                   KVCACHED_MIN_RESERVED_PAGES='0', KVCACHED_MAX_RESERVED_PAGES='0')
    command = [sys.executable, '-m', 'vllm.entrypoints.cli.main', 'serve', a.model,
               '--host', '127.0.0.1', '--port', str(port), '--served-model-name', 'syd-model',
               '--enforce-eager', '--dtype', a.dtype, '--max-model-len', '9216',
               '--max-num-seqs', '4', '--max-num-batched-tokens', '512',
               '--no-enable-prefix-caching', '--enable-chunked-prefill',
               '--kv-cache-memory-bytes', str(a.kv_mib*MiB), '--seed', '0',
               '--shutdown-timeout', '120']
    stream = log.open('w')
    process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                               start_new_session=True)
    rec = dict(label=label, name=name, port=port, log=str(log), command=command,
               reserve_overrides={key: value for key, value in env.items()
                                  if key.startswith('KVCACHED_') and 'RESERV' in key})
    s = rec | dict(process=process, stream=stream, record=rec)
    services.append(s)
    result['services'].append(rec)
    save()
    deadline = time.monotonic()+480
    while time.monotonic() < deadline:
        guard()
        if process.poll() is not None:
            raise RuntimeError(log.read_text(errors='replace')[-4000:])
        try:
            with http(s, '/health', timeout=2):
                pass
            rec['ready_control'] = query(s)
            print('READY', label, flush=True)
            return s
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
    raise TimeoutError('Startup '+label)


def query(s):
    if s['label'] == 'native':
        return None
    deadline = time.monotonic()+5
    while time.monotonic() < deadline:
        guard()
        snap = get_instance_memory_limit(s['name'])
        if snap['complete_membership']:
            return snap
        if s['process'].poll() is not None:
            raise RuntimeError('Service exited')
        time.sleep(.005)
    raise TimeoutError(snap)


def stop(s):
    rec, process = s['record'], s['process']
    if rec.get('stopped'):
        return
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
    try:
        process.wait(145)
    except subprocess.TimeoutExpired:
        rec['frontend_timeout'] = True
        process.kill()
        process.wait(10)
    s['stream'].close()
    rec['exit_code'] = process.returncode
    deadline = time.monotonic()+30
    while True:
        snap = get_instance_memory_limit(s['name'])
        if not snap['complete_membership'] or time.monotonic() > deadline:
            break
        time.sleep(.2)
    rec['control_after_exit'] = snap
    root = Path(get_tp_socket_dir(s['name']))
    rec['remaining_sockets'] = [str(f) for f in root.rglob('*.sock')] if root.exists() else []
    rec['forced_worker_exit'] = 'force killing' in Path(s['log']).read_text(errors='replace')
    rec['stopped'] = True
    save()
    print('STOPPED', s['label'], process.returncode, flush=True)


def generate(s, specs, tokens):
    # One batch payload preserves admission ordering across the reference and
    # elastic arms. All generated token IDs, not just text fragments, are saved.
    prompts = [[token]*length for token, length in specs]
    payload = dict(model='syd-model', prompt=prompts, max_tokens=tokens,
                   temperature=0, ignore_eos=True, return_token_ids=True)
    start_ns = time.monotonic_ns()
    with http(s, '/v1/completions', payload) as response:
        data = json.load(response)
    choices = sorted(data['choices'], key=lambda choice: choice['index'])
    ids = [choice['token_ids'] for choice in choices]
    assert len(ids) == len(specs) and all(len(row) == tokens for row in ids)
    return dict(start_ns=start_ns, end_ns=time.monotonic_ns(), token_ids=ids, usage=data.get('usage'))


# Constant observer context for every arm. Only this thread calls AscendCL.
import torch
import torch_npu  # noqa: F401
torch.npu.set_device(0)
torch.npu.init()
runtime = ctypes.CDLL('libascendcl.so')
mem_info = runtime.aclrtGetMemInfo
mem_info.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
mem_info.restype = ctypes.c_int


def sample(s):
    snap = query(s)
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    rc = mem_info(1, ctypes.byref(free), ctypes.byref(total))
    assert rc == 0, rc
    return dict(ns=time.monotonic_ns(), free_bytes=free.value, total_bytes=total.value,
                charged_bytes=sum(g['charged_bytes'] for g in snap['gpus']) if snap else None,
                control=snap)


def settle(s):
    # Keep the entire quiet window: a single lucky free-memory reading is not
    # evidence of a stable return. Do not force any trim or quota publication.
    samples = []
    for _ in range(12):
        guard()
        samples.append(sample(s))
        time.sleep(.1)
    return samples


def measure(s, name, specs, tokens):
    rec = dict(arm=s['label'], workload=name, specs=specs, tokens=tokens,
               before=settle(s), during=[])
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(generate, s, specs, tokens)
        while not future.done():
            guard()
            rec['during'].append(sample(s))
            time.sleep(.05)
        rec['response'] = future.result()
    rec['after'] = settle(s)
    result['measurements'].append(rec)
    save()
    peak = max((x['charged_bytes'] or 0 for x in rec['during']), default=0)
    print('CASE', s['label'], name, 'peak_MiB', peak//MiB,
          'after_MiB', (rec['after'][-1]['charged_bytes'] or 0)//MiB, flush=True)
    return rec


mixed = [(71, 8192), (72, 4096), (73, 1024), (74, 129)]
burst = [(81+i, 7168) for i in range(4)]
workloads = [('single-'+str(n), [(41+i, n)], 32)
             for i, n in enumerate((32, 129, 1024, 4096, 8192))]
for cycle in range(a.cycles):
    workloads.extend([(f'mixed-{cycle}', mixed, 64), (f'burst-{cycle}', burst, 64)])

save()
try:
    references = {}
    arms = ['native', 'zero']+(['default'] if a.include_default_reserve else [])
    for arm in arms:
        service = start(arm)
        # Warm the widest individual context and simultaneous burst in all arms.
        service['record']['warmup'] = [generate(service, [(91, 8192)], 16),
                                       generate(service, burst, 16)]
        service['record']['after_warmup'] = settle(service)
        for name, specs, tokens in workloads:
            record = measure(service, name, specs, tokens)
            if arm == 'native':
                references[name] = record['response']['token_ids']
            else:
                record['matched_native'] = record['response']['token_ids'] == references[name]
                record['mismatch_positions'] = [
                    [i for i, (x, y) in enumerate(zip(actual, expected)) if x != y]
                    for actual, expected in zip(record['response']['token_ids'], references[name])]
            save()
        stop(service)
    result['correctness_passed'] = all(r.get('matched_native', True) for r in result['measurements'])
    result['cleanup_passed'] = all(
        r.get('exit_code') == 0 and not r.get('frontend_timeout') and not r['forced_worker_exit']
        and not r['remaining_sockets'] and not r['control_after_exit']['complete_membership']
        for r in result['services'])
    result['status'] = 'passed' if result['correctness_passed'] and result['cleanup_passed'] else 'failed'
except BaseException as error:
    result.update(status='failed', error=repr(error))
    raise
finally:
    for service in services:
        try:
            stop(service)
        except BaseException as error:
            service['record']['cleanup_error'] = repr(error)
            result['status'] = 'failed'
    save()
print('RESULT', result['status'], str(a.output), flush=True)
raise SystemExit(0 if result['status'] == 'passed' else 1)
