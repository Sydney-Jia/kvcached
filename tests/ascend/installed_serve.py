# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Acceptance through the installed vLLM CLI/HTTP APIs; no runtime monkeypatches.

Run under the host's idle-device/foreign-process supervisor. This is functional
validation with ample HBM, not a throughput benchmark or a physical-pressure test.
"""
import argparse
import concurrent.futures
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

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--abort-file', type=Path, required=True)
parser.add_argument('--cycles', type=int, default=2)
parser.add_argument('--second-model')
parser.add_argument('--shutdown-timeout', type=int, default=0)
args = parser.parse_args()
assert os.getenv('ASCEND_RT_VISIBLE_DEVICES'), 'Select a device explicitly'
import kvcached
from kvcached.control import get_instance_memory_limit, set_instance_memory_limit
from kvcached.utils import get_tp_socket_dir

assert '/site-packages/' in kvcached.__file__, kvcached.__file__
result = {'package': kvcached.__file__, 'python': sys.executable, 'model': args.model,
          'status': 'running', 'services': [], 'checks': [],
          'purpose': 'installed HTTP correctness and lifecycle, not performance'}
services = []


def save():
    args.output.write_text(json.dumps(result, indent=2) + '\n')


def guard():
    if args.abort_file.exists():
        raise RuntimeError(args.abort_file.read_text())


def http(port, path, payload=None, timeout=120):
    guard()
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=data,
                                 headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=timeout)


def generate(service, prompt, tokens=32, stream=False, cancel=False):
    payload = dict(model='syd-model', prompt=prompt, max_tokens=tokens,
                   temperature=0, ignore_eos=True, return_token_ids=True, stream=stream)
    start = time.monotonic()
    with http(service['port'], '/v1/completions', payload) as response:
        if stream:
            ids, first = [], None
            for line in response:
                if not line.startswith(b'data: '):
                    continue
                content = line[6:].strip()
                if content == b'[DONE]':
                    break
                chunk = json.loads(content)
                for choice in chunk.get('choices', []):
                    delta = choice.get('token_ids') or []
                    if delta and first is None:
                        first = time.monotonic() - start
                    ids.extend(delta)
                if cancel and ids:
                    break
            answer = {'token_ids': ids, 'first_token_seconds': first, 'cancelled': cancel}
        else:
            data = json.load(response)
            answer = {'token_ids': data['choices'][0].get('token_ids'),
                      'usage': data.get('usage'), 'response': data}
    answer['elapsed_seconds'] = time.monotonic() - start
    assert answer['token_ids'], answer
    if not cancel:
        assert len(answer['token_ids']) == tokens, answer
    return answer


def start_service(label, elastic, model=None, namespace=None):
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    name = 'syd-http-' + args.output.stem[-12:] + '-' + (namespace or label)
    logfile = args.output.with_suffix('.' + label + '.log')
    env = dict(os.environ, ENABLE_KVCACHED=str(elastic).lower(),
               KVCACHED_AUTOPATCH='true', KVCACHED_IPC_NAME=name,
               KVCACHED_ASCEND_FIA_MODE='causal-window')
    command = [sys.executable, '-m', 'vllm.entrypoints.cli.main', 'serve', model or args.model,
               '--host', '127.0.0.1', '--port', str(port), '--served-model-name', 'syd-model',
               '--enforce-eager', '--dtype', 'bfloat16', '--max-model-len', '2048',
               '--max-num-seqs', '8', '--max-num-batched-tokens', '512',
               '--enable-prefix-caching', '--enable-chunked-prefill',
               '--kv-cache-memory-bytes', str(896 * 1024**2), '--seed', '0',
               '--shutdown-timeout', str(args.shutdown_timeout)]
    record = {'label': label, 'elastic': elastic, 'name': name, 'port': port,
              'log': str(logfile), 'command': command}
    stream = logfile.open('w')
    process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                               start_new_session=True)
    service = dict(record, process=process, log_stream=stream, record=record)
    services.append(service)
    result['services'].append(record)
    save()
    deadline = time.monotonic() + 420
    while time.monotonic() < deadline:
        guard()
        if process.poll() is not None:
            raise RuntimeError(f'{label} exited {process.returncode}: {logfile.read_text()[-4000:]}')
        try:
            with http(port, '/health', timeout=2):
                record['ready'] = True
                if elastic:
                    record['control_ready'] = get_instance_memory_limit(name)
                    assert record['control_ready']['complete_membership']
                save()
                print('READY', label, flush=True)
                return service
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
    raise TimeoutError(f'{label} startup timed out')


def stop_service(service):
    p, record = service['process'], service['record']
    if record.get('stopped'):
        return
    if p.poll() is None:
        # This Popen handle refers only to our own API frontend. Its native
        # shutdown owns worker cleanup; the host supervisor catches leftovers.
        p.send_signal(signal.SIGTERM)
    try:
        p.wait(45)
    except subprocess.TimeoutExpired:
        record['frontend_timeout'] = True
        p.kill()
        p.wait(10)
    service['log_stream'].close()
    record['exit_code'] = p.returncode
    time.sleep(2)
    root = Path(get_tp_socket_dir(service['name']))
    record['remaining_worker_sockets'] = [str(f) for f in root.rglob('*.sock')] if root.exists() else []
    record['control_after_exit'] = get_instance_memory_limit(service['name']) if service['elastic'] else None
    record['exit_control_observations'] = []
    # The native abort path can return before killed NPU workers finish their
    # driver teardown. Observe lease retirement rather than assuming that an
    # API frontend exit instantly means its device resources have disappeared.
    deadline = time.monotonic() + 30
    while record['control_after_exit'] and record['control_after_exit']['complete_membership'] and time.monotonic() < deadline:
        record['exit_control_observations'].append({'time':time.time(),'snapshot':record['control_after_exit']})
        time.sleep(.5)
        record['control_after_exit'] = get_instance_memory_limit(service['name'])
    log = Path(record['log']).read_text(errors='replace')
    record['shutdown_warnings'] = [line for line in log.splitlines()
        if any(word in line for word in ('unconfirmed', 'force killing', 'leaked semaphore'))]
    record['stopped'] = True
    save()
    print('STOPPED', service['label'], record['exit_code'], record['remaining_worker_sockets'], flush=True)


def set_limit(service, amount, wait=True):
    published = set_instance_memory_limit(amount, instance_name=service['name'])
    assert published['accepted'], published
    if not wait:
        return published
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        guard()
        snapshot = get_instance_memory_limit(service['name'])
        if snapshot['converged']:
            assert all(s['charged_bytes'] <= amount for s in snapshot['gpus']), snapshot
            return snapshot
        if snapshot['status'] in ('failed', 'unavailable', 'unsupported'):
            raise RuntimeError(str(snapshot))
        time.sleep(.1)
    raise TimeoutError(str(snapshot))


def trace(service):
    # Same prefix construction and execution order in native and elastic runs.
    outputs = []
    for prompt, stream in [([41] * 800, False), ([41] * 800, True),
                           ([42] * 389, False), ([43] * 129, False)]:
        outputs.append(generate(service, prompt, stream=stream))
    return outputs


def batch(service):
    prompts = [[70+i]*1536 for i in range(8)]
    with http(service['port'], '/v1/completions', dict(model='syd-model',prompt=prompts,
              max_tokens=32,temperature=0,ignore_eos=True,return_token_ids=True)) as response:
        data = json.load(response)
    ids = [c['token_ids'] for c in sorted(data['choices'],key=lambda c:c['index'])]
    assert len(ids)==8 and all(len(row)==32 for row in ids)
    return ids


save()
try:
    baseline = start_service('native', False)
    result['native_trace'] = trace(baseline)
    result['native_cold_return'] = generate(baseline, [44] * 800)
    stop_service(baseline)
    for cycle in range(args.cycles):
        service = start_service(f'elastic{cycle}', True, namespace='elastic')
        actual = trace(service)
        assert [r['token_ids'] for r in actual] == [r['token_ids'] for r in result['native_trace']]
        checks = {'cycle': cycle, 'trace': actual, 'matched_native': True}
        checks['before_limit'] = get_instance_memory_limit(service['name'])
        checks['shrunk'] = set_limit(service, 112 * 1024**2)
        assert checks['shrunk']['gpus'][0]['charged_bytes'] < checks['before_limit']['gpus'][0]['charged_bytes']
        checks['restored'] = set_limit(service, 896 * 1024**2)
        # Shrunk pages invalidate cached prefixes. Compare a disjoint cold
        # prompt here; prefix hit/miss semantics have separate offline probes.
        checks['cold_return'] = generate(service, [44] * 800)
        assert checks['cold_return']['token_ids'] == result['native_cold_return']['token_ids']
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            pending = [pool.submit(generate, service, [55+i]*640, 128) for i in range(4)]
            time.sleep(.2)
            checks['busy_limit_accepted'] = set_limit(service, 112 * 1024**2, wait=False)
            checks['busy_outputs'] = [f.result(timeout=120) for f in pending]
        checks['busy_converged'] = set_limit(service, 112 * 1024**2)
        checks['after_busy_restore'] = set_limit(service, 896 * 1024**2)
        checks['cancelled_stream'] = generate(service, [67]*800, tokens=512, stream=True, cancel=True)
        checks['after_cancel'] = generate(service, [68]*129)
        with http(service['port'], '/metrics') as metrics:
            checks['after_cancel_metrics'] = [line for line in metrics.read().decode().splitlines()
                if line.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{'))]
        try:
            http(service['port'], '/v1/completions', {'model':'syd-model', 'prompt':[41]*3000, 'max_tokens':16})
            raise AssertionError('Invalid request was accepted')
        except urllib.error.HTTPError as error:
            checks['invalid_request_status'] = error.code
            assert error.code == 400
        checks['health_after_invalid'] = http(service['port'], '/health').status
        result['checks'].append(checks)
        save()
        stop_service(service)
    if args.second_model:
        native_b = start_service('native-b', False, model=args.second_model)
        result['native_b_trace'] = trace(native_b)
        result['native_b_batch'] = batch(native_b)
        result['native_b_return'] = generate(native_b,[45]*800)
        result['native_b_after_exit'] = generate(native_b,[46]*256)
        stop_service(native_b)
        a = start_service('dual-a', True)
        b = start_service('dual-b', True, model=args.second_model)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            fa, fb = pool.submit(trace, a), pool.submit(trace, b)
            actual_a, actual_b = fa.result(), fb.result()
        assert [x['token_ids'] for x in actual_a] == [x['token_ids'] for x in result['native_trace']]
        assert [x['token_ids'] for x in actual_b] == [x['token_ids'] for x in result['native_b_trace']]
        actual_b_batch = batch(b)
        assert actual_b_batch == result['native_b_batch']
        dual = {'model_b':args.second_model,'trace_a':actual_a,'trace_b':actual_b,
                'batch_b':actual_b_batch,
                'matched_independent_native':True,
                'before_a':get_instance_memory_limit(a['name']),
                'before_b':get_instance_memory_limit(b['name'])}
        dual['shrink_a'] = set_limit(a,112*1024**2)
        dual['shrink_b'] = set_limit(b,96*1024**2)
        assert dual['shrink_a']['gpus'][0]['charged_bytes'] < dual['before_a']['gpus'][0]['charged_bytes']
        assert dual['shrink_b']['gpus'][0]['charged_bytes'] < dual['before_b']['gpus'][0]['charged_bytes']
        dual['restore_a'] = set_limit(a,896*1024**2)
        dual['restore_b'] = set_limit(b,896*1024**2)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            fa=pool.submit(generate,a,[44]*800)
            fb=pool.submit(generate,b,[45]*800)
            dual['return_a'],dual['return_b']=fa.result(),fb.result()
        assert dual['return_a']['token_ids'] == result['native_cold_return']['token_ids']
        assert dual['return_b']['token_ids'] == result['native_b_return']['token_ids']
        result['dual']=dual
        save()
        stop_service(a)
        # B must remain usable while A exits and its backing is reclaimed.
        dual['b_after_a_exit']=generate(b,[46]*256)
        assert dual['b_after_a_exit']['token_ids'] == result['native_b_after_exit']['token_ids']
        stop_service(b)
    assert all(not s['remaining_worker_sockets'] and not s.get('frontend_timeout') for s in result['services'])
    assert all(s['control_after_exit'] is None or not s['control_after_exit']['complete_membership'] for s in result['services'])
    result['status'] = 'passed'
except BaseException as error:
    result['status'] = 'failed'
    result['error'] = repr(error)
    raise
finally:
    for service in services:
        stop_service(service)
    save()
    print('RESULT', args.output, result['status'], flush=True)
