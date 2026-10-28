# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Bounded resize/streaming latency experiment on an otherwise idle Ascend.

Uses the installed package, two ordinary HTTP services and optional native-call
observation. This does not fill HBM, reserve a device, or measure production P99.
Run with the external ownership-aware supervisor; output includes all samples.
"""
import argparse
import concurrent.futures
import ctypes
import json
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--abort-file', type=Path, required=True)
p.add_argument('--rounds', type=int, default=3)
p.add_argument('--preload')
a = p.parse_args()
assert os.getenv('ASCEND_RT_VISIBLE_DEVICES')
import kvcached
from kvcached.control import get_instance_memory_limit, set_instance_memory_limit
from kvcached.utils import get_tp_socket_dir
assert '/site-packages/' in kvcached.__file__, kvcached.__file__

MiB = 1024**2
LIMIT = 560 * MiB
result = dict(status='running', package=kvcached.__file__, python=sys.executable,
              configuration=vars(a) | {'output':str(a.output), 'abort_file':str(a.abort_file)},
              services=[], idle=[], streaming=[], clock='CLOCK_MONOTONIC ns',
              caveats=['SSE arrival gaps include host/network scheduling',
                       'poll-based completion is an upper bound',
                       'native tracing can perturb timing; compare trace-off run'])
services = []

def guard():
    if a.abort_file.exists(): raise RuntimeError(a.abort_file.read_text())

def save():
    a.output.write_text(json.dumps(result, indent=2)+'\n')

def http(s, path, payload=None, timeout=120):
    guard()
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{s["port"]}{path}', data=data,
                                 headers={'Content-Type':'application/json'})
    return urllib.request.urlopen(req, timeout=timeout)

def start(label):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
    name='syd-lat-'+a.output.stem[-16:]+'-'+label
    log=a.output.with_suffix('.'+label+'.log')
    env=dict(os.environ,ENABLE_KVCACHED='true',KVCACHED_AUTOPATCH='true',
             KVCACHED_IPC_NAME=name,KVCACHED_ASCEND_FIA_MODE='causal-window')
    if a.preload:
        env['LD_PRELOAD']=a.preload
        env['KVCACHED_TIMING_PREFIX']=str(a.output.with_suffix('.'+label+'.vmm'))
    cmd=[sys.executable,'-m','vllm.entrypoints.cli.main','serve',a.model,
         '--host','127.0.0.1','--port',str(port),'--served-model-name','syd-model',
         '--enforce-eager','--dtype','bfloat16','--max-model-len','2048',
         '--max-num-seqs','8','--max-num-batched-tokens','512',
         '--enable-prefix-caching','--enable-chunked-prefill',
         '--kv-cache-memory-bytes',str(LIMIT),'--seed','0','--shutdown-timeout','60']
    stream=log.open('w'); proc=subprocess.Popen(cmd,env=env,stdout=stream,
                       stderr=subprocess.STDOUT,start_new_session=True)
    rec=dict(label=label,name=name,port=port,log=str(log),command=cmd)
    s=rec|{'process':proc,'stream':stream,'record':rec}
    services.append(s);result['services'].append(rec);save()
    until=time.monotonic()+420
    while time.monotonic()<until:
        guard()
        if proc.poll() is not None: raise RuntimeError(log.read_text()[-3000:])
        try:
            with http(s,'/health',timeout=2): pass
            rec['ready_control']=get_instance_memory_limit(name)
            assert rec['ready_control']['complete_membership']
            print('READY',label,flush=True);return s
        except (urllib.error.URLError,TimeoutError): time.sleep(1)
    raise TimeoutError('Startup '+label)

def stop(s):
    rec=s['record'];proc=s['process']
    if rec.get('stopped'): return
    if proc.poll() is None:proc.send_signal(signal.SIGTERM)
    try:proc.wait(80)
    except subprocess.TimeoutExpired:
        rec['frontend_timeout']=True;proc.kill();proc.wait(10)
    s['stream'].close();rec['exit_code']=proc.returncode
    until=time.monotonic()+30
    while True:
        snap=get_instance_memory_limit(s['name'])
        if not snap['complete_membership'] or time.monotonic()>until:break
        time.sleep(.2)
    rec['control_after_exit']=snap
    root=Path(get_tp_socket_dir(s['name']))
    rec['remaining_sockets']=[str(f) for f in root.rglob('*.sock')] if root.exists() else []
    text=Path(s['log']).read_text(errors='replace')
    rec['forced_worker_exit']='force killing' in text
    rec['stopped']=True;save();print('STOPPED',s['label'],proc.returncode,flush=True)

def query(s):return get_instance_memory_limit(s['name'])
def charge(snap):return sum(g['charged_bytes'] for g in snap['gpus'])

def limit(s, size):
    t=time.monotonic_ns()
    published=set_instance_memory_limit(size,instance_name=s['name'])
    accepted=time.monotonic_ns()
    assert published['accepted'],published
    rec=dict(start_ns=t,accepted_ns=accepted,target=size,published=published,polls=[])
    deadline=time.monotonic()+45
    while time.monotonic()<deadline:
        guard();t0=time.monotonic_ns();snap=query(s);t1=time.monotonic_ns()
        rec['polls'].append(dict(start_ns=t0,end_ns=t1,charged=charge(snap),
                                  converged=snap['converged'],status=snap['status']))
        if snap['converged']:
            rec.update(end_ns=t1,final=snap)
            return rec
        if snap['status'] in ('failed','unsupported','unavailable'):raise RuntimeError(snap)
        time.sleep(.005)
    raise TimeoutError(rec)

def generate(s, prompt, tokens=32, on_token=None):
    request=dict(model='syd-model',prompt=prompt,max_tokens=tokens,temperature=0,
                 ignore_eos=True,return_token_ids=True,stream=True,
                 stream_options={'include_usage':True})
    t0=time.monotonic_ns();ids=[];events=[];usage=None
    with http(s,'/v1/completions',request) as response:
        for line in response:
            guard()
            if not line.startswith(b'data: '):continue
            value=line[6:].strip()
            if value==b'[DONE]':break
            received=time.monotonic_ns();chunk=json.loads(value)
            if chunk.get('usage'):usage=chunk['usage']
            for choice in chunk.get('choices',[]):
                delta=choice.get('token_ids') or []
                if delta:
                    ids.extend(delta);events.append(dict(ns=received,count=len(delta),cumulative=len(ids)))
                    if on_token:on_token(len(ids),received)
    t1=time.monotonic_ns()
    assert len(ids)==tokens,(s['label'],len(ids),tokens)
    return dict(start_ns=t0,end_ns=t1,events=events,token_ids=ids,usage=usage)

def warm(s):return [generate(s,[41+i]*800) for i in range(4)]

# One stable observer context, initialized before either model. Its constant
# footprint is present in every arm. No continuous device polling during decode.
import torch
import torch_npu  # noqa: F401
torch.npu.set_device(0)
torch.npu.init()
runtime=ctypes.CDLL('libascendcl.so')
mem_info=runtime.aclrtGetMemInfo
mem_info.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
mem_info.restype=ctypes.c_int

def memory():
    samples=[]
    for _ in range(3):
        free,total=ctypes.c_size_t(),ctypes.c_size_t()
        t=time.monotonic_ns();rc=mem_info(1,ctypes.byref(free),ctypes.byref(total))
        assert rc==0,rc
        samples.append(dict(start_ns=t,end_ns=time.monotonic_ns(),free=free.value,total=total.value))
        time.sleep(.005)
    return samples

try:
    A=start('a');B=start('b')
    # Prime both token lengths before measured rounds to exclude cold setup.
    warm(A);generate(B,[91]*256,tokens=512)
    reference=generate(B,[91]*256,tokens=512)['token_ids']
    result['reference_b']=reference
    reference_active=generate(A,[88]*256,tokens=256)['token_ids']
    result['reference_active_a']=reference_active
    for i in range(a.rounds):
        # Measure two release sizes, preserving complete histories in output.
        for target in (336*MiB,112*MiB):
            limit(A,LIMIT);warm(A);before=query(A);time.sleep(.1)
            rec=dict(iteration=i,target=target,before=before,driver_before=memory())
            rec['shrink']=limit(A,target)
            rec['driver_after']=memory()
            rec['restore']=limit(A,LIMIT)
            rec['before_regrow']=query(A)
            rec['regrow_start_ns']=time.monotonic_ns();rec['return_requests']=warm(A)
            rec['regrow_end_ns']=time.monotonic_ns();rec['after_regrow']=query(A)
            assert charge(rec['shrink']['final'])<charge(before)
            assert charge(rec['after_regrow'])>charge(rec['before_regrow'])
            result['idle'].append(rec);save()
            print('IDLE',i,target,round((rec['shrink']['end_ns']-rec['shrink']['start_ns'])/1e6,2),flush=True)
        scenarios=['sham','shrink','regrow_sham','regrow','active_sham','active']
        random.Random(400+i).shuffle(scenarios)
        for scenario in scenarios:
            limit(A,LIMIT);warm(A)
            if scenario=='regrow':limit(A,112*MiB)
            trigger=threading.Event();active_ready=threading.Event()
            rec=dict(iteration=i,scenario=scenario,before=query(A))
            def progress(count,ns):
                if count>=32 and not trigger.is_set():
                    rec['trigger_ns']=ns;trigger.set()
            def active_progress(count,ns):
                if count>=32:active_ready.set()
            def change():
                if not trigger.wait(90):raise TimeoutError('B decode trigger')
                if scenario in ('active','active_sham'):
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as sub:
                        request=sub.submit(generate,A,[88]*256,256,active_progress)
                        if not active_ready.wait(90):raise TimeoutError('A decode trigger')
                        rec['resize']=limit(A,112*MiB if scenario=='active' else LIMIT)
                        rec['active_a']=request.result(90)
                        assert rec['active_a']['token_ids']==reference_active
                elif scenario in ('regrow','regrow_sham'):
                    rec['resize']=limit(A,LIMIT)
                    rec['regrow_start_ns']=time.monotonic_ns()
                    rec['return_requests']=warm(A)
                    rec['regrow_end_ns']=time.monotonic_ns()
                else:rec['resize']=limit(A,LIMIT if scenario=='sham' else 112*MiB)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                action=pool.submit(change)
                rec['stream_b']=generate(B,[91]*256,tokens=512,on_token=progress)
                action.result(120)
            assert rec['stream_b']['token_ids']==reference
            rec['after']=query(A)
            result['streaming'].append(rec);save()
            print('STREAM',i,scenario,'PASS',flush=True)
    result['status']='passed'
except BaseException as exc:
    result['status']='failed';result['error']=repr(exc)
    raise
finally:
    for service in reversed(services):stop(service)
    save()
