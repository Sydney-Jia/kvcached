# SPDX-License-Identifier: Apache-2.0
"""Two independent HTTP services on one NPU, open-loop arrivals and quota events.

Use a dedicated container plus a host supervisor. All inputs and failures are
saved. Calibration is separate from the frozen trace; no silent rate tuning.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request

MiB = 1024**2


def dump(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def http(port, path, data=None, timeout=180):
    encoded = None if data is None else json.dumps(data).encode()
    request = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=encoded,
                                     headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(request, timeout=timeout)


def rpc(service, method, *args):
    with http(service['port'], '/collective_rpc',
              {'method': method, 'args': list(args), 'timeout': 180}, timeout=190) as response:
        return json.load(response)


def limits(service):
    from kvcached.control import get_instance_memory_limit
    return get_instance_memory_limit(service['name'])


def resize(service, mib, timeout=45):
    from kvcached.control import set_instance_memory_limit
    start = time.monotonic()
    published = set_instance_memory_limit(mib * MiB, instance_name=service['name'])
    if not published.get('accepted'):
        raise RuntimeError(str(published))
    observations = []
    while time.monotonic() - start < timeout:
        state = limits(service)
        observations.append({'elapsed_s': time.monotonic()-start, 'state': state})
        if state['converged']:
            return {'published': published, 'elapsed_s': time.monotonic()-start,
                    'observations': observations}
        if state['status'] in ('failed', 'unavailable', 'unsupported'):
            raise RuntimeError(str(state))
        time.sleep(.1)
    raise TimeoutError(str(observations[-1]))


def prompt_ids(tokenizer, target=5000, start_number=1):
    # Tokenization is deterministic. The complete rendered chat carries the
    # task at its end; repeated filler is intentionally not prefix-cached.
    filler = 'The archive contains background notes about rivers, forests and roads. '
    text = filler * (target // 12 + 10)
    messages = [{'role': 'user', 'content': text + f'\nIgnore the background. Count from {start_number} to {start_number+31}, in order. Output only the comma-separated numbers, no explanation.'}]
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                        enable_thinking=False, return_dict=False)
    if hasattr(ids, 'keys'):
        ids = ids['input_ids']
    if ids and isinstance(ids[0], list):
        assert len(ids) == 1
        ids = ids[0]
    ids = list(ids)
    if len(ids) > target:
        # Preserve both chat boundaries and the complete final instruction.
        ids = ids[:24] + ids[-(target-24):]
    assert 0 < len(ids) <= target and all(isinstance(token, int) for token in ids)
    return ids


def request(service, prompt, index, offered, phase, origin, expected_numbers=None):
    start = time.monotonic()
    record = dict(service=service['label'], index=index, phase=phase,
                  offered_s=offered-origin, sent_s=start-origin,
                  client_dispatch_lag_s=max(0, start-offered), completed=False,
                  token_ids=[], text='', chunks=[])
    try:
        payload = dict(model='syd-model', prompt=prompt, temperature=0, max_tokens=128,
                       stream=True, return_token_ids=True,
                       stream_options={'include_usage': True})
        with http(service['port'], '/v1/completions', payload, timeout=240) as response:
            done = False
            for line in response:
                if not line.startswith(b'data: '):
                    continue
                body = line[6:].strip()
                if body == b'[DONE]':
                    done = True
                    break
                data = json.loads(body)
                if data.get('usage'):
                    record['usage'] = data['usage']
                for choice in data.get('choices', []):
                    ids = choice.get('token_ids') or []
                    text = choice.get('text') or ''
                    if ids or text:
                        record['chunks'].append({'at_s': time.monotonic()-origin,
                                                 'token_count': len(ids)})
                        record['token_ids'].extend(ids)
                        record['text'] += text
                    if choice.get('finish_reason'):
                        record['finish_reason'] = choice['finish_reason']
            record['completed'] = done
    except Exception as error:
        record['error'] = repr(error)
    end = time.monotonic()
    record['finished_s'] = end-origin
    record['latency_s'] = end-offered
    record['ttft_s'] = (record['chunks'][0]['at_s'] - record['offered_s']) if record['chunks'] else None
    count = record.get('usage', {}).get('completion_tokens', len(record['token_ids']))
    record['completion_tokens'] = count
    record['tpot_s'] = ((end-origin-record['chunks'][0]['at_s']) / max(1, count-1)) if record['chunks'] else None
    numbers = [int(x) for x in re.findall(r'\d+', record['text'])]
    expected_numbers = list(range(1, 33)) if expected_numbers is None else expected_numbers
    record['expected_numbers'] = expected_numbers
    record['correct'] = record['completed'] and numbers == expected_numbers
    return record


def start(args):
    assert not (args.output/'services.json').exists(), 'Use a fresh output directory'
    records = []
    for i, label in enumerate(('A', 'B')):
        name = f'syd-burst-{args.output.name}-{label}'
        port = 18470+i
        env = dict(os.environ, ENABLE_KVCACHED=str(not args.native).lower(),
                   KVCACHED_AUTOPATCH='true', KVCACHED_IPC_NAME=name,
                   KVCACHED_ASCEND_FIA_MODE='causal-window', VLLM_SERVER_DEV_MODE='1',
                   KVCACHED_PAGE_PREALLOC_ENABLED='true', KVCACHED_MIN_RESERVED_PAGES='12',
                   KVCACHED_MAX_RESERVED_PAGES='28', KVCACHED_MAX_CACHED_TOKENS='0')
        log = args.output/f'{label}.log'
        cmd = [sys.executable, '-m', 'vllm.entrypoints.cli.main', 'serve', args.model,
               '--host', '127.0.0.1', '--port', str(port), '--served-model-name', 'syd-model',
               '--enforce-eager', '--dtype', 'bfloat16', '--max-model-len', '8192',
               '--gpu-memory-utilization', '0.40',
               '--max-num-seqs', '16', '--max-num-batched-tokens', '512',
               '--no-enable-prefix-caching', '--enable-chunked-prefill',
               '--kv-cache-memory-bytes', str((args.initial_mib if args.native else args.virtual_mib)*MiB),
               '--seed', '0', '--shutdown-timeout', '30',
               '--worker-extension-cls', 'service_burst_probe.ServiceBurstProbe']
        with log.open('w') as stream:
            proc = subprocess.Popen(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        service = dict(label=label, name=name, port=port, pid=proc.pid,
                       cmd=cmd, native=args.native, log=str(log),
                       process_start_ticks=Path(f'/proc/{proc.pid}/stat').read_text().split()[21])
        records.append(service)
        dump(args.output/'services.json', records)
        deadline = time.monotonic()+600
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(log.read_text(errors='replace')[-5000:])
            try:
                with http(port, '/health', timeout=2):
                    pass
                if not args.native:
                    service['initial_limit'] = resize(service, args.initial_mib)
                dump(args.output/'services.json', records)
                print('READY', label, flush=True)
                break
            except (OSError, TimeoutError):
                time.sleep(1)
        else:
            raise TimeoutError(label)


def calibrate(args, services, prompt):
    out = {'cases': [], 'batches': [], 'prompt_tokens': len(prompt)}
    for label in ('A', 'B'):
        service = next(s for s in services if s['label'] == label)
        if not service['native']:
            out.setdefault('controls', []).append(resize(service, args.initial_mib))
        for size in (1, 2, 4, 8):
            origin = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=size) as executor:
                cases = list(executor.map(lambda j: request(service,prompt,j,origin,'calibration',origin), range(size)))
            elapsed = time.monotonic()-origin
            out['cases'].extend(cases)
            out['batches'].append({'service':label, 'size':size, 'elapsed_s':elapsed,
                                   'correct':sum(c['correct'] for c in cases)})
            dump(args.output/f'calibration-{args.name}.json',out)
            print('CALIBRATE',out['batches'][-1],flush=True)
            if not all(c['correct'] for c in cases):
                raise RuntimeError('Calibration correctness failure')
        out.setdefault('memory',{})[label]=rpc(service,'audit_trim_framework')
    dump(args.output/f'calibration-{args.name}.json',out)


def run_trace(args, services, prompt):
    config = json.loads(args.trace.read_text())
    path = args.output/args.name
    path.mkdir(exist_ok=False)
    dump(path/'trace.json',config)
    (path/'prompt.json').write_text(json.dumps(prompt))
    native = services[0]['native']
    initial=[]
    for service in services:
        if not native:
            initial.append(resize(service,config['initial_mib']))
        initial.append(rpc(service,'audit_trim_framework'))
    # All arms use the exact same fixed physical ballast from calibration.
    ballast_bytes = config.get('ballast_bytes',0)
    if ballast_bytes:
        initial.append(rpc(services[0],'audit_allocate_ballast',ballast_bytes))
    dump(path/'initial.json',initial)
    origin = time.monotonic()
    samples, actions, cases = [], [], []
    stop = threading.Event()
    def sample():
        while not stop.is_set():
            snap={'at_s':time.monotonic()-origin,'engines':{}}
            for service in services:
                item={}
                try:
                    with http(service['port'],'/metrics',timeout=3) as response:
                        lines=response.read().decode().splitlines()
                    item['metrics']=[line for line in lines if not line.startswith('#') and any(
                        name in line for name in ('num_requests_running','num_requests_waiting','kv_cache_usage_perc','num_preemptions_total'))]
                    if not native:
                        item['control']=limits(service)
                except Exception as error:
                    item['error']=repr(error)
                snap['engines'][service['label']]=item
            samples.append(snap)
            dump(path/'samples.json',samples)
            stop.wait(.5)
    def adjust():
        for event in config['events']:
            if stop.wait(max(0,origin+event['at_s']-time.monotonic())):
                return
            row={'scheduled_s':event['at_s'],'start_s':time.monotonic()-origin,'enabled':args.dynamic}
            try:
                if args.dynamic:
                    donor=next(s for s in services if s['label']==event['donor'])
                    receiver=next(s for s in services if s['label']==event['receiver'])
                    row['before']={s['label']:rpc(s,'driver_memory_snapshot') for s in services}
                    row['donor']=resize(donor,event['donor_mib'])
                    row['after_donor']={s['label']:rpc(s,'driver_memory_snapshot') for s in services}
                    row['receiver']=resize(receiver,event['receiver_mib'])
                    row['after_receiver']={s['label']:rpc(s,'driver_memory_snapshot') for s in services}
                row['end_s']=time.monotonic()-origin
            except Exception as error:
                row['error']=repr(error)
            actions.append(row)
            dump(path/'actions.json',actions)
    arrivals=[]
    offset=0
    for phase in config['phases']:
        for label in ('A','B'):
            rate=phase['rates'][label]
            n=0
            while rate>0 and n/rate<phase['duration_s']:
                arrivals.append((offset+n/rate,label,phase['name']))
                n+=1
        offset+=phase['duration_s']
    arrivals.sort()
    dump(path/'arrivals.json',arrivals)
    monitor=threading.Thread(target=sample,daemon=True);monitor.start()
    controller=threading.Thread(target=adjust,daemon=True);controller.start()
    futures=[]
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
            for i,(at,label,phase) in enumerate(arrivals):
                if args.abort_file.exists():
                    raise RuntimeError(args.abort_file.read_text())
                delay=origin+at-time.monotonic()
                if delay>0:
                    time.sleep(delay)
                service=next(s for s in services if s['label']==label)
                future=executor.submit(request,service,prompt,i,origin+at,phase,origin)
                futures.append(future)
            for future in concurrent.futures.as_completed(futures):
                cases.append(future.result())
                dump(path/'requests.json',cases)
        result={'completed':True,'dynamic':args.dynamic,'native':native,
                'correct':sum(c['correct'] for c in cases),'offered':len(arrivals),
                'elapsed_s':time.monotonic()-origin,'trace_sha256':hashlib.sha256(args.trace.read_bytes()).hexdigest()}
        dump(path/'summary.json',result)
        print('TRACE_DONE',args.name,result,flush=True)
    finally:
        stop.set();controller.join(50);monitor.join(10)
        # Completed and failed requests are retained even if the controller fails.
        cases=[f.result() for f in futures if f.done()]
        dump(path/'requests.json',cases)
        if ballast_bytes:
            dump(path/'release-ballast.json',rpc(services[0],'audit_release_ballast'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=('start','calibrate','trace','stop','status','prepare','correctness'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--native',action='store_true')
    p.add_argument('--initial-mib',type=int,default=1792)
    p.add_argument('--virtual-mib',type=int,default=4096)
    p.add_argument('--prompt-tokens',type=int,default=5000)
    p.add_argument('--trace',type=Path)
    p.add_argument('--name',default='trace01')
    p.add_argument('--dynamic',action='store_true')
    p.add_argument('--abort-file',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.command=='start':
        start(args);return
    services=json.loads((args.output/'services.json').read_text())
    if args.command=='stop':
        for s in services:
            proc=Path(f'/proc/{s["pid"]}')
            if proc.exists():
                with (proc/'stat').open() as f:
                    assert f.read().split()[21]==s['process_start_ticks']
                os.kill(s['pid'],signal.SIGTERM)
        return
    if args.command=='status':
        print(json.dumps([{**s,'memory':rpc(s,'driver_memory_snapshot'),
                           'control':None if s['native'] else limits(s)} for s in services]));return
    if args.command=='prepare':
        states=[]
        for s in services:
            if not s['native']:
                resize(s,args.initial_mib)
            states.append({'label':s['label'],'trim':rpc(s,'audit_trim_framework'),
                           'control':None if s['native'] else limits(s)})
        dump(args.output/f'prepare-{args.name}.json',states)
        print(json.dumps(states));return
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model)
    prompt=prompt_ids(tokenizer,args.prompt_tokens)
    if args.command=='correctness':
        result={'cases':[],'controls':[],'purpose':'Different expected outputs across live engines and requests'}
        for cycle in range(4):
            for s in services:
                if not s['native']:
                    result['controls'].append(resize(s,args.initial_mib))
            donor=services[cycle%2]
            receiver=services[1-cycle%2]
            origin=time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures=[]
                for i,s in enumerate([receiver,receiver,receiver,donor]):
                    first=101+cycle*40+i*200
                    tokens=prompt_ids(tokenizer,args.prompt_tokens,first)
                    futures.append(executor.submit(request,s,tokens,i,origin,f'unique{cycle}',origin,list(range(first,first+32))))
                time.sleep(1)
                if not donor['native']:
                    result['controls'].append(resize(donor,1024))
                    result['controls'].append(resize(receiver,2560))
                result['cases'].extend(f.result() for f in futures)
            dump(args.output/f'correctness-{args.name}.json',result)
        result['passed']=all(c['correct'] for c in result['cases'])
        dump(args.output/f'correctness-{args.name}.json',result)
        print('DISTINCT_OUTPUTS',result['passed'],len(result['cases']))
        if not result['passed']:
            raise RuntimeError('Distinct output validation failed')
        return
    if args.command=='calibrate':
        calibrate(args,services,prompt)
    else:
        run_trace(args,services,prompt)


if __name__=='__main__':
    main()
