# SPDX-License-Identifier: Apache-2.0
"""Finite host supervisor for a pinned DeepSeek/Qwen co-location experiment.

Machine bindings are external JSON. Only the two exact, initially stopped,
dedicated containers are used. Foreign device work invalidates this run; it is
never signalled. All model/client commands and raw device samples are retained.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from urllib.request import ProxyHandler, build_opener


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    a=p.parse_args(); cfg=json.loads(a.config.read_text())
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    abort=out/'abort'; monitoring=threading.Event()
    state={'started':time.time(),'config':cfg,'phase':'preflight','completed':[]}
    qprocess=None; ds_started=False; clients=[]
    qid=cfg['qwen_container']; dsid=cfg['deepseek_container']

    def run(cmd,timeout=40):
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout)
        if r.returncode: raise RuntimeError(f'{cmd[:4]}: {r.stderr[-3000:]} {r.stdout[-2000:]}')
        return r.stdout
    def info(cid): return json.loads(run(['docker','inspect',cid]))[0]
    def save(): (out/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    def phase(name):
        state['phase']=name;state['phase_started']=time.time();save();print(name,flush=True)
    def guard():
        if abort.exists(): raise RuntimeError(abort.read_text())
    def sample():
        raw=run(['npu-smi','info'])
        if len(re.findall(r'\|\s*[0-7]\s+\|[^\n]*\bOK\b',raw)) !=8:
            raise RuntimeError('Not all eight devices healthy')
        rows=[]
        for device,pid,memory in re.findall(r'\|\s*(\d+)\s*\|\s*(\d+)\s*\|[^|]*\|\s*(\d+)\s*\|',raw):
            proc=Path('/proc')/pid
            try:
                group=(proc/'cgroup').read_text()
                owner='qwen' if qid in group else 'deepseek' if dsid in group else 'foreign'
                rows.append(dict(device=int(device),pid=int(pid),memory_mib=int(memory),owner=owner,
                                 start_ticks=(proc/'stat').read_text().rsplit(')',1)[1].split()[19]))
            except FileNotFoundError: pass
        return dict(time=time.time(),phase=state['phase'],raw=raw,processes=rows)
    def monitor():
        with (out/'devices.jsonl').open('x') as stream:
            while not monitoring.is_set():
                try:
                    s=sample();stream.write(json.dumps(s)+'\n');stream.flush()
                    if any(r['owner']=='foreign' for r in s['processes']):
                        raise RuntimeError('Foreign NPU process observed; stop only our experiment')
                except Exception as exc:
                    abort.write_text(repr(exc));return
                monitoring.wait(2)
    def wait_ready(api,deadline,namespace=None):
        while time.time()<deadline:
            guard()
            if namespace:
                s=Path(namespace)/'service.json'
                if s.exists():
                    r=json.loads(s.read_text())
                    if r.get('ready'):return
                    if r.get('finished'):raise RuntimeError(f'Qwen failed: {r}')
            else:
                try:
                    with build_opener(ProxyHandler({})).open(api+'/health',timeout=2):return
                except OSError:pass
            time.sleep(2)
        raise TimeoutError('Readiness deadline: '+api)
    def qstart(label):
        nonlocal qprocess
        guard()
        d=info(qid)
        if d['State']['Running']:raise RuntimeError('Qwen container unexpectedly running')
        run(['docker','start',qid])
        host=out/label;host.mkdir()
        cont=cfg['qwen_output_root']+'/'+label
        qcfg=dict(cfg['qwen_service'],output=cont,stop_file=cont+'/stop',
                  abort_file=cfg['qwen_output_root']+'/abort')
        qcfg['ipc_name']='syd-colocate-'+out.name+'-'+label
        qcfg['engine_env']=dict(qcfg['engine_env'],KVCACHED_IPC_NAME=qcfg['ipc_name'])
        (host/'config.json').write_text(json.dumps(qcfg,indent=2))
        command=['docker','exec']
        for key,value in cfg['qwen_controller_env'].items():command+=['-e',key+'='+value]
        command += [qid,'bash','-c',
                    'ulimit -c 0; source /usr/local/Ascend/ascend-toolkit/set_env.sh; source /usr/local/Ascend/nnal/atb/set_env.sh; exec "$@"',
                    'syd-controller',cfg['qwen_python'],'-u',cfg['qwen_script'],
                    '--config',cont+'/config.json']
        (host/'command.json').write_text(json.dumps(command,indent=2))
        with (host/'controller.log').open('x') as log:
            qprocess=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        state['qwen_current']=label;save()
        wait_ready(qcfg['api'],time.time()+650,host)
    def qstop():
        nonlocal qprocess
        if qprocess is None:return
        label=state['qwen_current'];(out/label/'stop').touch()
        try:qprocess.wait(timeout=185)
        except subprocess.TimeoutExpired:raise RuntimeError('Qwen controller shutdown deadline; inspect exact own workers')
        deadline=time.time()+45
        while any(r['owner']=='qwen' for r in sample()['processes']):
            if time.time()>deadline:raise RuntimeError('Qwen NPU contexts remain; do not stop init')
            time.sleep(2)
        run(['docker','stop','--time','30',qid])
        state.setdefault('qwen_exits',[]).append(qprocess.returncode)
        qprocess=None;save()
    def dscommand(action):
        return [sys.executable,cfg['deepseek_helper'],action,'--container',dsid,
                '--host-output',cfg['deepseek_output'],'--container-output',cfg['deepseek_output_container']]
    def launch_client(label,which,background=False,start_at=0,rounds=3,warmups=1):
        guard()
        if which=='qwen':
            init=info(qid)['State']['Pid']
            command=['nsenter','-t',str(init),'-n',sys.executable]
        else:command=[sys.executable]
        command += [cfg['client_script'],'--api',cfg[which+'_api'],
                    '--model',cfg[which+'_model'],'--dp',str(1 if which=='qwen' else 4),
                    '--concurrency',*(['2'] if which=='qwen' else ['4','8']),
                    '--rounds',str(rounds),'--warmups',str(warmups),
                    '--start-at',str(start_at),'--abort-file',str(abort),
                    '--output',str(out/(label+'.json'))]
        if background:command+=['--stop-file',str(out/(label+'.stop'))]
        (out/(label+'.command.json')).write_text(json.dumps(command,indent=2))
        with (out/(label+'.log')).open('x') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        clients.append(process)
        return process
    def finish_client(process,label):
        deadline=time.time()+1800
        while process.poll() is None:
            guard()
            if time.time()>deadline:raise TimeoutError(label)
            time.sleep(1)
        if process.returncode:raise RuntimeError(f'{label} failed; inspect complete response JSON')
        state['completed'].append(label);save()
    def replay(label,which='deepseek',**kwargs):
        phase(label);process=launch_client(label,which,**kwargs);finish_client(process,label)

    try:
        for cid in (qid,dsid):
            d=info(cid)
            assert d['Id']==cid and len(cid)==64,'Exact container required'
            assert not d['State']['Running'],'Dedicated container must initially be stopped'
        initial=sample();(out/'initial-devices.json').write_text(json.dumps(initial,indent=2))
        assert not initial['processes'],'Devices not idle before experiment'
        thread=threading.Thread(target=monitor,daemon=True);thread.start()
        phase('qwen-solo-start');qstart('qwen-solo')
        replay('qwen-solo-reference','qwen',warmups=2)
        phase('qwen-solo-stop');qstop();guard()
        phase('deepseek-start')
        command=dscommand('start')+cfg['deepseek_start_args']
        (out/'deepseek-start-command.json').write_text(json.dumps(command,indent=2))
        ds_started=True
        state['deepseek_launch']=run(command);save()
        wait_ready(cfg['deepseek_api'],time.time()+1200)
        replay('A-deepseek-alone',warmups=2)
        phase('qwen-resident-start')
        qstart('qwen-resident')
        replay('qwen-resident-warmup','qwen',rounds=1)
        time.sleep(3)
        replay('B-qwen-idle-resident')
        phase('C-both-active')
        start_at=time.time()+4
        qp=launch_client('C-qwen-active','qwen',background=True,start_at=start_at)
        dp=launch_client('C-deepseek-active','deepseek',start_at=start_at)
        finish_client(dp,'C-deepseek-active')
        (out/'C-qwen-active.stop').touch();finish_client(qp,'C-qwen-active')
        phase('qwen-resident-stop');qstop()
        replay('A2-deepseek-recovery',rounds=1)
        state['passed']=True
    except Exception as exc:
        state.update(passed=False,error=repr(exc));print(repr(exc),flush=True)
    finally:
        for process in clients:
            if process.poll() is None:process.terminate()
        for process in clients:
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
        try:qstop()
        except Exception as exc:state['qwen_cleanup_error']=repr(exc)
        if ds_started:
            try:state['deepseek_stop']=run(dscommand('stop'),timeout=160)
            except Exception as exc:state['deepseek_cleanup_error']=repr(exc)
        monitoring.set()
        if 'thread' in locals():thread.join(timeout=45)
        try:state['final_devices']=sample()
        except Exception as exc:state['final_sample_error']=repr(exc)
        state['finished']=time.time();save()
    return 0 if state.get('passed') and not any(k.endswith('_error') for k in state) else 1


if __name__=='__main__':
    raise SystemExit(main())
