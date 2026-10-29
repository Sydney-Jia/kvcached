# SPDX-License-Identifier: Apache-2.0
"""Own one experimental engine; sample its CPU control ledger without an NPU context."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from urllib.request import ProxyHandler, build_opener


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    a=p.parse_args()
    cfg=json.loads(a.config.read_text())
    out=Path(cfg['output']); out.mkdir(parents=True, exist_ok=True)
    stop=Path(cfg['stop_file']); abort=Path(cfg['abort_file'])
    record={'started':time.time(),'config':cfg,'controller_pid':os.getpid()}
    from kvcached.control import get_instance_memory_limit
    def save():
        (out/'service.json').write_text(json.dumps(record,indent=2)+'\n')
    def interrupted(signum,frame):
        raise RuntimeError(f'Controller interrupted: {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    process=None
    with (out/'service.log').open('x') as log, (out/'ledger.jsonl').open('x') as ledger:
        try:
            process=subprocess.Popen(cfg['command'],env=dict(os.environ,**cfg['engine_env']),
                                     stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            record['pid']=process.pid
            record['start_ticks']=Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
            save()
            ready=False
            while time.time()-record['started']<cfg.get('max_seconds',3600):
                if stop.exists(): break
                if abort.exists(): raise RuntimeError(abort.read_text())
                if process.poll() is not None: raise RuntimeError(f'Engine exited {process.returncode}')
                snap=get_instance_memory_limit(cfg['ipc_name'])
                ledger.write(json.dumps({'time':time.time(),'control':snap})+'\n');ledger.flush()
                if not ready:
                    try:
                        with build_opener(ProxyHandler({})).open(cfg['api']+'/health',timeout=1): pass
                        ready=True; record['ready']=time.time(); save()
                    except OSError: pass
                    if time.time()-record['started']>600: raise TimeoutError('Engine startup deadline')
                time.sleep(.25)
            else: raise TimeoutError('Engine experiment deadline')
            record['passed']=True
        except Exception as exc:
            record.update(passed=False,error=repr(exc))
        finally:
            if process is not None:
                if process.poll() is None: process.send_signal(signal.SIGTERM)
                try: process.wait(timeout=145)
                except subprocess.TimeoutExpired:
                    record['frontend_timeout']=True
                    process.kill();process.wait(timeout=20)
                record['exit_code']=process.returncode
                deadline=time.time()+25
                while time.time()<deadline:
                    snap=get_instance_memory_limit(cfg['ipc_name'])
                    if not snap.get('complete_membership'): break
                    time.sleep(.25)
                record['control_after_exit']=snap
            record['finished']=time.time();save()
    return 0 if record.get('passed') and record.get('exit_code')==0 else 1


if __name__=='__main__':
    raise SystemExit(main())
