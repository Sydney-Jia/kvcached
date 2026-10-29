# SPDX-License-Identifier: Apache-2.0
"""Run matched native/elastic DeepSeek experiments from an immutable launcher.

Machine paths stay in external local/runtime JSON files. This script neither
stops services nor changes installed packages. The dev RPC binds localhost.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launcher', type=Path, required=True)
    parser.add_argument('--local', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=('native', 'elastic'), required=True)
    parser.add_argument('--release', type=Path)
    parser.add_argument('--serialize-stream-init', action='store_true')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Use a fresh output directory')
    if args.mode == 'elastic' and args.release is None:
        parser.error('Elastic mode requires a fixed kvcached release')
    sys.path.insert(0, str(args.launcher))
    from runtime.checks import check, inspect_devices, runtime_environment, save_json
    from runtime.configuration import make_plan, read_json

    plan = make_plan(args.launcher, 'four-card-megamoe', read_json(args.local),
                     read_json(args.runtime), 'combined')
    command = plan['command']
    index = command.index('--speculative-config')
    del command[index:index + 2]
    command[command.index('--enable-prefix-caching')] = '--no-enable-prefix-caching'
    command[command.index('--host') + 1] = '127.0.0.1'
    # Fixed native byte budget keeps profiling noise out of the comparison.
    command.extend(['--kv-cache-memory-bytes', str(2 * 1024**3)])
    environment = plan['environment']
    environment.update(ENABLE_KVCACHED='0', KVCACHED_AUTOPATCH='0', VLLM_SERVER_DEV_MODE='1')
    if args.mode == 'elastic':
        assert args.release.is_dir()
        environment['PYTHONPATH'] = str(args.release) + ':' + environment['PYTHONPATH']
        worker = 'kvcached.integration.vllm.ascend_native_worker.NativeElasticWorker'
        if args.serialize_stream_init:
            worker = 'kvcached.integration.vllm.ascend_serial_stream_worker.SerialStreamNativeWorker'
            environment['KVCACHED_NATIVE_STREAM_INIT_LOCK'] = str(args.output_dir / 'stream-init.lock')
        command.extend(['--worker-cls', worker])
    plan['experiment'] = {'mode': args.mode, 'prefix_caching': False,
                          'speculation': False, 'kv_bytes': 2 * 1024**3,
                          'serialized_stream_init': args.serialize_stream_init,
                          'kvcached_release': str(args.release) if args.release else None}
    plan['api'] = 'http://127.0.0.1:' + str(plan['local']['roles']['combined']['port'])
    result = check(args.launcher, plan, full_weights=False, allow_shared=False)
    args.output_dir.mkdir(parents=True)
    env = runtime_environment(plan, args.output_dir)
    save_json(args.output_dir / 'plan.json', plan)
    save_json(args.output_dir / 'preflight.json', result)
    save_json(args.output_dir / 'environment.json', env)
    devices = plan['local']['roles']['combined']['devices']
    save_json(args.output_dir / 'device-prelaunch.json', inspect_devices(devices, False))
    print(json.dumps(plan['experiment']), flush=True)
    os.chdir(args.output_dir)
    os.execvpe(command[0], command, env)


if __name__ == '__main__':
    main()
