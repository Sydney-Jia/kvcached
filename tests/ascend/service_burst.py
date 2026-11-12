# SPDX-License-Identifier: Apache-2.0
"""Bounded open-loop text traffic against already running, owned services.

Ported from archive/ascend-service-burst-20261105. Deployment, device pressure
and allocator implementation are external. Native-residency controls do not
pretend to implement the legacy quota API. Preserve failed runs and raw streams.
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time
from urllib.request import ProxyHandler, Request, build_opener


def dump(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def http(service, path, body=None, timeout=120):
    request = Request(service['url'].rstrip('/') + path,
                      data=None if body is None else json.dumps(body).encode(),
                      headers={'Content-Type': 'application/json'})
    return build_opener(ProxyHandler({})).open(request, timeout=timeout)


def control(service, action):
    if service['control'] != 'native-residency':
        raise ValueError('This service has no native residency control')
    if action not in ('status', 'held', 'elastic'):
        raise ValueError('Only native status/held/elastic are supported')
    # A rejected mutation is not retried: some ranks may already have applied it.
    with http(service, '/collective_rpc', {
        'method': 'kvcached_native_control', 'args': [action], 'timeout': 120}, 130) as response:
        reply = json.load(response)
    rows = reply.get('results', [])
    if len(rows) != service['control_results'] or not all(
            row.get('ok') is True and row.get('action') == action
            and row.get('poisoned') is False for row in rows):
        raise RuntimeError('Native control rejected/incomplete: ' + json.dumps(reply))
    return reply


def arrivals(config):
    rows, offset = [], 0.0
    for phase in config['phases']:
        duration = phase['duration_s']
        if not isinstance(duration, (int, float)) or not 0 < duration <= 3600:
            raise ValueError('Phase duration must be finite and positive, at most 3600 s')
        for label, rate in phase['rates'].items():
            if label not in config['services'] or not math.isfinite(rate) or not 0 <= rate <= 1000:
                raise ValueError('Invalid service/rate')
            choices = phase['cases'][label]
            if not choices or any(key not in config['cases'] for key in choices):
                raise ValueError('Missing request case')
            for i in range(math.ceil(duration * rate)):
                rows.append({'at_s': offset + i / rate, 'service': label,
                             'phase': phase['name'], 'case': choices[i % len(choices)]})
                if len(rows) > 10000:
                    raise ValueError('Trace exceeds 10000 offered requests')
        offset += duration
        if offset > 3600:
            raise ValueError('Trace exceeds one hour')
    rows.sort(key=lambda row: (row['at_s'], row['service']))
    if not rows:
        raise ValueError('Trace offers no requests')
    for i, row in enumerate(rows):
        row['index'] = i
    return rows, offset


def validate(config, dynamic):
    if not config.get('services') or not config.get('cases'):
        raise ValueError('Services and cases are required')
    for service in config['services'].values():
        if not re.fullmatch(r'https?://[^\s/?#]+(?::\d+)?', service['url']):
            raise ValueError('Use an explicit HTTP(S) service origin')
        if service['control'] not in ('none', 'native-residency'):
            raise ValueError('Unknown control adapter')
        if service['control'] == 'native-residency' and (
                type(service.get('control_results')) is not int or service['control_results'] < 1):
            raise ValueError('Declare the number of results actually returned by the frontend')
    for case in config['cases'].values():
        if case.get('path', '/v1/chat/completions') not in ('/v1/chat/completions', '/v1/completions'):
            raise ValueError('Only text chat/completion endpoints are supported')
        if case['payload'].get('tools') or case['payload'].get('n', 1) != 1:
            raise ValueError('This text probe requires one completion without tool calls')
        if not 1 <= case['payload'].get('max_tokens', 0) <= 8192:
            raise ValueError('Each request needs a bounded max_tokens')
        if ('expected_text' in case) == ('expected_numbers' in case):
            raise ValueError('Each case requires exactly one explicit correctness oracle')
        if 'expected_numbers' in case and (not case['expected_numbers'] or any(
                type(value) is not int or value < 0 for value in case['expected_numbers'])):
            raise ValueError('Counting oracle requires nonnegative integers')
        if 'expected_text' in case and not isinstance(case['expected_text'], str):
            raise ValueError('Text oracle must be a string')
    offered, duration = arrivals(config)
    previous = -1
    for event in config.get('events', []):
        if (not math.isfinite(event['at_s']) or not 0 <= event['at_s'] < duration
                or event['at_s'] < previous
                or event['service'] not in config['services']
                or event['action'] not in ('held', 'elastic')):
            raise ValueError('Invalid or unordered residency event')
        previous = event['at_s']
        if dynamic and config['services'][event['service']]['control'] != 'native-residency':
            raise ValueError('Dynamic events require a native-residency service')
    if set(config.get('slo', {})) - {'ttft_s', 'latency_s', 'tpot_s'}:
        raise ValueError('Unknown SLO metric')
    for value in config.get('slo', {}).values():
        if not math.isfinite(value) or value <= 0:
            raise ValueError('SLO thresholds must be finite and positive')
    return offered


def request(service, case, offered, origin):
    begin = time.monotonic()
    row = dict(offered, sent_s=begin-origin, client_dispatch_lag_s=max(0, begin-origin-offered['at_s']),
               completed=False, correct=False, content='', reasoning='', token_ids=[], chunks=[])
    try:
        payload = dict(case['payload'], model=service['model'], stream=True,
                       stream_options={'include_usage': True})
        with http(service, case.get('path', '/v1/chat/completions'), payload) as response:
            done, finished = False, False
            for line in response:
                if time.monotonic() - begin > 240:
                    raise TimeoutError('Request exceeded 240 s')
                if not line.startswith(b'data: '):
                    continue
                raw = line[6:].strip()
                if raw == b'[DONE]':
                    done = True
                    break
                data = json.loads(raw)
                if data.get('error'):
                    raise RuntimeError(str(data['error']))
                row.setdefault('events', []).append(data)
                if data.get('usage'):
                    row['usage'] = data['usage']
                for choice in data.get('choices', []):
                    delta = choice.get('delta') or {}
                    if delta.get('tool_calls'):
                        raise ValueError('Unexpected tool call in text trace')
                    text = choice.get('text') or delta.get('content') or ''
                    thought = delta.get('reasoning_content') or delta.get('reasoning') or ''
                    ids = choice.get('token_ids') or []
                    if text or thought or ids:
                        row['chunks'].append({'at_s': time.monotonic()-origin, 'token_count': len(ids)})
                    row['content'] += text
                    row['reasoning'] += thought
                    row['token_ids'].extend(ids)
                    if choice.get('finish_reason'):
                        row['finish_reason'] = choice['finish_reason']
                        finished = True
            row['completed'] = done and finished
            if 'expected_text' in case:
                matches = row['content'] == case['expected_text']
            else:
                # An explanation, repeated answer or negative sign must not be
                # silently accepted by extracting only its digit substrings.
                matches = bool(re.fullmatch(r'[\d\s,;]+', row['content'])) and [
                    int(n) for n in re.findall(r'\d+', row['content'])] == case['expected_numbers']
            row['correct'] = row['completed'] and matches
    except Exception as error:
        row['error'] = repr(error)
    end = time.monotonic() - origin
    row.update(finished_s=end, latency_s=end-offered['at_s'])
    row['ttft_s'] = row['chunks'][0]['at_s']-offered['at_s'] if row['chunks'] else None
    row['completion_tokens'] = row.get('usage', {}).get('completion_tokens')
    row['tpot_s'] = ((end-row['chunks'][0]['at_s'])/(row['completion_tokens']-1)
                     if row['chunks'] and (row['completion_tokens'] or 0) > 1 else None)
    return row


def output_signature(row):
    return row['content'], row['reasoning'], row.get('token_ids', []), row.get('finish_reason')


def summarize(offered, cases, actions, events, slo, reference=None):
    by_id = {row['index']: row for row in cases}
    if len(by_id) != len(cases) or set(by_id) - {r['index'] for r in offered}:
        raise ValueError('Duplicate or unknown response identity')
    per_service = {}
    for item in offered:
        row = by_id.get(item['index'])
        stats = per_service.setdefault(item['service'], dict(offered=0, completed=0, correct=0,
                                                            slo_pass=0 if slo else None))
        stats['offered'] += 1
        if not row:
            continue
        if any(row[key] != item[key] for key in ('service', 'case', 'phase')):
            raise ValueError('Response attribution changed')
        stats['completed'] += bool(row['completed'])
        same = reference is None or (item['index'] in reference
                                     and output_signature(row) == output_signature(reference[item['index']]))
        row['matches_reference'] = same if reference is not None else None
        valid = row['correct'] and same
        stats['correct'] += bool(valid)
        if slo:
            stats['slo_pass'] += bool(valid and all(
                row.get(key) is not None and row[key] <= limit for key, limit in slo.items()))
    cohorts = {}
    for item in offered:
        key = item['service'] + '/' + item['phase']
        cohort = cohorts.setdefault(key, {'offered': 0, 'completed': 0, 'correct': 0, 'latencies': []})
        cohort['offered'] += 1
        row = by_id.get(item['index'])
        if row and row['completed']:
            cohort['completed'] += 1
            cohort['correct'] += bool(row['correct'] and row.get('matches_reference') is not False)
            cohort['latencies'].append(row)
    for cohort in cohorts.values():
        rows = cohort.pop('latencies')
        for metric in ('ttft_s', 'latency_s', 'tpot_s', 'client_dispatch_lag_s'):
            values = sorted(r[metric] for r in rows if r.get(metric) is not None)
            cohort[metric] = {f'p{p}': values[math.ceil(p/100*len(values))-1] if values else None
                              for p in (50, 95)}
    controls_ok = len(actions) == len(events) and all('error' not in action for action in actions)
    return dict(completed=controls_ok and len(cases) == len(offered)
                and all(s['correct'] == s['offered'] for s in per_service.values()),
                controls_ok=controls_ok, offered=len(offered), received=len(cases), services=per_service,
                missing_request_ids=sorted({r['index'] for r in offered} - by_id.keys()), cohorts=cohorts,
                latency_scope='Percentiles of completed requests; offered/completed counts retained. '
                              'Client average TPOT and response chunks, not per-token tail latency')


def run(config, out, abort_file, dynamic=False, workers=64, reference_dir=None):
    offered = validate(config, dynamic)
    if not 1 <= workers <= 128:
        raise ValueError('workers must be between 1 and 128')
    out.mkdir(parents=True, exist_ok=False)
    workload = {'cases': config['cases'], 'phases': config['phases'], 'slo': config.get('slo', {})}
    fingerprint = hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest()
    dump(out/'config.json', config)
    dump(out/'workload.json', {'sha256': fingerprint, 'workload': workload})
    dump(out/'arrivals.json', offered)
    reference = None
    if reference_dir:
        old = json.loads((reference_dir/'workload.json').read_text(encoding='utf-8'))
        if old['sha256'] != fingerprint:
            raise ValueError('Reference has a different offered workload')
        prior = json.loads((reference_dir/'summary.json').read_text(encoding='utf-8'))
        if not prior['completed']:
            raise ValueError('Reference run did not complete')
        reference = {r['index']: r for r in json.loads((reference_dir/'requests.json').read_text(encoding='utf-8'))}
    origin, stop = time.monotonic(), threading.Event()
    cases, actions, samples, futures = [], [], [], []

    def wait_until(at):
        while time.monotonic() < origin + at:
            if stop.wait(min(.1, origin + at - time.monotonic())) or abort_file.exists():
                return False
        return not stop.is_set() and not abort_file.exists()

    def adjust():
        for event in config.get('events', []):
            if not wait_until(event['at_s']):
                return
            row = dict(event, start_s=time.monotonic()-origin, enabled=dynamic)
            try:
                if dynamic:
                    row['result'] = control(config['services'][event['service']], event['action'])
            except Exception as error:
                row['error'] = repr(error)
            row['end_s'] = time.monotonic()-origin
            actions.append(row)
            dump(out/'actions.json', actions)

    def sample():
        while not stop.is_set():
            row = {'at_s': time.monotonic()-origin, 'services': {}}
            for label, service in config['services'].items():
                try:
                    with http(service, '/metrics', timeout=2) as response:
                        lines = response.read().decode().splitlines()
                    row['services'][label] = [line for line in lines if not line.startswith('#') and any(
                        key in line for key in ('num_requests_running', 'num_requests_waiting',
                                               'num_preemptions_total', 'kv_cache_usage_perc'))]
                except Exception as error:
                    row['services'][label] = {'error': repr(error)}
            samples.append(row)
            dump(out/'samples.json', samples)
            stop.wait(.5)

    controller = threading.Thread(target=adjust, daemon=True)
    monitor = threading.Thread(target=sample, daemon=True)
    controller.start()
    monitor.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            for item in offered:
                if not wait_until(item['at_s']):
                    raise RuntimeError('Trace aborted; unsent offered requests remain failures')
                futures.append(executor.submit(request, config['services'][item['service']],
                                               config['cases'][item['case']], item, origin))
            for future in concurrent.futures.as_completed(futures):
                cases.append(future.result())
                dump(out/'requests.json', cases)
        # Events may occur after the final request, but before the trace ends.
        remaining = max(e['at_s'] for e in config.get('events', []) or [{'at_s': 0}])
        controller.join(max(0, origin+remaining-time.monotonic()) + 140)
        if controller.is_alive():
            raise TimeoutError('Control thread did not complete')
    except BaseException as error:
        dump(out/'failure.json', {'error': repr(error)})
        raise
    finally:
        stop.set()
        controller.join(140)
        monitor.join(2*len(config['services'])+5)
        cases = [future.result() for future in futures if future.done() and not future.cancelled()]
        summary = summarize(offered, cases, actions, config.get('events', []), config.get('slo', {}), reference)
        telemetry_ok = bool(samples) and not any(isinstance(v, dict) and 'error' in v
                                                  for s in samples for v in s['services'].values())
        summary.update(dynamic=dynamic, workload_sha256=fingerprint, telemetry_complete=telemetry_ok,
                       control_scope='Only results returned by frontend; collect all DP worker logs separately')
        summary['completed'] &= telemetry_ok and not (out/'failure.json').exists()
        dump(out/'requests.json', sorted(cases, key=lambda r: r['index']))
        dump(out/'summary.json', summary)
    if not summary['completed']:
        raise RuntimeError('Incomplete or incorrect trace; evidence retained')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--abort-file', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--workers', type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.workers <= 128:
        parser.error('workers must be between 1 and 128')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    print(json.dumps(run(config, args.out, args.abort_file, args.dynamic, args.workers, args.reference)))


if __name__ == '__main__':
    main()
