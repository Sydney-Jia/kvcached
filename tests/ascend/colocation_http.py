# SPDX-License-Identifier: Apache-2.0
"""Bounded HTTP replay for separately supervised co-location experiments.

No framework/device imports. Stream timing is measured inside the client, not
around docker exec. Raw events and complete answers are retained. Chunk gaps
are not called token latency: speculative decoding can emit multiple tokens.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import threading
import time
from urllib.request import ProxyHandler, Request, build_opener

CASES = [
    ('sequence', '只输出一个 JSON 数组，包含从 0 到 15 的所有整数，按从小到大排列。不要解释或 Markdown。', list(range(16))),
    ('sorting', '将 [12,-3,7,0,19,4,-8,2,11,5] 升序排列，只输出 JSON 数组，不要解释或 Markdown。', [-8,-3,0,2,4,5,7,11,12,19]),
    ('arithmetic', '计算后只输出 JSON 对象。a 为 17×23 的计算结果，b 为 19+28，c 为 (81-9)÷8，这三个值必须为 JSON 数字；tag 为字符串 NPU-OK。不要算式、解释或 Markdown。', {'a':391,'b':47,'c':9,'tag':'NPU-OK'}),
    ('records', '输入 [{"id":"A","qty":3,"price":7},{"id":"B","qty":5,"price":4}]。按原顺序输出 JSON 数组，每项只保留 id 并新增 total=qty×price。不要解释或 Markdown。', [{'id':'A','total':21},{'id':'B','total':20}]),
]
PROMPT = ('Inference performance depends on compute, memory bandwidth, batching and communication. ' * 32
          + 'Explain these tradeoffs in Chinese.')


def request_one(args, rank, prompt, expected=None):
    body = dict(model=args.model, messages=[dict(role='user', content=prompt)],
                stream=True, stream_options={'include_usage': True},
                temperature=0, max_tokens=128, seed=0,
                chat_template_kwargs={'thinking': False, 'enable_thinking': False})
    if expected is None:
        body['ignore_eos'] = True
    req = Request(args.api.rstrip('/')+'/v1/chat/completions', json.dumps(body).encode(),
                  {'Content-Type':'application/json', 'X-data-parallel-rank':str(rank)})
    started, wall = time.perf_counter(), time.time()
    row = dict(rank=rank, start_unix=wall, body=body, events=[])
    text, reasoning, usage, finish, done = [], [], None, None, False
    try:
        with build_opener(ProxyHandler({})).open(req, timeout=300) as response:
            for line in response:
                if not line.startswith(b'data: '):
                    continue
                data = line[6:].strip()
                if data == b'[DONE]':
                    done = True
                    break
                event = json.loads(data)
                elapsed = time.perf_counter()-started
                row['events'].append({'elapsed_s':elapsed, 'event':event})
                if event.get('error'):
                    raise RuntimeError(str(event['error']))
                if event.get('usage'):
                    usage = event['usage']
                for choice in event.get('choices', []):
                    delta = choice.get('delta', {})
                    content = delta.get('content') or ''
                    thought = delta.get('reasoning_content') or delta.get('reasoning') or ''
                    if content or thought:
                        row.setdefault('ttft_s', elapsed)
                        row['last_token_s'] = elapsed
                    if content:
                        row.setdefault('content_ttft_s', elapsed)
                        text.append(content)
                    reasoning.append(thought)
                    finish = choice.get('finish_reason') or finish
        row.update(text=''.join(text), reasoning=''.join(reasoning), usage=usage,
                   finish_reason=finish, done=done)
        assert done and usage and 'ttft_s' in row, 'Incomplete stream'
        n = usage['completion_tokens']
        row['decode_average_s_per_output_token'] = (row['last_token_s']-row['ttft_s'])/max(1,n-1)
        if expected is None:
            assert n == 128, f'Expected 128 output tokens, got {n}'
            row['passed'] = True
        else:
            row['expected'] = expected
            row['parsed'] = json.loads(row['text'])
            row['passed'] = row['parsed'] == expected and finish == 'stop'
    except Exception as exc:
        row.update(passed=False, error=repr(exc), text=''.join(text), usage=usage)
    row.update(end_unix=time.time(), latency_s=time.perf_counter()-started)
    return row


def batch(args, count):
    gate = threading.Barrier(count)
    def one(i):
        gate.wait(timeout=30)
        return request_one(args, i % args.dp, PROMPT)
    start = time.time()
    with ThreadPoolExecutor(count) as pool:
        rows = list(pool.map(one, range(count)))
    end = time.time()
    return dict(concurrency=count, start_unix=start, end_unix=end, requests=rows,
                passed=all(r['passed'] for r in rows),
                ttft_median_s=statistics.median(r.get('ttft_s', float('nan')) for r in rows),
                latency_median_s=statistics.median(r['latency_s'] for r in rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--api', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--dp', type=int, default=1)
    p.add_argument('--concurrency', type=int, nargs='+', default=[4,8])
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--warmups', type=int, default=1)
    p.add_argument('--start-at', type=float, default=0)
    p.add_argument('--stop-file', type=Path)
    p.add_argument('--abort-file', type=Path)
    p.add_argument('--max-seconds', type=float, default=1800)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = dict(started=time.time(), arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                  answers=[], warmup=[], rounds=[])
    def save():
        a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    def guard():
        if a.abort_file and a.abort_file.exists():
            raise RuntimeError(a.abort_file.read_text())
    with a.output.open('x') as f:
        json.dump(report, f)
    try:
        while time.time()<a.start_at:
            guard()
            time.sleep(min(.1,a.start_at-time.time()))
        guard()
        tasks=[(case, rank) for case in CASES for rank in range(a.dp)]
        with ThreadPoolExecutor(max(1,a.dp)) as pool:
            report['answers']=list(pool.map(lambda t: request_one(a,t[1],t[0][1],t[0][2]),tasks))
        save()
        if not all(x['passed'] for x in report['answers']):
            raise RuntimeError('Structured correctness check failed')
        for count in a.concurrency:
            for _ in range(a.warmups):
                guard(); report['warmup'].append(batch(a,count)); save()
            for _ in range(a.rounds):
                guard(); report['rounds'].append(batch(a,count)); save()
        while a.stop_file and not a.stop_file.exists() and time.time()-report['started']<a.max_seconds:
            guard(); report['rounds'].append(batch(a,a.concurrency[-1])); save()
        if a.stop_file and not a.stop_file.exists():
            raise TimeoutError('Background replay deadline reached without stop marker')
        report['passed']=all(r['passed'] for r in report['warmup']+report['rounds'])
    except Exception as exc:
        report.update(passed=False,error=repr(exc))
    finally:
        report['finished']=time.time(); save()
    print(json.dumps({k:report[k] for k in ('passed','started','finished')}),flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
