# SPDX-License-Identifier: Apache-2.0
"""The migrated traffic protocol must not turn missing output into a success."""
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('burst_under_test',
    Path(__file__).parent / 'ascend/service_burst.py')
burst = importlib.util.module_from_spec(spec)
spec.loader.exec_module(burst)


def config(url='http://127.0.0.1:1'):
    return {'services': {label: {'url': url, 'model': label, 'control': 'native-residency',
                                'control_results': 1} for label in ('a', 'b')},
            'cases': {'one': {'payload': {'prompt': '1, 2', 'max_tokens': 8},
                              'path': '/v1/completions', 'expected_numbers': [1, 2]}},
            'phases': [{'name': 'burst', 'duration_s': .08,
                        'rates': {'a': 30, 'b': 15}, 'cases': {'a': ['one'], 'b': ['one']}}],
            'events': [{'at_s': .01, 'service': 'a', 'action': 'elastic'},
                       {'at_s': .03, 'service': 'a', 'action': 'held'}]}


def sse(text='1, 2', finish=True, done=True, error=False):
    rows = [{'choices': [{'delta': {'content': text}, 'token_ids': [10, 11]}]},
            {'choices': [], 'usage': {'completion_tokens': 2}}]
    if finish:
        rows.append({'choices': [{'finish_reason': 'stop'}]})
    if error:
        rows.append({'error': {'message': 'worker failed'}})
    return (''.join('data: ' + json.dumps(row) + '\n\n' for row in rows)
            + ('data: [DONE]\n\n' if done else '')).encode()


class BurstProtocolTest(unittest.TestCase):
    def test_open_loop_zero_rate_case_cycle_and_bounds(self):
        c = config()
        offered = burst.validate(c, True)
        self.assertEqual([r['service'] for r in offered], ['a', 'b', 'a', 'a', 'b'])
        self.assertEqual([r['index'] for r in offered], list(range(5)))
        c['phases'][0]['rates']['b'] = 0
        self.assertEqual(len(burst.validate(c, True)), 3)
        for value in (-1, float('nan'), float('inf'), 1001):
            with self.subTest(value=value), self.assertRaises(ValueError):
                c['phases'][0]['rates']['a'] = value
                burst.validate(c, True)

    def test_explicit_mutation_adapter_and_bounded_oracles(self):
        for mutate in (lambda c: c['services']['a'].update(control='none'),
                       lambda c: c['events'][0].update(at_s=-1),
                       lambda c: c['cases']['one']['payload'].update(max_tokens=0),
                       lambda c: c['cases']['one'].update(expected_numbers=[-1])):
            c = config()
            mutate(c)
            with self.assertRaises(ValueError):
                burst.validate(c, True)

    def test_all_returned_workers_must_accept_without_mutation_retry(self):
        service = config()['services']['a']
        good = {'ok': True, 'poisoned': False, 'action': 'held'}
        for rows in ([], [dict(good, ok=False)], [dict(good, poisoned=True)],
                     [dict(good, action='elastic')], [good, good]):
            with self.subTest(rows=rows), patch.object(burst, 'http', return_value=io.BytesIO(
                    json.dumps({'results': rows}).encode())) as call:
                with self.assertRaises(RuntimeError):
                    burst.control(service, 'held')
                self.assertEqual(call.call_count, 1)
        with patch.object(burst, 'http', return_value=io.BytesIO(json.dumps({'results': [good]}).encode())):
            self.assertTrue(burst.control(service, 'held')['results'][0]['ok'])

    def response(self, stream):
        c = config()
        offered = burst.arrivals(c)[0][0]
        with patch.object(burst, 'http', return_value=io.BytesIO(stream)):
            return burst.request(c['services']['a'], c['cases']['one'], offered, time.monotonic())

    def test_full_stream_usage_and_signature(self):
        row = self.response(sse())
        self.assertTrue(row['completed'] and row['correct'])
        self.assertEqual(row['token_ids'], [10, 11])
        self.assertEqual(row['completion_tokens'], 2)
        self.assertIsNotNone(row['tpot_s'])
        self.assertEqual(len(row['events']), 3)

    def test_truncation_error_and_extra_text_are_not_success(self):
        for stream in (sse(done=False), sse(finish=False), sse(error=True),
                       sse('answer: 1, 2'), sse('-1, 2'), sse('1, 2, 2')):
            with self.subTest(stream=stream):
                self.assertFalse(self.response(stream)['correct'])

    def test_missing_requests_attribution_and_reference(self):
        c = config()
        offered = burst.arrivals(c)[0]
        row = self.response(sse())
        result = burst.summarize(offered, [row], [], c['events'], {})
        self.assertFalse(result['completed'])
        self.assertEqual(result['missing_request_ids'], [1, 2, 3, 4])
        self.assertEqual(result['services']['a']['offered'], 3)
        self.assertIsNone(result['services']['a']['slo_pass'])
        with self.assertRaises(ValueError):
            burst.summarize(offered, [dict(row, service='b')], [], [], {})
        with self.assertRaises(ValueError):
            burst.summarize(offered, [row, row], [], [], {})
        wrong = dict(row, content='1, 3')
        result = burst.summarize([offered[0]], [row], [], [], {'latency_s': 10}, {0: wrong})
        self.assertEqual(result['services']['a']['correct'], 0)
        self.assertEqual(result['services']['a']['slo_pass'], 0)

    def test_real_http_paired_trace_and_failure_evidence(self):
        actions = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'vllm:num_requests_running 0\n')
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                self.send_response(200)
                self.end_headers()
                if self.path == '/collective_rpc':
                    actions.append(body['args'][0])
                    self.wfile.write(json.dumps({'results': [dict(ok=True, poisoned=False,
                                                     action=body['args'][0])]}).encode())
                else:
                    self.wfile.write(sse())
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as root:
                root = Path(root)
                c = config('http://127.0.0.1:' + str(server.server_port))
                baseline = burst.run(c, root/'baseline', root/'abort', workers=2)
                self.assertTrue(baseline['completed'])
                self.assertEqual(actions, [])
                elastic = burst.run(c, root/'elastic', root/'abort', True, 2, root/'baseline')
                self.assertTrue(elastic['completed'])
                self.assertEqual(actions, ['elastic', 'held'])
                changed = copy.deepcopy(c)
                changed['cases']['one']['payload']['max_tokens'] = 9
                with self.assertRaisesRegex(ValueError, 'different offered workload'):
                    burst.run(changed, root/'changed', root/'abort', reference_dir=root/'baseline')
                (root/'abort').touch()
                with self.assertRaisesRegex(RuntimeError, 'aborted'):
                    burst.run(c, root/'aborted', root/'abort')
                record = json.loads((root/'aborted/summary.json').read_text())
                self.assertFalse(record['completed'])
                self.assertEqual(len(record['missing_request_ids']), 5)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
