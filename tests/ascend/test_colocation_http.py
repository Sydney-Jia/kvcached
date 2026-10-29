# SPDX-License-Identifier: Apache-2.0
"""CPU checks for complete-stream and objective-answer acceptance."""
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('colocation_http',Path(__file__).with_name('colocation_http.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class StreamChecks(unittest.TestCase):
    def replay(self, text='[0, 1]', done=True, tokens=6, expected=(0,1)):
        events=[{'choices':[{'delta':{'content':text},'finish_reason':'stop'}]},
                {'choices':[],'usage':{'prompt_tokens':20,'completion_tokens':tokens}}]
        stream=b''.join(b'data: '+json.dumps(e).encode()+b'\n\n' for e in events)
        if done:stream+=b'data: [DONE]\n\n'
        opener=SimpleNamespace(open=lambda *a,**k:io.BytesIO(stream))
        args=SimpleNamespace(api='http://127.0.0.1',model='test')
        with patch.object(module,'build_opener',return_value=opener):
            return module.request_one(args,0,'test',None if expected is None else list(expected))

    def test_complete_answer(self):
        row=self.replay();self.assertTrue(row['passed']);self.assertEqual(row['parsed'],[0,1])
        self.assertEqual(len(row['events']),2)
    def test_truncated_stream_is_not_success(self):
        self.assertFalse(self.replay(done=False)['passed'])
    def test_wrong_answer_is_retained(self):
        row=self.replay(text='[1, 0]');self.assertFalse(row['passed']);self.assertEqual(row['text'],'[1, 0]')
    def test_fixed_token_workload_rejects_short_generation(self):
        self.assertFalse(self.replay(tokens=127,expected=None)['passed'])
        self.assertTrue(self.replay(tokens=128,expected=None)['passed'])


if __name__=='__main__':unittest.main()
