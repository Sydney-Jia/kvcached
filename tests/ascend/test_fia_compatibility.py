# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""CPU contract checks for the opt-in experiment adapter, not kernel accuracy."""
import types
import unittest

from fia_compatibility import install, status


class CompatibilityContract(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.output = object()

        def original(*args, **kwargs):
            self.calls.append((args, kwargs))
            return self.output

        self.module = types.SimpleNamespace(npu_fused_infer_attention_score=original)

    def test_opt_in_and_idempotence(self):
        original = self.module.npu_fused_infer_attention_score
        self.assertEqual(status(self.module), {"enabled": False})
        self.assertIs(self.module.npu_fused_infer_attention_score, original)
        install(self.module)
        wrapper = self.module.npu_fused_infer_attention_score
        install(self.module)
        self.assertIs(self.module.npu_fused_infer_attention_score, wrapper)

    def test_causal_arguments_preserve_inputs_and_result(self):
        install(self.module)
        query, block_table = object(), object()
        arguments = dict(query=query, block_table=block_table, input_layout="TND",
                         sparse_mode=3, actual_seq_lengths=[1, 390])
        returned = self.module.npu_fused_infer_attention_score(**arguments)
        self.assertIs(returned, self.output)
        actual = self.calls[-1][1]
        self.assertIs(actual["query"], query)
        self.assertIs(actual["block_table"], block_table)
        self.assertEqual(actual["actual_seq_lengths"], [1, 390])
        self.assertEqual((actual["sparse_mode"], actual["pre_tokens"], actual["next_tokens"]),
                         (4, 2147483647, 0))
        self.assertEqual(arguments["sparse_mode"], 3)
        self.assertEqual(status(self.module)["redirected_calls"], 1)

    def test_unrelated_calls_are_unchanged(self):
        install(self.module)
        for patch in ({"sparse_mode": 0}, {"sparse_mode": 4, "pre_tokens": 128},
                      {"block_table": None}, {"input_layout": "BSH"}):
            arguments = dict(input_layout="TND", block_table=object(), sparse_mode=3)
            arguments.update(patch)
            self.module.npu_fused_infer_attention_score(**arguments)
            self.assertEqual(self.calls[-1], ((), arguments))
        arguments = dict(input_layout="TND", block_table=object(), sparse_mode=3)
        self.module.npu_fused_infer_attention_score("positional", **arguments)
        self.assertEqual(self.calls[-1], (("positional",), arguments))
        self.assertEqual(status(self.module)["redirected_calls"], 0)

    def test_status_cannot_mutate_counter(self):
        install(self.module)
        snapshot = status(self.module)
        snapshot["redirected_calls"] = 99
        self.assertEqual(status(self.module)["redirected_calls"], 0)


if __name__ == "__main__":
    unittest.main()
