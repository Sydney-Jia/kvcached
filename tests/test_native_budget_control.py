# SPDX-License-Identifier: Apache-2.0
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "native_budget_client_test", Path(__file__).parent / "ascend/native_budget_control.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BudgetClientTests(unittest.TestCase):
    def fixture(self):
        return dict(
            results=[
                dict(
                    revision=2,
                    observed_ticket=9,
                    expected_ranks=[0, 1],
                    ranks=[
                        dict(rank=i, ticket=9, ok=True, fatal=None, budget=dict(revision=2))
                        for i in (0, 1)
                    ],
                )
            ]
        )

    def test_requires_every_rank_with_current_revision(self):
        self.assertEqual(module.validate_status(self.fixture())["revision"], 2)
        for change in ("missing", "stale", "duplicate", "failure"):
            reply = copy.deepcopy(self.fixture())
            ranks = reply["results"][0]["ranks"]
            if change == "missing":
                ranks.pop()
            elif change == "stale":
                ranks[1]["budget"]["revision"] = 1
            elif change == "duplicate":
                ranks[1]["rank"] = 0
            else:
                ranks[1]["ok"] = False
            with self.subTest(change=change), self.assertRaises(ValueError):
                module.validate_status(reply)


if __name__ == "__main__":
    unittest.main()
