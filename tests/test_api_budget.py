import json
import tempfile
import unittest
from pathlib import Path

from src import api_budget


class ApiBudgetTests(unittest.TestCase):
    def test_estimate_uses_audited_prices(self):
        self.assertAlmostEqual(api_budget.estimate_cost("gpt-4o-mini", 1_000_000, 0), 0.15)
        self.assertAlmostEqual(api_budget.estimate_cost("gpt-5.6-luna", 0, 1_000_000), 1.20)

    def test_record_and_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "usage.jsonl"
            row = api_budget.record(
                model="gpt-4o-mini", input_tokens=1000, output_tokens=100,
                category="test", cache_key="abc", path=path,
            )
            self.assertGreater(row["estimated_cost_usd"], 0)
            self.assertEqual(len(path.read_text().splitlines()), 1)
            with self.assertRaises(api_budget.BudgetExceeded):
                api_budget.assert_can_spend(
                    estimated_max_usd=1.0, path=path, limit_usd=0.00001
                )

    def test_unknown_model_fails_closed(self):
        with self.assertRaises(ValueError):
            api_budget.estimate_cost("unpriced-model", 1, 1)


if __name__ == "__main__":
    unittest.main()
