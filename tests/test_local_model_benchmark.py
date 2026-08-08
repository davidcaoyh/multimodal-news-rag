import json
import unittest

from src.benchmark_local_models import parse_answer


class LocalBenchmarkTests(unittest.TestCase):
    def test_parse_answer_accepts_valid_json(self):
        choice, evidence, confidence = parse_answer(json.dumps({
            "choice": "b", "visible_evidence": "a podium", "confidence": 0.8,
        }))
        self.assertEqual((choice, evidence, confidence), ("B", "a podium", 0.8))

    def test_parse_answer_rejects_non_json_and_invalid_choice(self):
        self.assertEqual(parse_answer("Caption A"), (None, None, None))
        self.assertIsNone(parse_answer('{"choice":"C"}')[0])


if __name__ == "__main__":
    unittest.main()
