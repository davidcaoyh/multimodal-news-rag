import unittest

from src.final_validation import export
from src.research_final import frozen_sample


class FinalResearchTests(unittest.TestCase):
    def test_frozen_sample_is_balanced_and_unique(self):
        sample = frozen_sample()
        self.assertEqual(len(sample), 20)
        self.assertTrue(sample.id.is_unique)
        self.assertEqual(set(sample.category.value_counts()), {4})

    def test_human_validation_export_is_blind(self):
        sample = export()
        self.assertEqual(len(sample), 50)
        self.assertTrue(sample.sample_id.is_unique)
        self.assertNotIn("config", sample.columns)
        self.assertNotIn("supported", sample.columns)
        self.assertNotIn("support", sample.columns)


if __name__ == "__main__":
    unittest.main()
