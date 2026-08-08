import unittest

from src import research_data


class ResearchDataTests(unittest.TestCase):
    def test_roles_partition_corpus(self):
        pool = research_data.ids("pool")
        dev = research_data.ids("development")
        final = research_data.ids("final_test")
        self.assertFalse(pool & dev)
        self.assertFalse(pool & final)
        self.assertFalse(dev & final)
        self.assertEqual(len(pool | dev | final), 1023)

    def test_generation_contract(self):
        pool = research_data.ids("pool")
        heldout = research_data.ids("development") | research_data.ids("final_test")
        research_data.assert_generation_contract(pool, heldout)
        with self.assertRaises(AssertionError):
            research_data.assert_generation_contract(pool | {next(iter(heldout))}, heldout)

    def test_inherited_queries_are_development_only(self):
        q = research_data.development_queries()
        self.assertEqual(len(q), 150)
        self.assertTrue(set(q.test_id) <= research_data.ids("development"))


if __name__ == "__main__":
    unittest.main()
