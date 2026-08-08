import unittest

from src.build_research_split import UnionFind, choose_final_groups


class ResearchSplitTests(unittest.TestCase):
    def test_union_find_connects_transitively(self):
        uf = UnionFind(["a", "b", "c", "d"])
        uf.union("a", "b")
        uf.union("b", "c")
        self.assertEqual(uf.find("a"), uf.find("c"))
        self.assertNotEqual(uf.find("a"), uf.find("d"))

    def test_group_selection_never_splits_a_group(self):
        groups = [["a", "b"], ["c"], ["d", "e", "f"], ["g"]]
        chosen = choose_final_groups(groups, target=4, seed=42)
        for group in groups:
            self.assertIn(len(set(group) & chosen), (0, len(group)))

    def test_group_selection_is_deterministic(self):
        groups = [[str(i)] for i in range(20)]
        self.assertEqual(
            choose_final_groups(groups, target=5, seed=1508),
            choose_final_groups(groups, target=5, seed=1508),
        )


if __name__ == "__main__":
    unittest.main()

