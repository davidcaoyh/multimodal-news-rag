"""Build a duplicate-group-safe research split without changing baseline data.

The inherited 150-item test set has already been used for model exploration, so
it becomes the development set. Every near-duplicate component touching it also
becomes development data. A new seeded final-test set is sampled by whole
duplicate components from the remaining articles; all other articles form the
retrieval pool.

Requires E04's candidate table:

    python -m src.audit_leakage
    python -m src.build_research_split
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "data.parquet"
BASELINE_POOL = ROOT / "data" / "processed" / "index_pool.parquet"
BASELINE_TEST = ROOT / "data" / "processed" / "test.parquet"
PAIRS = ROOT / "results" / "experiments" / "E04_leakage_audit" / "candidate_pairs.csv"
DEFAULT_OUT = ROOT / "data" / "research"
DEFAULT_META = ROOT / "results" / "experiments" / "E05_research_split"


class UnionFind:
    def __init__(self, values):
        self.parent = {v: v for v in values}

    def find(self, value):
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def choose_final_groups(groups: list[list[str]], target: int, seed: int) -> set[str]:
    """Seeded whole-group selection, preferring the closest count to target."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    selected: set[str] = set()
    count = 0
    for idx in order:
        group = groups[int(idx)]
        proposed = count + len(group)
        if proposed <= target or abs(proposed - target) < abs(count - target):
            selected.update(group)
            count = proposed
        if count == target:
            break
    return selected


def build(target_final: int = 150, seed: int = 1508):
    data = pd.read_parquet(DATA).copy()
    baseline_pool = pd.read_parquet(BASELINE_POOL)
    baseline_test = pd.read_parquet(BASELINE_TEST)
    pairs = pd.read_csv(PAIRS)

    ids = set(data.id)
    if ids != set(baseline_pool.id) | set(baseline_test.id):
        raise ValueError("baseline pool/test do not partition data.parquet")

    uf = UnionFind(ids)
    for row in pairs.itertuples(index=False):
        uf.union(row.id_a, row.id_b)

    components: dict[str, list[str]] = {}
    for article_id in sorted(ids):
        components.setdefault(uf.find(article_id), []).append(article_id)

    inherited_dev = set(baseline_test.id)
    dev_roots = {uf.find(article_id) for article_id in inherited_dev}
    dev_ids = {
        article_id
        for root in dev_roots
        for article_id in components[root]
    }

    eligible_groups = [
        members for root, members in components.items() if root not in dev_roots
    ]
    final_ids = choose_final_groups(eligible_groups, target_final, seed)
    pool_ids = ids - dev_ids - final_ids

    role = {}
    role.update({x: "development" for x in dev_ids})
    role.update({x: "final_test" for x in final_ids})
    role.update({x: "pool" for x in pool_ids})
    data["research_role"] = data.id.map(role)
    data["duplicate_group"] = data.id.map(lambda x: uf.find(x))

    # Candidate near-duplicate edges may never cross research roles.
    role_by_id = dict(zip(data.id, data.research_role))
    crossing = pairs[
        pairs.apply(lambda r: role_by_id[r.id_a] != role_by_id[r.id_b], axis=1)
    ]
    if len(crossing):
        raise AssertionError(f"{len(crossing)} duplicate edges cross research roles")
    if set(dev_ids) & set(final_ids) or set(dev_ids) & set(pool_ids) or set(final_ids) & set(pool_ids):
        raise AssertionError("research roles overlap")

    summary = {
        "experiment_id": "E05",
        "seed": seed,
        "target_final_test": target_final,
        "corpus_rows": len(data),
        "pool_rows": len(pool_ids),
        "development_rows": len(dev_ids),
        "final_test_rows": len(final_ids),
        "inherited_test_rows": len(inherited_dev),
        "extra_development_rows_from_duplicate_groups": len(dev_ids - inherited_dev),
        "duplicate_components": sum(len(x) > 1 for x in components.values()),
        "largest_duplicate_component": max(map(len, components.values())),
        "candidate_edges_crossing_roles": len(crossing),
        "policy": (
            "Inherited test becomes development because it was used for alpha exploration. "
            "Near-duplicate connected components are assigned atomically."
        ),
    }
    return data, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-final", type=int, default=150)
    parser.add_argument("--seed", type=int, default=1508)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--meta-out", type=Path, default=DEFAULT_META)
    args = parser.parse_args()
    if args.target_final < 1:
        parser.error("--target-final must be positive")

    data, summary = build(args.target_final, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    args.meta_out.mkdir(parents=True, exist_ok=True)
    # Store only the role/group manifest. Article content remains canonical in
    # data/processed/data.parquet; duplicating it into four research parquets adds
    # megabytes and creates multiple sources of truth for the same corpus.
    data[["id", "research_role", "duplicate_group"]].to_parquet(
        args.out / "split_manifest.parquet", index=False
    )
    (args.meta_out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out.relative_to(ROOT)}/ and {args.meta_out.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
