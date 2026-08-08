"""Accessors and invariants for the duplicate-group-safe research roles."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "data.parquet"
MANIFEST = ROOT / "data" / "research" / "split_manifest.parquet"
BASELINE_QUERIES = ROOT / "data" / "processed" / "queries.parquet"


@lru_cache(maxsize=1)
def articles() -> pd.DataFrame:
    data = pd.read_parquet(DATA)
    roles = pd.read_parquet(MANIFEST)
    out = data.merge(roles, on="id", validate="one_to_one")
    if len(out) != len(data):
        raise AssertionError("research manifest does not cover the full corpus")
    return out


def ids(role: str) -> set[str]:
    if role not in {"pool", "development", "final_test"}:
        raise ValueError(f"unknown research role {role!r}")
    df = articles()
    return set(df.loc[df.research_role == role, "id"])


def development_queries() -> pd.DataFrame:
    """Return the 150 inherited queries now designated for development.

    The extra 16 duplicate-linked development articles have no inherited query
    and are excluded from query-level tuning; they remain excluded from the pool.
    """
    q = pd.read_parquet(BASELINE_QUERIES)
    dev = ids("development")
    out = q[q.test_id.isin(dev)].copy()
    if len(out) != 150 or not set(out.test_id) <= dev:
        raise AssertionError("expected all 150 inherited queries in development")
    return out


def assert_generation_contract(candidate_ids: set[str], heldout_ids: set[str]) -> None:
    if candidate_ids & heldout_ids:
        raise AssertionError("generation candidates overlap held-out ids")
    if candidate_ids != ids("pool"):
        raise AssertionError("research generation must use exactly the research pool")

