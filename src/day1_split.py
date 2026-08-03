"""
Day 1 helper: add coarse story-type labels and split into index_pool / test.

Reads  data/processed/data.parquet
Writes data/processed/index_pool.parquet and data/processed/test.parquet

Run from repo root:  python3 src/day1_split.py
"""
import pandas as pd

SRC = "data/processed/data.parquet"


def bucket(section):
    """Coarse story-type label. DEAD WEIGHT — do not build analysis on `story_type`.

    BBC's `section` is geographic ("Middle East", "Wales", "US & Canada"), not topical,
    so this maps 68% of the corpus to "other", puts n=1 in "sport", and leaves 142 rows
    with no section at all. Nothing reads `story_type`: it never enters the split (D6),
    Day 2 ignores it, and Day 5's stratified cut was dropped rather than run on it.

    Kept only because day1_split.py writes the COMMITTED parquets. Deleting it would
    drop a column from a frozen artifact that must not be regenerated (see CLAUDE.md
    § Data). The correct fix, if the stratified analysis is ever revived, is to
    hand-label the 150 test items event_centric / abstract_topical — not to widen this.
    """
    s = (section or "").lower()
    if any(w in s for w in ["world", "uk", "politics"]):
        return "event_politics"
    if "business" in s or "econom" in s:
        return "markets_business"
    if "sport" in s:
        return "sport"
    if any(w in s for w in ["tech", "science", "health"]):
        return "sci_tech"
    return "other"


def main():
    df = pd.read_parquet(SRC)
    df["story_type"] = df["section"].map(bucket)
    print("story_type counts:\n", df["story_type"].value_counts(), "\n")

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_test = min(150, len(df) // 5)
    test = df.iloc[:n_test]
    index_pool = df.iloc[n_test:]

    # lightweight leakage guard: drop index rows whose headline duplicates a test headline
    test_heads = set(test["headline"].str.lower().str.strip())
    index_pool = index_pool[
        ~index_pool["headline"].str.lower().str.strip().isin(test_heads)
    ]

    test.to_parquet("data/processed/test.parquet", index=False)
    index_pool.to_parquet("data/processed/index_pool.parquet", index=False)
    print(f"index_pool: {len(index_pool)}   test: {len(test)}")


if __name__ == "__main__":
    main()
