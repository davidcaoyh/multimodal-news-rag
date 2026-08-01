"""
Day 1 helper: add coarse story-type labels and split into index_pool / test.

Reads  data/processed/data.parquet
Writes data/processed/index_pool.parquet and data/processed/test.parquet

Run from repo root:  python3 src/day1_split.py
"""
import pandas as pd

SRC = "data/processed/data.parquet"


def bucket(section):
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
