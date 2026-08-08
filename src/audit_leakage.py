"""Audit exact and near-duplicate leakage across the committed corpus split.

This is a read-only research tool. It never rewrites the committed parquet files.
It writes an experiment-specific pair table and JSON summary so any future split
decision is based on inspectable evidence rather than an undocumented threshold.

Text candidates use character n-gram TF-IDF cosine similarity. Character n-grams
are intentionally robust to punctuation and small editorial updates in follow-up
news stories. Images use file SHA-256 for exact matches and a dependency-free
64-bit difference hash (dHash) for resized/recompressed near-duplicates.

Run from the repository root:

    python -m src.audit_leakage
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .embed import clean_body

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "data.parquet"
POOL = ROOT / "data" / "processed" / "index_pool.parquet"
TEST = ROOT / "data" / "processed" / "test.parquet"
DEFAULT_OUT = ROOT / "results" / "experiments" / "E04_leakage_audit"


def normalize_text(text: str) -> str:
    """Canonical form used only for exact duplicate detection."""
    text = clean_body(text or "").lower()
    return re.sub(r"\W+", " ", text).strip()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dhash64(path: Path) -> int:
    """64-bit horizontal difference hash, invariant to ordinary resizing."""
    with Image.open(path) as image:
        pixels = np.asarray(
            image.convert("L").resize((9, 8), Image.Resampling.LANCZOS),
            dtype=np.int16,
        )
    bits = (pixels[:, 1:] > pixels[:, :-1]).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _pair_record(df: pd.DataFrame, i: int, j: int, kind: str, **metrics) -> dict:
    a, b = df.iloc[i], df.iloc[j]
    return {
        "kind": kind,
        "id_a": a.id,
        "split_a": a.split,
        "headline_a": a.headline,
        "id_b": b.id,
        "split_b": b.split,
        "headline_b": b.headline,
        "cross_split": bool(a.split != b.split),
        **metrics,
    }


def text_pairs(df: pd.DataFrame, threshold: float, neighbors: int) -> list[dict]:
    cleaned = [normalize_text(x) for x in df.body]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=100_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(cleaned)
    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(matrix)
    distances, indices = nn.kneighbors(
        matrix, n_neighbors=min(neighbors + 1, len(df)), return_distance=True
    )

    out, seen = [], set()
    for i, (ds, js) in enumerate(zip(distances, indices)):
        for distance, j in zip(ds, js):
            j = int(j)
            if i == j:
                continue
            pair = tuple(sorted((i, j)))
            if pair in seen:
                continue
            similarity = 1.0 - float(distance)
            if similarity < threshold:
                continue
            seen.add(pair)
            kind = "text_exact" if cleaned[i] == cleaned[j] else "text_near"
            out.append(_pair_record(
                df, i, j, kind, text_similarity=round(similarity, 6),
                image_hamming=np.nan,
            ))
    return out


def image_pairs(df: pd.DataFrame, max_hamming: int) -> list[dict]:
    hashes = []
    for relative in df.image_path:
        path = ROOT / relative
        hashes.append((file_sha256(path), dhash64(path)))

    out = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            exact = hashes[i][0] == hashes[j][0]
            distance = hamming64(hashes[i][1], hashes[j][1])
            if not exact and distance > max_hamming:
                continue
            out.append(_pair_record(
                df, i, j, "image_exact" if exact else "image_near",
                text_similarity=np.nan, image_hamming=distance,
            ))
    return out


def load_corpus() -> pd.DataFrame:
    df = pd.read_parquet(DATA).copy()
    pool_ids = set(pd.read_parquet(POOL).id)
    test_ids = set(pd.read_parquet(TEST).id)
    if pool_ids & test_ids:
        raise ValueError("committed pool and test IDs overlap")
    unknown = set(df.id) - pool_ids - test_ids
    if unknown:
        raise ValueError(f"{len(unknown)} corpus rows belong to neither split")
    df["split"] = np.where(df.id.isin(test_ids), "test", "pool")
    return df.sort_values("id").reset_index(drop=True)


def summarize(pairs: pd.DataFrame, df: pd.DataFrame, args) -> dict:
    by_kind = pairs.kind.value_counts().sort_index().to_dict() if len(pairs) else {}
    cross = pairs[pairs.cross_split] if len(pairs) else pairs
    return {
        "experiment_id": "E04",
        "corpus_rows": int(len(df)),
        "pool_rows": int((df.split == "pool").sum()),
        "test_rows": int((df.split == "test").sum()),
        "text_threshold": args.text_threshold,
        "text_neighbors_per_item": args.text_neighbors,
        "image_max_hamming": args.image_max_hamming,
        "candidate_pairs": int(len(pairs)),
        "cross_split_pairs": int(len(cross)),
        "cross_split_articles": int(len(set(cross.id_a) | set(cross.id_b))) if len(cross) else 0,
        "pairs_by_kind": {str(k): int(v) for k, v in by_kind.items()},
        "cross_split_by_kind": (
            {str(k): int(v) for k, v in cross.kind.value_counts().sort_index().to_dict().items()}
            if len(cross) else {}
        ),
        "interpretation": (
            "Candidate pairs require manual inspection. Similarity thresholds identify "
            "possible leakage; they do not by themselves prove duplicate news content."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-threshold", type=float, default=0.85)
    parser.add_argument("--text-neighbors", type=int, default=10)
    parser.add_argument("--image-max-hamming", type=int, default=5)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not 0 <= args.text_threshold <= 1:
        parser.error("--text-threshold must be in [0, 1]")
    if args.text_neighbors < 1:
        parser.error("--text-neighbors must be positive")
    if not 0 <= args.image_max_hamming <= 64:
        parser.error("--image-max-hamming must be in [0, 64]")

    df = load_corpus()
    records = text_pairs(df, args.text_threshold, args.text_neighbors)
    records.extend(image_pairs(df, args.image_max_hamming))
    pairs = pd.DataFrame(records)
    if len(pairs):
        pairs = pairs.sort_values(
            ["cross_split", "kind", "text_similarity", "image_hamming"],
            ascending=[False, True, False, True], na_position="last",
        ).reset_index(drop=True)
    else:
        pairs = pd.DataFrame(columns=[
            "kind", "id_a", "split_a", "headline_a", "id_b", "split_b",
            "headline_b", "cross_split", "text_similarity", "image_hamming",
        ])

    args.out.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.out / "candidate_pairs.csv", index=False)
    summary = summarize(pairs, df, args)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    if summary["cross_split_pairs"]:
        print("\nCross-split candidates:")
        cols = ["kind", "text_similarity", "image_hamming", "headline_a", "headline_b"]
        print(pairs[pairs.cross_split][cols].to_string(index=False))
    print(f"\nwrote {args.out.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()

