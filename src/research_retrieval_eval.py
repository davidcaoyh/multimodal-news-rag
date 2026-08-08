"""Development-only evaluation of fusion weights and reciprocal-rank fusion."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .research_data import development_queries, ids
from .retrieve import component_scores

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "experiments" / "E07_research_retrieval"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)


def rank_of(frame: pd.DataFrame, score: np.ndarray, gold: str) -> int:
    order = np.argsort(-score, kind="stable")
    positions = np.flatnonzero(frame.iloc[order].id.to_numpy() == gold)
    return int(positions[0]) + 1 if len(positions) else len(frame) + 1


def run(limit: int | None = None, out: Path = DEFAULT_OUT) -> tuple[pd.DataFrame, pd.DataFrame]:
    queries = development_queries()
    if limit:
        queries = queries.head(limit)
    candidates = ids("pool") | ids("development")
    rows = []
    started = time.time()
    for number, row in enumerate(queries.itertuples(index=False), 1):
        scores = component_scores(row.query_qa, candidate_ids=candidates)
        rec = {"test_id": row.test_id, "query": row.query_qa}
        for alpha in ALPHAS:
            fused = alpha * scores.n_text.to_numpy() + (1 - alpha) * scores.n_img.to_numpy()
            rec[f"rank_alpha_{alpha:g}"] = rank_of(scores, fused, row.test_id)

        text_rank = scores.n_text.rank(method="first", ascending=False).to_numpy()
        image_rank = scores.n_img.rank(method="first", ascending=False).to_numpy()
        rrf = 1 / (60 + text_rank) + 1 / (60 + image_rank)
        rec["rank_rrf"] = rank_of(scores, rrf, row.test_id)
        rows.append(rec)
        if number % 10 == 0 or number == len(queries):
            print(f"  {number}/{len(queries)} queries ({time.time() - started:.1f}s)", flush=True)

    per_query = pd.DataFrame(rows)
    metrics = []
    rank_cols = [c for c in per_query if c.startswith("rank_")]
    for column in rank_cols:
        label = column.removeprefix("rank_")
        ranks = per_query[column]
        metrics.append({
            "method": label,
            "recall_at_1": float((ranks <= 1).mean()),
            "recall_at_5": float((ranks <= 5).mean()),
            "recall_at_10": float((ranks <= 10).mean()),
            "mrr": float((1 / ranks).mean()),
            "median_rank": float(ranks.median()),
        })
    summary = pd.DataFrame(metrics).sort_values(
        ["recall_at_5", "mrr"], ascending=False
    ).reset_index(drop=True)

    text = per_query["rank_alpha_1"]
    best_method = summary.iloc[0].method
    best = per_query[f"rank_{best_method}"]
    diagnostics = {
        "experiment_id": "E07",
        "split": "development only",
        "queries": len(per_query),
        "candidate_articles": len(candidates),
        "selected_by": "highest recall@5, then MRR",
        "selected_method": best_method,
        "text_recall_at_5": float((text <= 5).mean()),
        "selected_recall_at_5": float((best <= 5).mean()),
        "visual_rescues_at_5": int(((text > 5) & (best <= 5)).sum()),
        "fusion_harms_at_5": int(((text <= 5) & (best > 5)).sum()),
        "pure_image_only_hits_at_5": int(((text > 5) & (per_query["rank_alpha_0"] <= 5)).sum()),
    }
    out.mkdir(parents=True, exist_ok=True)
    per_query.to_csv(out / "per_query.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print("\n" + summary.to_string(index=False))
    print("\n" + json.dumps(diagnostics, indent=2))
    return per_query, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run(args.limit, args.out)


if __name__ == "__main__":
    main()
