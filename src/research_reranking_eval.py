"""Development-only comparison of lexical retrieval and two-stage reranking."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .lexical import scores as lexical_scores
from .research_data import development_queries, ids
from .research_retrieval_eval import rank_of
from .retrieve import component_scores

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "experiments" / "E08_research_reranking"


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi - lo < 1e-9 else (x - lo) / (hi - lo)


def _two_stage(base: np.ndarray, lexical: np.ndarray, n: int = 30,
               base_weight: float = 0.75) -> np.ndarray:
    """Retrieve n on base score, then rerank only those candidates with TF-IDF."""
    candidate = np.argsort(-base, kind="stable")[:n]
    out = np.full_like(base, -np.inf)
    out[candidate] = (
        base_weight * _minmax(base[candidate])
        + (1 - base_weight) * _minmax(lexical[candidate])
    )
    return out


def run(limit: int | None = None, out: Path = DEFAULT_OUT):
    queries = development_queries()
    if limit:
        queries = queries.head(limit)
    candidates = ids("pool") | ids("development")
    rows, started = [], time.time()
    for number, row in enumerate(queries.itertuples(index=False), 1):
        dense = component_scores(row.query_qa, candidate_ids=candidates)
        lex = lexical_scores(row.query_qa, candidates)
        merged = dense.merge(lex, on="id", validate="one_to_one")
        text = merged.n_text.to_numpy()
        image = merged.n_img.to_numpy()
        lexical = _minmax(merged.s_lex.to_numpy())
        mm = 0.75 * text + 0.25 * image
        methods = {
            "dense_text": text,
            "lexical": lexical,
            "dense_text_lexical": 0.75 * text + 0.25 * lexical,
            "text_rerank_top30": _two_stage(text, lexical),
            "mm_no_rerank": mm,
            "mm_text_rerank_top30": _two_stage(mm, lexical),
        }
        rec = {"test_id": row.test_id, "query": row.query_qa}
        rec.update({f"rank_{name}": rank_of(merged, score, row.test_id)
                    for name, score in methods.items()})
        rows.append(rec)
        if number % 25 == 0 or number == len(queries):
            print(f"  {number}/{len(queries)} ({time.time() - started:.1f}s)", flush=True)

    per_query = pd.DataFrame(rows)
    metrics = []
    for column in [c for c in per_query if c.startswith("rank_")]:
        ranks = per_query[column]
        metrics.append({
            "method": column.removeprefix("rank_"),
            "recall_at_1": float((ranks <= 1).mean()),
            "recall_at_5": float((ranks <= 5).mean()),
            "recall_at_10": float((ranks <= 10).mean()),
            "mrr": float((1 / ranks).mean()),
            "median_rank": float(ranks.median()),
        })
    summary = pd.DataFrame(metrics).sort_values(
        ["recall_at_5", "mrr"], ascending=False
    ).reset_index(drop=True)
    diagnostics = {
        "experiment_id": "E08",
        "split": "development only",
        "queries": len(per_query),
        "candidate_articles": len(candidates),
        "rerank_depth": 30,
        "selected_method": summary.iloc[0].method,
        "selection_rule": "highest recall@5, then MRR",
    }
    out.mkdir(parents=True, exist_ok=True)
    per_query.to_csv(out / "per_query.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print("\n" + summary.to_string(index=False))
    print("\n" + json.dumps(diagnostics, indent=2))
    return per_query, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run(args.limit, args.out)


if __name__ == "__main__":
    main()
