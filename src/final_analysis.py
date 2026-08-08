"""Free, post-freeze descriptive retrieval/category/efficiency analysis."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import api_budget
from .lexical import scores as lexical_scores
from .research_data import ids
from .research_retrieval_eval import rank_of
from .retrieve import component_scores

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "experiments" / "E12_final_comparison"


def _retrieval():
    queries = pd.read_csv(OUT / "queries.csv")
    candidates = ids("pool") | ids("final_test")
    rows, started = [], time.time()
    for row in queries.itertuples(index=False):
        before = time.perf_counter()
        dense = component_scores(row.query, candidate_ids=candidates)
        elapsed = time.perf_counter() - before
        lex = lexical_scores(row.query, candidates)
        frame = dense.merge(lex, on="id", validate="one_to_one")
        text = frame.n_text.to_numpy()
        image = frame.n_img.to_numpy()
        lexical = frame.s_lex.to_numpy()
        mm = 0.75 * text + 0.25 * image
        rows.append({
            "test_id": row.test_id, "category": row.category,
            "rank_dense_text": rank_of(frame, text, row.test_id),
            "rank_image": rank_of(frame, image, row.test_id),
            "rank_mm_alpha_0.75": rank_of(frame, mm, row.test_id),
            "rank_lexical": rank_of(frame, lexical, row.test_id),
            "retrieval_seconds": elapsed,
        })
    per_query = pd.DataFrame(rows)
    per_query.to_csv(OUT / "retrieval_per_query.csv", index=False)
    metrics = []
    for column in [c for c in per_query if c.startswith("rank_")]:
        rank = per_query[column]
        metrics.append({
            "method": column.removeprefix("rank_"),
            "recall_at_1": float((rank <= 1).mean()),
            "recall_at_5": float((rank <= 5).mean()),
            "recall_at_10": float((rank <= 10).mean()),
            "mrr": float((1 / rank).mean()),
        })
    summary = pd.DataFrame(metrics)
    summary.to_csv(OUT / "retrieval_metrics.csv", index=False)
    latency = {
        "queries": len(per_query),
        "wall_seconds_including_encoder_load": time.time() - started,
        "median_seconds_per_query": float(per_query.retrieval_seconds.median()),
        "p95_seconds_per_query": float(per_query.retrieval_seconds.quantile(.95)),
    }
    return summary, latency


def _category():
    path = OUT / "evaluation" / "per_item_by_category.csv"
    per_item = pd.read_csv(path)
    usable = per_item[per_item.kind == "usable"]
    wide = usable.pivot(index=["test_id", "category"], columns="config",
                        values="faithfulness")
    rows = []
    for left, right in [("B1", "M"), ("M", "M_vision")]:
        paired = wide[[left, right]].dropna().copy()
        paired["delta"] = paired[right] - paired[left]
        for category, group in paired.groupby(level="category"):
            rows.append({"comparison": f"{right}-{left}", "category": category,
                         "n": len(group), "mean_difference": float(group.delta.mean())})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "category_differences.csv", index=False)
    return out


def _efficiency(latency: dict):
    files = [
        ROOT / "data" / "index" / "text.faiss",
        ROOT / "data" / "index" / "img.faiss",
        ROOT / "data" / "index" / "text_emb.npy",
        ROOT / "data" / "index" / "img_emb.npy",
    ]
    record = {
        "retrieval": latency,
        "index_bytes": {p.name: p.stat().st_size for p in files if p.exists()},
        "api": api_budget.summary(),
        "embedding_build_wall_seconds_observed": 40.0,
        "generation_latency": "not systematically instrumented; do not reconstruct post hoc",
    }
    (OUT / "efficiency.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    retrieval, latency = _retrieval()
    categories = _category()
    efficiency = _efficiency(latency)
    print(retrieval.to_string(index=False))
    print("\n" + categories.to_string(index=False))
    print("\n" + json.dumps(efficiency, indent=2))


if __name__ == "__main__":
    main()
