"""Run cached, resumable development generation on diagnostic retrieval cases."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from .generate import summarize
from .research_data import development_queries, ids

ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL = ROOT / "results" / "experiments" / "E07_research_retrieval" / "per_query.csv"
DEFAULT_OUT = ROOT / "results" / "experiments" / "E09_development_generation" / "summaries.csv"
CONFIGS = ("B1", "M_nocap", "M", "M_vision")


def diagnostic_sample(n: int = 12) -> pd.DataFrame:
    """Deterministic mix of visual rescues, harms, and typical cases."""
    ranks = pd.read_csv(RETRIEVAL)
    ranks["shift"] = ranks["rank_alpha_0.75"] - ranks["rank_alpha_1"]
    rescue = ranks[(ranks["rank_alpha_1"] > 5) & (ranks["rank_alpha_0.75"] <= 5)].copy()
    rescue["stratum"] = "visual_rescue"
    harm = ranks[(ranks["rank_alpha_1"] <= 5) & (ranks["rank_alpha_0.75"] > 5)].copy()
    harm["stratum"] = "visual_harm"
    # Rank movements outside the top-5 boundary still reveal useful conflicts.
    shifted = ranks[~ranks.test_id.isin(set(rescue.test_id) | set(harm.test_id))].copy()
    shifted = pd.concat([
        shifted.nlargest(3, "shift").assign(stratum="image_downrank"),
        shifted.nsmallest(3, "shift").assign(stratum="image_uprank"),
    ])
    used = set(rescue.test_id) | set(harm.test_id) | set(shifted.test_id)
    neutral = ranks[~ranks.test_id.isin(used)].sample(
        n=max(0, n - len(rescue) - len(harm) - len(shifted)), random_state=42
    ).assign(stratum="neutral")
    sample = pd.concat([rescue, harm, shifted, neutral], ignore_index=True).head(n)
    queries = development_queries()[["test_id", "query_qa"]].rename(
        columns={"query_qa": "query"}
    )
    sample = sample.drop(columns=["query"], errors="ignore").merge(
        queries[["test_id", "query"]], on="test_id", validate="one_to_one"
    )
    return sample[["test_id", "query", "stratum", "rank_alpha_1", "rank_alpha_0.75"]].rename(
        columns={"rank_alpha_1": "rank_text", "rank_alpha_0.75": "rank_mm"}
    )


def run(n: int = 12, configs=CONFIGS, out: Path = DEFAULT_OUT, delay: float = 31.0):
    sample = diagnostic_sample(n)
    pool, forbidden = ids("pool"), ids("development") | ids("final_test")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        rows = pd.read_csv(out).to_dict("records")
    else:
        rows = []
    done = {(r["test_id"], r["config"]) for r in rows}
    total = len(sample) * len(configs)
    for item in sample.itertuples(index=False):
        for config in configs:
            if (item.test_id, config) in done:
                continue
            started = time.time()
            rec = summarize(
                item.query,
                config=config,
                k=5,
                alpha=0.75,
                tau=0.35,
                test_id=item.test_id,
                candidate_ids=pool,
                forbidden_ids=forbidden,
            )
            rec.update({"stratum": item.stratum,
                        "gold_rank_text": item.rank_text,
                        "gold_rank_mm": item.rank_mm})
            rows.append(rec)
            pd.DataFrame(rows).to_csv(out, index=False)
            done.add((item.test_id, config))
            print(f"  {len(done)}/{total} {item.test_id} {config} "
                  f"({time.time() - started:.1f}s)", flush=True)
            if len(done) < total and delay:
                time.sleep(delay)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--delay", type=float, default=31.0,
                        help="seconds between API calls (project limit is 2 RPM)")
    args = parser.parse_args()
    run(args.n, tuple(args.configs), args.out, args.delay)


if __name__ == "__main__":
    main()
