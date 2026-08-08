"""Prepare and run the frozen 20-item held-out final comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from . import api_budget
from .generate import _get_client, summarize
from .research_data import articles, ids

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "experiments" / "E12_final_comparison"
QUERIES = OUT_DIR / "queries.csv"
SUMMARIES = OUT_DIR / "summaries.csv"
CONFIGS = ("B1", "M", "M_vision")


def category(section: str) -> str:
    section = str(section or "")
    if section in {"Tennis", "Darts", "Gymnastics", "Sport"}:
        return "sports"
    if section in {"Business", "Technology", "Science & Environment"}:
        return "business_technology_science"
    if section in {"UK Politics", "Middle East"}:
        return "politics_conflict"
    if section in {"US & Canada", "Europe", "China", "Asia", "Africa",
                   "Latin America & Caribbean"}:
        return "international"
    return "society_culture"


def frozen_sample() -> pd.DataFrame:
    frame = articles()
    frame = frame[frame.research_role == "final_test"].copy()
    frame["category"] = frame.section.map(category)
    pieces = []
    for name in ["sports", "business_technology_science", "politics_conflict",
                 "international", "society_culture"]:
        group = frame[frame.category == name]
        if len(group) < 4:
            raise RuntimeError(f"final category {name} has only {len(group)} articles")
        pieces.append(group.sample(n=4, random_state=42))
    sample = pd.concat(pieces, ignore_index=True)
    if len(sample) != 20 or not sample.id.is_unique:
        raise AssertionError("frozen final sample must contain 20 unique articles")
    return sample[["id", "headline", "section", "category"]]


QUERY_SCHEMA = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "string"}, "query": {"type": "string"}},
        "required": ["id", "query"],
        "additionalProperties": False,
    }}},
    "required": ["queries"],
    "additionalProperties": False,
}


def prepare_queries() -> pd.DataFrame:
    if QUERIES.exists():
        return pd.read_csv(QUERIES)
    sample = frozen_sample()
    items = "\n".join(f"{r.id}\t{r.headline}" for r in sample.itertuples(index=False))
    prompt = f"""For each ID and news headline below, write exactly one natural search
question a reader might ask. Use only information in the headline; do not add answers or
outside facts. Do not copy the headline verbatim. Keep each question under 18 words and
preserve every ID exactly.\n\n{items}"""
    key = hashlib.sha256(prompt.encode()).hexdigest()[:24]
    api_budget.assert_can_spend(estimated_max_usd=0.10)
    response = _get_client().chat.completions.create(
        model="gpt-4o-mini", temperature=0, seed=42, max_tokens=3000,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_schema", "json_schema": {
            "name": "final_queries", "strict": True, "schema": QUERY_SCHEMA,
        }},
    )
    parsed = json.loads(response.choices[0].message.content or "")
    api_budget.record(
        model="gpt-4o-mini", input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens, category="final_query_generation",
        cache_key=key,
    )
    generated = pd.DataFrame(parsed["queries"])
    if set(generated.id) != set(sample.id) or len(generated) != len(sample):
        raise RuntimeError("final query generator did not return every frozen ID exactly once")
    out = sample.merge(generated, on="id", validate="one_to_one").rename(
        columns={"id": "test_id"}
    )
    if (out["query"].str.split().str.len().gt(18).any()
            or not out["query"].str.endswith("?").all()):
        raise RuntimeError("final query QC failed length or question-mark constraint")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(QUERIES, index=False)
    return out


def run(limit: int | None = None, delay: float = 31.0):
    queries = prepare_queries()
    if limit:
        queries = queries.head(limit)
    pool = ids("pool")
    forbidden = ids("development") | ids("final_test")
    rows = pd.read_csv(SUMMARIES).to_dict("records") if SUMMARIES.exists() else []
    done = {(r["test_id"], r["config"]) for r in rows}
    total = len(queries) * len(CONFIGS)
    for item in queries.itertuples(index=False):
        for config in CONFIGS:
            if (item.test_id, config) in done:
                continue
            rec = summarize(
                item.query, config=config, k=5, alpha=0.75, tau=0.35,
                test_id=item.test_id, candidate_ids=pool, forbidden_ids=forbidden,
            )
            rec.update({"section": item.section, "category": item.category})
            rows.append(rec)
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(SUMMARIES, index=False)
            done.add((item.test_id, config))
            print(f"  {len(done)}/{total} {item.test_id} {config}", flush=True)
            if len(done) < total and delay:
                time.sleep(delay)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--delay", type=float, default=31.0)
    args = parser.parse_args()
    if args.prepare:
        print(prepare_queries().to_string(index=False))
    else:
        run(args.limit, args.delay)


if __name__ == "__main__":
    main()
