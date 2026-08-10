"""E12b — E12's frozen protocol run on the FULL 150-item final-test role.

E12 executed `configs/final_research.json` on 20 articles (four from each of five
category families) and produced 12 jointly usable pairs. At that n every interval
crossed zero, and M_vision-M flipped sign between the `usable` (+0.037, n=12) and
`nonhard` (-0.015, n=17) cuts — the signature of noise, not of an effect.

This module changes exactly one thing: the sample is now every article whose
research_role is `final_test`. Nothing is re-selected. k, alpha, tau, the prompt,
the generator, the decoding parameters and the judge are read from the same frozen
config, so this is E12 with more items, not a new experiment — hence E12b.

Two consequences worth stating in the write-up:

* **Different estimand.** E12's 20 were category-balanced (4 each), which
  over-weights `sports` (4 of 4 in the corpus) and under-weights
  `society_culture` (4 of 74). The full 150 is the population of the declared
  final-test role, so it estimates the average effect over that role rather than
  over a balanced sample. The category cuts stay descriptive, as declared.
* **The 20 are regenerated, not reused.** `data/llm_cache/` is gitignored and
  absent on a fresh clone, so their calls are re-billed. Their *prompts* are
  byte-identical though — verified against the committed E12 artifacts — so
  comparing the regenerated summaries with E12's gives a free determinism check
  on the generator across machines. See `--compare-e12`.
"""

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
from .research_final import QUERY_SCHEMA, category

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "experiments" / "E12b_full_final_sample"
E12_DIR = ROOT / "results" / "experiments" / "E12_final_comparison"
QUERIES = OUT_DIR / "queries.csv"
SUMMARIES = OUT_DIR / "summaries.csv"
CONFIG = json.loads((ROOT / "configs" / "final_research.json").read_text())
CONFIGS = tuple(CONFIG["systems"])          # B1, M, M_vision
K = CONFIG["retrieval"]["k"]                # 5
ALPHA = CONFIG["retrieval"]["alpha"]        # 0.75
TAU = CONFIG["retrieval"]["tau"]            # 0.35

# 130 new headlines at ~29 output tokens each would overrun the 3000-token cap of
# a single query-generation call, so they go out in batches.
QUERY_BATCH = 40

# Pacing is set by TOKENS, not requests. The measured limits on this project are
# 10,000 RPM but only 200,000 TPM, and a generation call carries ~6,205 tokens
# (five articles of evidence). Requests are therefore nowhere near their ceiling
# while tokens sit right on theirs: unpaced serial calls ran ~2 s apart, i.e.
# ~186k tokens/min, and the run died on a 429 after 235 rows. Deriving the gap
# from the token budget instead of guessing a delay keeps that from recurring.
TPM_LIMIT = 200_000
AVG_TOKENS_PER_CALL = 6_205
TPM_SAFETY = 0.80
CALL_GAP = AVG_TOKENS_PER_CALL / (TPM_LIMIT * TPM_SAFETY) * 60  # ~2.3 s
MAX_RATE_LIMIT_RETRIES = 6


def full_sample() -> pd.DataFrame:
    """Every final_test article. No sampling, therefore no selection to declare."""
    frame = articles()
    frame = frame[frame.research_role == "final_test"].copy()
    frame["category"] = frame.section.map(category)
    if len(frame) != 150 or not frame.id.is_unique:
        raise AssertionError(f"expected 150 unique final_test articles, got {len(frame)}")
    return frame[["id", "headline", "section", "category"]].reset_index(drop=True)


def _generate_queries(sample: pd.DataFrame) -> pd.DataFrame:
    """Ask the generator for one natural question per headline, in batches."""
    out = []
    for start in range(0, len(sample), QUERY_BATCH):
        chunk = sample.iloc[start : start + QUERY_BATCH]
        items = "\n".join(f"{r.id}\t{r.headline}" for r in chunk.itertuples(index=False))
        # Byte-identical wording to research_final.prepare_queries, so the 20
        # inherited items would have received exactly this instruction too.
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
            output_tokens=response.usage.completion_tokens,
            category="scaleup_query_generation", cache_key=key,
        )
        got = pd.DataFrame(parsed["queries"])
        if set(got.id) != set(chunk.id) or len(got) != len(chunk):
            raise RuntimeError(f"query generator dropped or invented IDs in batch {start}")
        out.append(got)
        print(f"  queries {min(start + QUERY_BATCH, len(sample))}/{len(sample)}", flush=True)
    return pd.concat(out, ignore_index=True)


def prepare_queries() -> pd.DataFrame:
    """Frozen E12 queries verbatim + one generated question per new article.

    The 20 inherited queries are copied, never regenerated. Regenerating them
    could return different wording, which would change their prompts and quietly
    stop them from being the same items E12 measured.
    """
    if QUERIES.exists():
        return pd.read_csv(QUERIES)
    sample = full_sample()
    inherited = pd.read_csv(E12_DIR / "queries.csv")[["test_id", "query"]].rename(
        columns={"test_id": "id"}
    )
    if not set(inherited.id) <= set(sample.id):
        raise AssertionError("E12 queries reference ids outside the final_test role")
    missing = sample[~sample.id.isin(inherited.id)]
    print(f"carrying over {len(inherited)} E12 queries, generating {len(missing)} new")
    generated = _generate_queries(missing) if len(missing) else pd.DataFrame(
        columns=["id", "query"]
    )
    every = pd.concat([inherited, generated], ignore_index=True)
    out = sample.merge(every, on="id", validate="one_to_one").rename(
        columns={"id": "test_id"}
    )
    if len(out) != len(sample):
        raise RuntimeError("query table does not cover every final_test article once")
    # Same QC gate E12 applied. Inherited rows already passed it; new ones must too.
    if (out["query"].str.split().str.len().gt(18).any()
            or not out["query"].str.endswith("?").all()):
        bad = out[out["query"].str.split().str.len().gt(18)
                  | ~out["query"].str.endswith("?")]
        raise RuntimeError(f"query QC failed for {len(bad)} rows:\n{bad.to_string()}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(QUERIES, index=False)
    return out


def _summarize_paced(**kw) -> dict:
    """summarize() with backoff on 429.

    The SDK retries twice by default, which is not enough when the whole run sits
    at the token ceiling: a sustained overshoot exhausts them and kills the run
    mid-way. Retrying here means a transient limit costs seconds, not the job.
    """
    import openai

    for attempt in range(MAX_RATE_LIMIT_RETRIES):
        try:
            return summarize(**kw)
        except openai.RateLimitError:
            wait = CALL_GAP * (2 ** attempt)
            print(f"    429, backing off {wait:.1f}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"still rate limited after {MAX_RATE_LIMIT_RETRIES} retries")


def run(limit: int | None = None, delay: float | None = None) -> pd.DataFrame:
    """Generate every (item, arm) into the E12b directory. Resumable.

    `delay` defaults to CALL_GAP, derived from the measured token budget. The 31 s
    gap in research_final was sized for a 2 RPM limit that no longer applies, but
    dropping it to zero overshoots TPM instead — see the constants above.
    """
    if delay is None:
        delay = CALL_GAP
    queries = prepare_queries()
    if limit:
        queries = queries.head(limit)
    pool = ids("pool")
    forbidden = ids("development") | ids("final_test")
    rows = pd.read_csv(SUMMARIES).to_dict("records") if SUMMARIES.exists() else []
    done = {(r["test_id"], r["config"]) for r in rows}
    total = len(queries) * len(CONFIGS)
    t0 = time.time()
    for item in queries.itertuples(index=False):
        for config in CONFIGS:
            if (item.test_id, config) in done:
                continue
            api_budget.assert_can_spend(estimated_max_usd=0.05)
            rec = _summarize_paced(
                query=item.query, config=config, k=K, alpha=ALPHA, tau=TAU,
                test_id=item.test_id, candidate_ids=pool, forbidden_ids=forbidden,
            )
            rec.update({"section": item.section, "category": item.category})
            rows.append(rec)
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(SUMMARIES, index=False)
            done.add((item.test_id, config))
            print(f"  {len(done)}/{total} {item.test_id} {config} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if delay:
                time.sleep(delay)
    return pd.DataFrame(rows)


def compare_e12() -> pd.DataFrame:
    """Free determinism check: did the 20 overlapping items regenerate identically?

    Retrieval was already verified bit-identical to the committed macOS artifacts,
    so these 20 received byte-identical prompts. Any difference here is the
    generator, not the pipeline.
    """
    new = pd.read_csv(SUMMARIES).fillna({"evidence": "", "summary": ""})
    old = pd.read_csv(E12_DIR / "summaries.csv").fillna({"evidence": "", "summary": ""})

    def norm(s):
        return str(s).replace("\r\n", "\n").replace("\r", "\n")

    merged = old.merge(new, on=["test_id", "config"], suffixes=("_e12", "_e12b"))
    merged["evidence_same"] = [norm(a) == norm(b) for a, b in
                               zip(merged.evidence_e12, merged.evidence_e12b)]
    merged["summary_same"] = [norm(a) == norm(b) for a, b in
                              zip(merged.summary_e12, merged.summary_e12b)]
    print(f"overlapping rows: {len(merged)}")
    print(f"  evidence identical : {int(merged.evidence_same.sum())}/{len(merged)}")
    print(f"  summary  identical : {int(merged.summary_same.sum())}/{len(merged)}")
    diff = merged[~merged.summary_same]
    if len(diff):
        print("\nitems whose summary changed:")
        for r in diff.itertuples(index=False):
            print(f"  {r.test_id} {r.config}")
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="build queries only")
    parser.add_argument("--limit", type=int, help="first N items (smoke test)")
    parser.add_argument("--delay", type=float, default=None,
                        help=f"seconds between calls (default {CALL_GAP:.1f}, from the "
                             f"{TPM_LIMIT:,} TPM budget)")
    parser.add_argument("--compare-e12", action="store_true",
                        help="diff the 20 overlapping items against E12")
    args = parser.parse_args()
    if args.compare_e12:
        compare_e12()
    elif args.prepare:
        print(prepare_queries().to_string(index=False))
    else:
        run(args.limit, args.delay)
        print(json.dumps(api_budget.summary(), indent=2))


if __name__ == "__main__":
    main()
