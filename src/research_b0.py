"""B0 (no-retrieval) arm added to the frozen final-test role, OpenAI-only.

Generates B0 with gpt-4o-mini on the identical 150 final-test queries E12b used
for B1/M/M_vision (results/experiments/E12b_full_final_sample/queries.csv), so
this is a paired addition to that run rather than a reuse of the old baseline's
B0 (which was scored on a different, non-group-safe item set entirely).

Judged with gpt-5.6-luna (research_evaluation.MODEL) -- the same judge already
scoring B1/M/M_vision in E12b -- against the union of B1's and M's evidence for
each item, mirroring D14's rule for the original baseline ("B0's claims are
judged against the union of B1's and M's evidence"). B0 and B1/M/M_vision are
now on one consistent generator, judge, and final-test item set.

    python -m src.research_b0 --generate   # 150 gpt-4o-mini calls, no retrieval
    python -m src.research_b0 --judge       # decompose+verify vs union evidence
    python -m src.research_b0 --report      # usable-cut faithfulness + paired diff
    python -m src.research_b0 --limit 3     # smoke test any of the above
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import api_budget
from .evaluate import DECOMPOSE_SYSTEM, refusal_kind
from .research_data import ids
from .research_evaluation import (
    DECOMPOSE_SCHEMA, TEXT_VERIFY_SCHEMA, MODEL, _call, _normalize_item_id,
)
from .research_scaleup import CALL_GAP, K, ALPHA, TAU, _summarize_paced

ROOT = Path(__file__).resolve().parents[1]
E12B_DIR = ROOT / "results" / "experiments" / "E12b_full_final_sample"
QUERIES = E12B_DIR / "queries.csv"
B1M_SOURCE = E12B_DIR / "summaries.csv"

OUT_DIR = ROOT / "results" / "experiments" / "E14_b0_openai"
SUMMARIES = OUT_DIR / "summaries.csv"
EVAL_DIR = OUT_DIR / "evaluation"
CLAIMS = EVAL_DIR / "claims.csv"


# --------------------------------------------------------------- generation

def generate(limit: int | None = None) -> pd.DataFrame:
    queries = pd.read_csv(QUERIES)
    if limit:
        queries = queries.head(limit)
    pool = ids("pool")
    forbidden = ids("development") | ids("final_test")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(SUMMARIES).to_dict("records") if SUMMARIES.exists() else []
    done = {r["test_id"] for r in rows}
    total = len(queries)
    for n, item in enumerate(queries.itertuples(index=False), 1):
        if item.test_id in done:
            continue
        api_budget.assert_can_spend(estimated_max_usd=0.02)
        rec = _summarize_paced(
            query=item.query, config="B0", k=K, alpha=ALPHA, tau=TAU,
            test_id=item.test_id, candidate_ids=pool, forbidden_ids=forbidden,
        )
        rec.update({"section": item.section, "category": item.category})
        rows.append(rec)
        pd.DataFrame(rows).to_csv(SUMMARIES, index=False)
        done.add(item.test_id)
        print(f"  generate {n}/{total} {item.test_id}", flush=True)
        if n < total:
            time.sleep(CALL_GAP)
    return pd.DataFrame(rows)


# -------------------------------------------------------------------- judge

def _union_evidence(b1m: pd.DataFrame, test_id: str) -> str:
    rows = b1m[b1m.test_id == test_id]
    parts = []
    for config in ("B1", "M"):
        row = rows[rows.config == config]
        if len(row):
            parts.append(f"--- evidence from {config} retrieval ---\n{row.iloc[0].evidence}")
    return "\n\n".join(parts)


def _judge_one(test_id: str, summary: str, evidence: str) -> list[dict]:
    decompose_prompt = (
        "Split the summary below into atomic, self-contained factual claims.\n"
        "Do not assess truth and do not use outside knowledge. Preserve the "
        "item_id exactly.\n\nITEM A\nSUMMARY:\n" + summary
    )
    decomposed = _call(DECOMPOSE_SYSTEM, decompose_prompt, DECOMPOSE_SCHEMA, "b0_decompose")
    items = {_normalize_item_id(x["item_id"]): [c.strip() for c in x["claims"] if c.strip()]
             for x in decomposed["items"]}
    claims = items.get("A", [])
    if not claims:
        return []
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, 1))
    verify_prompt = (
        "For every claim, classify support using only the supplied evidence. "
        "`text_only` means the evidence supports it; `unsupported` includes "
        "partial, contradicted, or absent support. Never use outside knowledge. "
        "Give a reason of at most 15 words. Return one verdict per numbered claim.\n\n"
        f"ITEM A\nEVIDENCE:\n{evidence}\nCLAIMS:\n{numbered}"
    )
    verified = _call(
        "You verify factual claims against only the supplied evidence.",
        verify_prompt, TEXT_VERIFY_SCHEMA, "b0_verify",
    )
    by_item = {_normalize_item_id(x["item_id"]): x["verdicts"] for x in verified["items"]}
    by_index = {int(v["claim_index"]): v for v in by_item.get("A", [])}
    out = []
    for i, claim in enumerate(claims, 1):
        v = by_index.get(i, {"support": "unsupported", "reason": "NO_VERDICT_RETURNED"})
        out.append({
            "test_id": test_id, "config": "B0", "claim_index": i, "claim": claim,
            "support": v["support"], "supported": v["support"] != "unsupported",
            "reason": v["reason"],
        })
    return out


def judge(limit: int | None = None) -> pd.DataFrame:
    if not SUMMARIES.exists():
        raise RuntimeError("run --generate first")
    b0 = pd.read_csv(SUMMARIES).fillna({"evidence": "", "summary": ""})
    b1m = pd.read_csv(B1M_SOURCE).fillna({"evidence": "", "summary": ""})
    if limit:
        b0 = b0.head(limit)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(CLAIMS).to_dict("records") if CLAIMS.exists() else []
    done = {r["test_id"] for r in rows}
    test_ids = [t for t in b0.test_id if t not in done]
    total = len(b0)
    for n, test_id in enumerate(b0.test_id, 1):
        if test_id in done:
            continue
        summary = b0.loc[b0.test_id == test_id, "summary"].iloc[0]
        if refusal_kind(summary) == "hard":
            done.add(test_id)
            print(f"  judge {n}/{total} {test_id} (hard refusal, skipped)", flush=True)
            continue
        api_budget.assert_can_spend(estimated_max_usd=0.15)
        evidence = _union_evidence(b1m, test_id)
        rows.extend(_judge_one(test_id, summary, evidence))
        pd.DataFrame(rows).to_csv(CLAIMS, index=False)
        done.add(test_id)
        print(f"  judge {n}/{total} {test_id}, {len(rows)} claims so far", flush=True)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- report

def _bootstrap(values: np.ndarray, seed: int = 42):
    if not len(values):
        return [None, None]
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def report() -> dict:
    b0 = pd.read_csv(SUMMARIES).fillna({"summary": ""})
    claims = pd.read_csv(CLAIMS) if CLAIMS.exists() else pd.DataFrame(
        columns=["test_id", "config", "supported"]
    )
    b0["kind"] = b0.summary.map(refusal_kind)
    per_item = claims.groupby("test_id").supported.mean().rename("faithfulness").reset_index()
    per_item = per_item.merge(b0[["test_id", "kind"]], on="test_id", how="right")

    usable = per_item[per_item.kind == "usable"]
    b0_faithfulness = float(usable.faithfulness.mean()) if len(usable) else None
    refusal_counts = b0.kind.value_counts().to_dict()

    # Paired B0-vs-B1 now that both share generator, judge, and item set.
    b1_claims_path = E12B_DIR / "evaluation" / "per_item.csv"
    paired = None
    if b1_claims_path.exists():
        b1_per_item = pd.read_csv(b1_claims_path)
        b1_usable = b1_per_item[(b1_per_item.config == "B1") & (b1_per_item.kind == "usable")]
        wide = usable[["test_id", "faithfulness"]].merge(
            b1_usable[["test_id", "faithfulness"]], on="test_id",
            suffixes=("_B0", "_B1"), how="inner",
        )
        diff = (wide.faithfulness_B1 - wide.faithfulness_B0).to_numpy()
        lo, hi = _bootstrap(diff)
        paired = {"n": len(diff), "mean_difference_B1_minus_B0": float(diff.mean()) if len(diff) else None,
                  "ci_low": lo, "ci_high": hi}

    diagnostics = {
        "experiment_id": "E14",
        "generator": "gpt-4o-mini",
        "judge": MODEL,
        "items": int(b0.test_id.nunique()),
        "usable_count": int(len(usable)),
        "refusal_counts": {str(k): int(v) for k, v in refusal_counts.items()},
        "b0_faithfulness_usable": b0_faithfulness,
        "paired_B1_minus_B0_usable": paired,
        "budget": api_budget.summary(),
    }
    per_item.to_csv(EVAL_DIR / "per_item.csv", index=False)
    (EVAL_DIR / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(json.dumps(diagnostics, indent=2))
    return diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not (args.generate or args.judge or args.report):
        args.generate = args.judge = args.report = True
    if args.generate:
        generate(args.limit)
    if args.judge:
        judge(args.limit)
    if args.report:
        report()


if __name__ == "__main__":
    main()
