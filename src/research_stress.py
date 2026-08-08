"""Wrong-image stress test on cases where the clean vision judge used pixels."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .generate import INSTRUCTIONS, _image_data_url, generate
from .research_evaluation import (
    DECOMPOSE_SCHEMA,
    VERIFY_SCHEMA,
    _bootstrap,
    _call,
    _normalize_item_id,
)

ROOT = Path(__file__).resolve().parents[1]
GENERATIONS = ROOT / "results" / "experiments" / "E09_development_generation" / "summaries.csv"
CLAIMS = ROOT / "results" / "experiments" / "E10_development_claim_evaluation" / "claims.csv"
PER_ITEM = ROOT / "results" / "experiments" / "E10_development_claim_evaluation" / "per_item.csv"
DEFAULT_OUT = ROOT / "results" / "experiments" / "E11_wrong_image_stress"


def _sample():
    claims = pd.read_csv(CLAIMS)
    ids = list(dict.fromkeys(claims[(claims.config == "M_vision") &
                                     claims.support.isin(["image_only", "both"])].test_id))
    clean = pd.read_csv(GENERATIONS)
    clean = clean[(clean.config == "M_vision") & clean.test_id.isin(ids)].copy()
    return clean.set_index("test_id").loc[ids].reset_index()


def _corrupted_content(row, wrong_paths: list[str]):
    prompt = f"{INSTRUCTIONS}\nQUERY: {row.query}\n\nEVIDENCE:\n{row.evidence}\n"
    content = [{"type": "text", "text": prompt}]
    for i, path in enumerate(wrong_paths, 1):
        content.extend([
            {"type": "text", "text": f"IMAGE {i} corresponds to evidence article [{i}]."},
            {"type": "image_url", "image_url": {
                "url": _image_data_url(path), "detail": "low",
            }},
        ])
    return content


def generate_corruptions(out: Path, delay: float = 31.0):
    clean = _sample()
    path = out / "summaries.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {r["test_id"] for r in rows}
    for i, row in enumerate(clean.itertuples(index=False)):
        if row.test_id in done:
            continue
        donor = clean.iloc[(i + 1) % len(clean)]
        wrong_paths = str(donor.image_paths).split("|")
        summary = generate(_corrupted_content(row, wrong_paths))
        rows.append({
            "test_id": row.test_id, "query": row.query,
            "config": "M_vision_wrong", "wrong_image_from": donor.test_id,
            "evidence": row.evidence, "summary": summary,
            "clean_image_paths": row.image_paths,
            "wrong_image_paths": "|".join(wrong_paths),
        })
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"  generated {len(rows)}/{len(clean)}", flush=True)
        if len(rows) < len(clean) and delay:
            time.sleep(delay)
    return pd.DataFrame(rows)


def _decompose_all(rows: pd.DataFrame):
    labels = {tid: chr(65 + i) for i, tid in enumerate(rows.test_id)}
    sections = [f"ITEM {labels[r.test_id]}\nSUMMARY:\n{r.summary}"
                for r in rows.itertuples(index=False)]
    prompt = """Split every summary below into atomic, self-contained factual claims.
Do not assess truth and do not use outside knowledge. Preserve each opaque item_id.
Return exactly one item for each supplied item.\n\n""" + "\n\n".join(sections)
    result = _call(
        "You split news summaries into atomic factual claims without assessing truth.",
        prompt, DECOMPOSE_SCHEMA, "stress_decompose",
    )
    by_label = {_normalize_item_id(x["item_id"]): x["claims"] for x in result["items"]}
    return labels, {tid: by_label[label] for tid, label in labels.items()}


def _verify_clean(row, item_claims: list[str]):
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(item_claims, 1))
    text = f"""Judge each claim against the CLEAN evidence bundle below. Text and images
are both allowed. Classify support as text_only, image_only, both, or unsupported.
Do not use outside knowledge. Return one verdict for ITEM A per numbered claim.

ITEM A
EVIDENCE:
{row.evidence}
CLAIMS:
{numbered}
"""
    content = [{"type": "text", "text": text}]
    for i, path in enumerate(str(row.clean_image_paths).split("|"), 1):
        content.extend([
            {"type": "text", "text": f"ITEM A CLEAN IMAGE I{i}"},
            {"type": "image_url", "image_url": {
                "url": _image_data_url(path), "detail": "low",
            }},
        ])
    result = _call(
        "You verify factual claims against only the supplied clean text and images.",
        content, VERIFY_SCHEMA, "stress_verify_clean",
    )
    item = result["items"][0]
    return {int(v["claim_index"]): v for v in item["verdicts"]}


def evaluate(rows: pd.DataFrame, out: Path):
    labels, claims = _decompose_all(rows)
    records = []
    for number, row in enumerate(rows.itertuples(index=False), 1):
        verdicts = _verify_clean(row, claims[row.test_id])
        for i, claim in enumerate(claims[row.test_id], 1):
            v = verdicts.get(i, {"support": "unsupported", "reason": "NO_VERDICT"})
            records.append({
                "test_id": row.test_id, "claim_index": i, "claim": claim,
                "support": v["support"], "supported": v["support"] != "unsupported",
                "reason": v["reason"],
            })
        pd.DataFrame(records).to_csv(out / "claims.csv", index=False)
        print(f"  verified {number}/{len(rows)}", flush=True)
    judged = pd.DataFrame(records)
    wrong = judged.groupby("test_id").supported.mean().rename("wrong_faithfulness")
    correct = pd.read_csv(PER_ITEM)
    correct = correct[correct.config == "M_vision"].set_index("test_id").faithfulness
    paired = pd.concat([correct.rename("correct_faithfulness"), wrong], axis=1).dropna()
    paired["difference"] = paired.wrong_faithfulness - paired.correct_faithfulness
    paired.to_csv(out / "paired.csv")
    lo, hi = _bootstrap(paired.difference.to_numpy())
    diagnostics = {
        "experiment_id": "E11",
        "split": "development visual-support cases",
        "cases": len(paired),
        "correct_mean": float(paired.correct_faithfulness.mean()),
        "wrong_image_mean": float(paired.wrong_faithfulness.mean()),
        "wrong_minus_correct": float(paired.difference.mean()),
        "ci_low": lo, "ci_high": hi,
        "unsupported_claim_rate_wrong_image": float((~judged.supported).mean()),
    }
    (out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(json.dumps(diagnostics, indent=2))


def run(out: Path = DEFAULT_OUT, delay: float = 31.0):
    out.mkdir(parents=True, exist_ok=True)
    rows = generate_corruptions(out, delay)
    if delay:
        time.sleep(delay)
    evaluate(rows, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--delay", type=float, default=31.0)
    args = parser.parse_args()
    run(args.out, args.delay)


if __name__ == "__main__":
    main()
