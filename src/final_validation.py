"""Export and score the frozen judge's blind 50-claim human validation set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from .evaluate import refusal_kind

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "results" / "experiments"
# E12 stays the default so the committed E13a artifacts keep resolving. E12b is
# passed explicitly by E13b, whose conclusion is the one that now needs a human
# check: validating the judge on E12's superseded claims would measure its
# reliability on data no result depends on.
BASE = EXPERIMENTS / "E12_final_comparison"


def _paths(base: Path) -> tuple[Path, Path, Path]:
    return (base / "evaluation" / "claims.csv", base / "summaries.csv",
            base / "human_validation_50.csv")


CLAIMS, SUMMARIES, SAMPLE = _paths(BASE)


def _sample_id(row) -> str:
    text = f"{row.test_id}|{row.config}|{row.claim_index}|{row.claim}"
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _eligible(base: Path = BASE) -> pd.DataFrame:
    claims_path, summaries_path, _ = _paths(base)
    claims = pd.read_csv(claims_path)
    summaries = pd.read_csv(summaries_path).fillna({"evidence": "", "image_paths": ""})
    summaries["kind"] = summaries.summary.map(refusal_kind)
    usable = summaries[summaries.kind == "usable"]
    frame = claims.merge(
        usable[["test_id", "config", "query", "evidence", "image_paths"]],
        on=["test_id", "config"], validate="many_to_one",
    )
    frame["sample_id"] = frame.apply(_sample_id, axis=1)
    return frame


def export(base: Path = BASE) -> pd.DataFrame:
    _, _, sample_path = _paths(base)
    if sample_path.exists():
        return pd.read_csv(sample_path)
    frame = _eligible(base).sample(n=50, random_state=42)
    # System label and automated verdict are deliberately absent. image_paths is
    # populated only where pixels were valid judge evidence; this cannot be hidden
    # without asking the human to judge against evidence the system never received.
    blind = frame[["sample_id", "query", "claim", "evidence", "image_paths"]].copy()
    blind["human_supported"] = ""
    blind["human_modality"] = ""
    blind["notes"] = ""
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    blind.to_csv(sample_path, index=False)
    return blind


def validate(base: Path = BASE) -> dict:
    _, _, sample_path = _paths(base)
    human = pd.read_csv(sample_path).fillna("")
    human["human_supported"] = human.human_supported.astype(str).str.strip().str.lower()
    if not human.human_supported.isin(["y", "n"]).all():
        missing = int((~human.human_supported.isin(["y", "n"])).sum())
        raise SystemExit(f"{missing}/50 human_supported cells still need y or n")
    key = _eligible(base)[["sample_id", "supported", "support"]]
    scored = human.merge(key, on="sample_id", validate="one_to_one")
    y_true = scored.human_supported.eq("y")
    y_pred = scored.supported.astype(bool)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
    result = {
        "n": len(scored),
        "agreement": float((y_true == y_pred).mean()),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_human_rows_judge_columns": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        },
    }
    modality = scored.human_modality.astype(str).str.strip().str.lower()
    valid = modality.isin(["text_only", "image_only", "both", "unsupported"])
    if valid.any():
        result["modality_labels_n"] = int(valid.sum())
        result["modality_agreement"] = float((modality[valid] == scored.support[valid]).mean())
    result["source"] = base.name
    path = base / "evaluation" / "human_validation_metrics.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--base", type=Path, default=BASE,
                        help="experiment directory (default E12_final_comparison)")
    args = parser.parse_args()
    base = args.base if args.base.is_absolute() else EXPERIMENTS / args.base.name
    if args.validate:
        validate(base)
    else:
        print(export(base).to_string(index=False))


if __name__ == "__main__":
    main()
