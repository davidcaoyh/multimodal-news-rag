"""Batched blind claim evaluation with image-aware support attribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import api_budget
from .evaluate import DECOMPOSE_SYSTEM, refusal_kind
from .generate import _cache_safe_content, _get_client, _image_data_url

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "results" / "experiments" / "E09_development_generation" / "summaries.csv"
DEFAULT_OUT = ROOT / "results" / "experiments" / "E10_development_claim_evaluation"
MODEL = "gpt-5.6-luna"
MAX_TOKENS = 6000
CONFIGS = ("B1", "M_nocap", "M", "M_vision")
MIN_API_INTERVAL = 31.0
_last_api_call = 0.0

DECOMPOSE_SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "item_id": {"type": "string"},
            "claims": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["item_id", "claims"],
        "additionalProperties": False,
    }}},
    "required": ["items"],
    "additionalProperties": False,
}

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "item_id": {"type": "string"},
            "verdicts": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "claim_index": {"type": "integer"},
                    "support": {"type": "string", "enum": [
                        "text_only", "image_only", "both", "unsupported"
                    ]},
                    "reason": {"type": "string"},
                },
                "required": ["claim_index", "support", "reason"],
                "additionalProperties": False,
            }},
        },
        "required": ["item_id", "verdicts"],
        "additionalProperties": False,
    }}},
    "required": ["items"],
    "additionalProperties": False,
}

# A physically text-only request cannot attribute support to pixels. A narrower
# schema prevents caption markers such as ``[IMAGE 2: "..."]`` from being
# misclassified as visual evidence merely because their textual label says IMAGE.
TEXT_VERIFY_SCHEMA = json.loads(json.dumps(VERIFY_SCHEMA))
TEXT_VERIFY_SCHEMA["properties"]["items"]["items"]["properties"]["verdicts"][
    "items"
]["properties"]["support"]["enum"] = ["text_only", "unsupported"]


def _call(system: str, content, schema: dict, category: str):
    global _last_api_call
    safe = _cache_safe_content(content)
    blob = json.dumps([MODEL, system, safe, schema], sort_keys=True)
    key = hashlib.sha256(blob.encode()).hexdigest()[:24]
    cache = ROOT / "data" / "llm_cache" / f"research_judge_{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())["response"]
    api_budget.assert_can_spend(estimated_max_usd=0.15)
    remaining = MIN_API_INTERVAL - (time.monotonic() - _last_api_call)
    if remaining > 0:
        time.sleep(remaining)
    response = _get_client().chat.completions.create(
        model=MODEL,
        max_completion_tokens=MAX_TOKENS,
        reasoning_effort="none",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": content}],
        response_format={"type": "json_schema", "json_schema": {
            "name": "research_judge", "strict": True, "schema": schema,
        }},
    )
    _last_api_call = time.monotonic()
    if response.choices[0].finish_reason == "length":
        raise RuntimeError("research judge reached its output-token limit")
    parsed = json.loads(response.choices[0].message.content or "")
    usage = api_budget.record(
        model=MODEL,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        category=category,
        cache_key=cache.stem,
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({
        "model": MODEL, "system": system, "content": safe, "response": parsed,
        "usage": {"in": response.usage.prompt_tokens,
                  "out": response.usage.completion_tokens},
        "estimated_cost_usd": usage["estimated_cost_usd"],
    }, indent=2))
    return parsed


def _opaque_map(test_id: str) -> dict[str, str]:
    labels = list("ABCD")
    random.Random(int(test_id, 16)).shuffle(labels)
    return dict(zip(CONFIGS, labels))


def _normalize_item_id(value: str) -> str:
    value = str(value).strip().upper()
    return value.removeprefix("ITEM ").strip()


def _decompose(rows: pd.DataFrame, labels: dict[str, str]):
    sections = []
    for config in CONFIGS:
        summary = rows.loc[rows.config == config, "summary"].iloc[0]
        sections.append(f"ITEM {labels[config]}\nSUMMARY:\n{summary}")
    prompt = """Split every summary below into atomic, self-contained factual claims.
Do not assess truth and do not use outside knowledge. Preserve each opaque item_id.
Return exactly one item for each supplied item.\n\n""" + "\n\n".join(sections)
    result = _call(DECOMPOSE_SYSTEM, prompt, DECOMPOSE_SCHEMA, "research_decompose")
    return {_normalize_item_id(x["item_id"]): [c.strip() for c in x["claims"] if c.strip()]
            for x in result["items"]}


def _verification_content(rows: pd.DataFrame, labels: dict[str, str], claims: dict,
                          configs: tuple[str, ...], include_images: bool = False):
    sections = []
    for config in configs:
        row = rows[rows.config == config].iloc[0]
        numbered = "\n".join(
            f"{i}. {claim}" for i, claim in enumerate(claims[labels[config]], 1)
        ) or "(no claims)"
        allowance = ("This item may use its labeled images as well as its text."
                     if include_images else "This item may use text evidence only.")
        sections.append(
            f"ITEM {labels[config]}\n{allowance}\nEVIDENCE:\n{row.evidence}\n"
            f"CLAIMS:\n{numbered}"
        )
    instructions = """For every claim, classify support using only that item's allowed
evidence. `text_only` means text/caption support; `image_only` means pixels support it;
`both` means each modality independently supports it; `unsupported` includes partial,
contradicted, or absent support. Never use outside knowledge. Give a reason of at most
15 words. Return every opaque item_id and one verdict per numbered claim.\n\n"""
    content = [{"type": "text", "text": instructions + "\n\n".join(sections)}]
    if include_images:
        vision_label = labels["M_vision"]
        vision = rows[rows.config == "M_vision"].iloc[0]
        for i, path in enumerate(str(vision.image_paths).split("|"), 1):
            if not path:
                continue
            content.extend([
                {"type": "text", "text": f"ITEM {vision_label} IMAGE I{i}"},
                {"type": "image_url", "image_url": {
                    "url": _image_data_url(path), "detail": "low",
                }},
            ])
        return content
    # No image blocks are even present in the non-vision verification request.
    return content[0]["text"]


def _evaluate_case(rows: pd.DataFrame) -> list[dict]:
    test_id = rows.test_id.iloc[0]
    labels = _opaque_map(test_id)
    claims = _decompose(rows, labels)
    if set(claims) != set(labels.values()):
        raise RuntimeError(f"decomposer item mismatch for {test_id}: {set(claims)}")
    text_configs = ("B1", "M_nocap", "M")
    text_content = _verification_content(rows, labels, claims, text_configs)
    text_judged = _call(
        "You verify factual claims against only the supplied evidence.",
        text_content, TEXT_VERIFY_SCHEMA, "research_verify_text",
    )
    vision_content = _verification_content(
        rows, labels, claims, ("M_vision",), include_images=True
    )
    vision_judged = _call(
        "You verify factual claims against only the supplied text and images.",
        vision_content, VERIFY_SCHEMA, "research_verify_vision",
    )
    verdicts = {_normalize_item_id(x["item_id"]): x["verdicts"]
                for x in [*text_judged["items"], *vision_judged["items"]]}
    reverse = {label: config for config, label in labels.items()}
    out = []
    for label, item_claims in claims.items():
        by_index = {int(v["claim_index"]): v for v in verdicts.get(label, [])}
        config = reverse[label]
        for i, claim in enumerate(item_claims, 1):
            verdict = by_index.get(i, {
                "support": "unsupported", "reason": "NO_VERDICT_RETURNED"
            })
            support = verdict["support"]
            if config != "M_vision" and support in {"image_only", "both"}:
                raise RuntimeError(f"judge used disallowed image evidence for {test_id}/{config}")
            out.append({
                "test_id": test_id, "config": config, "claim_index": i,
                "claim": claim, "support": support,
                "supported": support != "unsupported", "reason": verdict["reason"],
            })
    return out


def _bootstrap(values: np.ndarray, seed: int = 42):
    if not len(values):
        return [None, None]
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _report(claims: pd.DataFrame, summaries: pd.DataFrame, out: Path):
    per_item = claims.groupby(["test_id", "config"]).supported.mean().rename(
        "faithfulness"
    ).reset_index()
    kinds = summaries[["test_id", "config", "summary"]].copy()
    kinds["kind"] = kinds.summary.map(refusal_kind)
    per_item = per_item.merge(kinds[["test_id", "config", "kind"]],
                              on=["test_id", "config"], validate="one_to_one")
    cuts = []
    for cut, allowed in (("nonhard", {"usable", "soft"}), ("usable", {"usable"})):
        part = per_item[per_item.kind.isin(allowed)]
        agg = part.groupby("config").faithfulness.agg(["count", "mean"]).reset_index()
        agg.insert(0, "cut", cut)
        cuts.append(agg)
    summary = pd.concat(cuts, ignore_index=True)
    pairs = []
    for cut, allowed in (("nonhard", {"usable", "soft"}), ("usable", {"usable"})):
        eligible = per_item[per_item.kind.isin(allowed)]
        wide = eligible.pivot(index="test_id", columns="config", values="faithfulness")
        for left, right in [("B1", "M"), ("M_nocap", "M"), ("M", "M_vision")]:
            valid = wide[[left, right]].dropna()
            diff = (valid[right] - valid[left]).to_numpy()
            lo, hi = _bootstrap(diff)
            pairs.append({"cut": cut, "comparison": f"{right}-{left}", "n": len(diff),
                          "mean_difference": float(diff.mean()),
                          "ci_low": lo, "ci_high": hi})
    usable_vision_ids = set(kinds[(kinds.config == "M_vision") & (kinds.kind == "usable")].test_id)
    vision = claims[(claims.config == "M_vision") & claims.supported
                    & claims.test_id.isin(usable_vision_ids)]
    visual_rate = float(vision.support.isin(["image_only", "both"]).mean()) if len(vision) else 0.0
    diagnostics = {
        "experiment_id": "E10",
        "split": "development diagnostic sample",
        "items": int(summaries.test_id.nunique()),
        "summaries": len(summaries),
        "claims": len(claims),
        "visual_contribution_rate": visual_rate,
        "refusal_counts": summaries.assign(kind=summaries.summary.map(refusal_kind))
                                   .groupby(["config", "kind"]).size()
                                   .rename("n").reset_index().to_dict("records"),
        "budget": api_budget.summary(),
    }
    summary.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(pairs).to_csv(out / "paired_differences.csv", index=False)
    per_item.to_csv(out / "per_item.csv", index=False)
    (out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(pd.DataFrame(pairs).to_string(index=False))
    print(json.dumps(diagnostics, indent=2))


def run(source: Path = DEFAULT_IN, out: Path = DEFAULT_OUT, delay: float = 31.0,
        limit: int | None = None):
    summaries = pd.read_csv(source).fillna({"evidence": "", "image_paths": ""})
    out.mkdir(parents=True, exist_ok=True)
    claims_path = out / "claims.csv"
    existing = pd.read_csv(claims_path) if claims_path.exists() else pd.DataFrame()
    completed = set(existing.test_id) if len(existing) else set()
    all_rows = existing.to_dict("records") if len(existing) else []
    test_ids = list(dict.fromkeys(summaries.test_id))
    if limit:
        test_ids = test_ids[:limit]
        summaries = summaries[summaries.test_id.isin(test_ids)].copy()
    for number, test_id in enumerate(test_ids, 1):
        if test_id in completed:
            continue
        rows = summaries[summaries.test_id == test_id]
        if set(rows.config) != set(CONFIGS):
            raise RuntimeError(f"incomplete four-arm case {test_id}")
        all_rows.extend(_evaluate_case(rows))
        pd.DataFrame(all_rows).to_csv(claims_path, index=False)
        print(f"  {number}/{len(test_ids)} cases, {len(all_rows)} claims", flush=True)
        if number < len(test_ids) and delay:
            time.sleep(delay)
    claims = pd.DataFrame(all_rows)
    _report(claims, summaries, out)
    return claims


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--delay", type=float, default=31.0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run(args.source, args.out, args.delay, args.limit)


if __name__ == "__main__":
    main()
