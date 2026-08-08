"""Benchmark local vision generators on development-only BBC images.

The task is forced-choice caption discrimination. Each example contains the
article's associated image, its true caption, and a randomly sampled caption
from a different section. Correct-option order is randomized. This is a noisy
proxy (BBC captions are story summaries, not literal alt text), so it is used to
screen runtime/model feasibility rather than as a final project metric.

    python -m src.benchmark_local_models
    python -m src.benchmark_local_models --limit 4 --models qwen3-vl:4b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .local_model import chat

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "data.parquet"
MANIFEST = ROOT / "data" / "research" / "split_manifest.parquet"
DEFAULT_OUT = ROOT / "results" / "experiments" / "E06_local_model_benchmark"
DEFAULT_MODELS = ("qwen3-vl:4b", "qwen3-vl:8b", "gemma3:12b")
SCHEMA = {
    "type": "object",
    "properties": {
        "choice": {"type": "string", "enum": ["A", "B"]},
        "visible_evidence": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["choice", "visible_evidence", "confidence"],
}


def build_examples(limit: int, seed: int) -> list[dict]:
    data = pd.read_parquet(DATA)
    roles = pd.read_parquet(MANIFEST)
    dev = data.merge(roles, on="id").query("research_role == 'development'").copy()
    dev = dev[
        dev.caption.fillna("").str.len().ge(20)
        & dev.image_path.fillna("").map(lambda p: (ROOT / p).exists())
    ]
    if len(dev) < limit:
        raise ValueError(f"only {len(dev)} eligible development rows for limit={limit}")

    rng = np.random.default_rng(seed)
    sampled = dev.iloc[rng.choice(len(dev), size=limit, replace=False)]
    examples = []
    for row in sampled.itertuples(index=False):
        candidates = dev[(dev.id != row.id) & (dev.section != row.section)]
        distractor = candidates.iloc[int(rng.integers(len(candidates)))]
        correct = "A" if int(rng.integers(2)) == 0 else "B"
        options = {
            correct: row.caption.strip(),
            "B" if correct == "A" else "A": distractor.caption.strip(),
        }
        examples.append({
            "id": row.id,
            "image_path": row.image_path,
            "section": row.section,
            "correct": correct,
            "caption_a": options["A"],
            "caption_b": options["B"],
        })
    return examples


def prompt_for(example: dict) -> str:
    return f"""You are matching a news photograph to its associated BBC story caption.
Choose which caption is more likely associated with the photograph. Base the choice on
visible people, objects, setting, signs, and other visual evidence. Do not use outside
knowledge. Return only the requested JSON object.

Caption A: {example['caption_a']}
Caption B: {example['caption_b']}"""


def parse_answer(text: str) -> tuple[str | None, str | None, float | None]:
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None, None, None
    choice = str(parsed.get("choice", "")).upper()
    if choice not in {"A", "B"}:
        choice = None
    confidence = parsed.get("confidence")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = None
    return choice, parsed.get("visible_evidence"), confidence


def run(models: list[str], limit: int, seed: int, out: Path) -> None:
    examples = build_examples(limit, seed)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(examples).to_csv(out / "examples.csv", index=False)
    rows = []
    for model in models:
        for index, example in enumerate(examples, 1):
            print(f"[{model}] {index}/{len(examples)} {example['id']}", flush=True)
            result = chat(
                prompt_for(example),
                model=model,
                image_paths=[ROOT / example["image_path"]],
                max_tokens=384,
                response_format=SCHEMA,
            )
            choice, visible_evidence, confidence = parse_answer(result["text"])
            rows.append({
                "model": model,
                "id": example["id"],
                "correct_option": example["correct"],
                "predicted_option": choice,
                "is_correct": choice == example["correct"],
                "json_valid": choice is not None,
                "confidence": confidence,
                "visible_evidence": visible_evidence,
                "response": result["text"],
                **{k: result[k] for k in (
                    "wall_seconds", "total_seconds", "load_seconds",
                    "prompt_tokens", "output_tokens", "done_reason",
                )},
            })
            pd.DataFrame(rows).to_csv(out / "predictions.csv", index=False)

    results = pd.DataFrame(rows)
    summary = (
        results.groupby("model", sort=False)
        .agg(
            examples=("id", "size"),
            accuracy=("is_correct", "mean"),
            json_valid_rate=("json_valid", "mean"),
            median_wall_seconds=("wall_seconds", "median"),
            mean_wall_seconds=("wall_seconds", "mean"),
            median_output_tokens=("output_tokens", "median"),
        )
        .reset_index()
    )
    summary.to_csv(out / "summary.csv", index=False)
    metadata = {
        "experiment_id": "E06",
        "task": "development-only forced-choice image/caption association",
        "seed": seed,
        "limit": limit,
        "models": models,
        "chance_accuracy": 0.5,
        "warning": "Screening proxy only: BBC captions are not literal image descriptions.",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("\n" + summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1508)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    run(args.models, args.limit, args.seed, args.out)


if __name__ == "__main__":
    main()

