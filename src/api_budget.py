"""Persistent, conservative budget guard for hosted model calls.

The OpenAI project has a $10 prepaid balance. This module stops project code at
$8 estimated usage, leaving room for delayed accounting and manual diagnostics.
Cached calls never reach this module and therefore never add ledger entries.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "results" / "api_usage.jsonl"
DEFAULT_BUDGET_USD = 8.0

# USD per one million tokens. Input is deliberately charged at the uncached
# rate even when the provider reports cached tokens, making the ledger an upper
# estimate. Update only with an experiment-ledger note and an official source.
PRICES = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-5.6-luna": {"input": 0.20, "output": 1.20},
}


class BudgetExceeded(RuntimeError):
    pass


def budget_usd() -> float:
    return float(os.environ.get("API_BUDGET_USD", DEFAULT_BUDGET_USD))


def _price(model: str) -> dict[str, float]:
    for name, price in PRICES.items():
        if model == name or model.startswith(name + "-"):
            return price
    raise ValueError(f"No audited API price configured for model {model!r}")


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    price = _price(model)
    return (
        max(0, int(input_tokens)) * price["input"]
        + max(0, int(output_tokens)) * price["output"]
    ) / 1_000_000


def entries(path: Path = LEDGER) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def spent_usd(path: Path = LEDGER) -> float:
    return sum(float(row.get("estimated_cost_usd", 0)) for row in entries(path))


def assert_can_spend(
    *, estimated_max_usd: float = 0.10, path: Path = LEDGER,
    limit_usd: float | None = None,
) -> None:
    limit = budget_usd() if limit_usd is None else limit_usd
    spent = spent_usd(path)
    if spent + estimated_max_usd > limit:
        raise BudgetExceeded(
            f"API budget stop: ${spent:.4f} recorded + ${estimated_max_usd:.2f} "
            f"reserve would exceed ${limit:.2f}."
        )


def record(
    *, model: str, input_tokens: int, output_tokens: int, category: str,
    cache_key: str, path: Path = LEDGER,
) -> dict:
    cost = estimate_cost(model, input_tokens, output_tokens)
    row = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "category": category,
        "model": model,
        "cache_key": cache_key,
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "estimated_cost_usd": round(cost, 8),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def summary(path: Path = LEDGER) -> dict:
    rows = entries(path)
    by_model: dict[str, float] = {}
    for row in rows:
        by_model[row["model"]] = by_model.get(row["model"], 0) + float(
            row["estimated_cost_usd"]
        )
    return {
        "calls": len(rows),
        "estimated_spend_usd": round(spent_usd(path), 6),
        "budget_usd": budget_usd(),
        "remaining_software_budget_usd": round(budget_usd() - spent_usd(path), 6),
        "by_model_usd": {k: round(v, 6) for k, v in sorted(by_model.items())},
    }

