"""
Day 5: the two metrics that can actually answer the research question.

    python -m src.evaluate --recall              # FREE, no API. recall@k + alpha sweep
    python -m src.evaluate --judge --limit 3     # cheap smoke test of the judge path
    python -m src.evaluate --judge               # faithfulness (budget guarded; needs key)
    python -m src.evaluate --report              # aggregate -> results/metrics.csv
    python -m src.evaluate --dry-run             # print one judge request, send nothing

Why only these two metrics (D12). Any number computed from the text stream alone
is maximised by B1 *by definition*, because B1 is the argmax of that stream — mean
s_text over the retrieved set is <= B1's on 150/150 test queries, a mathematical
identity that reads as "the method under test is worse". A fair B1-vs-M comparison
needs ground truth OUTSIDE both streams. There are exactly two here:

  recall@k       the withheld gold article. Free, no API, and neither arm can tune
                 for it — the headline is indexed in neither stream (D8).
  faithfulness   an external judge from a different model family (D4).

Everything else was rejected: ROUGE-L needs a reference written from the article
the split deliberately withholds, so the highest-scoring system is one that leaks
(D5); the demo's on-screen statistics are diagnostics and say so (D12).

--------------------------------------------------------------------- the judge

Two passes, both automated:

  1. decompose   summary -> atomic claims.  Sees the summary ONLY.
  2. verify      (evidence, claims) -> supported yes/no per claim.

faithfulness = supported / total, hallucination = 1 - faithfulness. "Supported"
means grounded in the retrieved evidence, not true in the world — the judge
compares two texts in front of it and needs no outside knowledge, which is why
this automates reliably.

Pass 1 is deliberately blind to the evidence. A decomposer that can see the
evidence shapes claim boundaries to fit it — splitting an unsupported sentence
into a supported half and an unsupported half, or merging two claims into one
that the evidence happens to cover. Claim granularity would then depend on the
arm, and faithfulness = supported/total is a ratio over that granularity.

Held constant across B0/B1/M: judge model, both prompts, decoding parameters.
The prompts contain no condition label, and _assert_blind() enforces it.

  Honest limitation, state it in the report: the judge is blind to the LABEL, not
  to the evidence. M's evidence block carries [IMAGE n: "..."] caption lines and
  B1's does not, so a judge could in principle infer the arm. Stripping the
  captions is worse — the judge must score claims against the evidence that was
  actually in the generator's prompt (D7 Rule 2), and a claim drawn from a caption
  would become unsupportable by construction.

--------------------------------------------- temperature=0 is NOT used here

The research extension has no Anthropic credential, so GPT-5.6 Luna replaces
Claude as judge. It runs with reasoning_effort="none" and strict structured
output. Determinism and bounded cost come from the schema, serial execution,
prompt-hash cache, and the persistent $8 software budget. This remains a different
model family from the GPT-4o-mini generator, but the same-provider limitation must
be reported and checked against the 50 human labels.

The generator is untouched and still runs gpt-4o-mini at temperature=0 (that call
is an OpenAI chat-completion, where temperature=0 is still accepted and still
load-bearing). Different family from the judge, which is the point of D4.
"""
import argparse
import concurrent.futures as futures
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "llm_cache"
SUMMARIES = ROOT / "results" / "summaries.csv"
CLAIMS_OUT = ROOT / "results" / "claims.csv"
METRICS_OUT = ROOT / "results" / "metrics.csv"
RECALL_OUT = ROOT / "results" / "recall.csv"

# D4: different model family from the generator, so faithfulness is not self-graded.
JUDGE_MODEL = "gpt-5.6-luna"
JUDGE_MAX_TOKENS = 2000
SEED = 42
WORKERS = 1            # serial by design: cost and failure containment

ABSTAIN = "INSUFFICIENT_EVIDENCE"

# The canonical three. `paired_ids()` is defined on THESE ONLY — adding the ablation here
# would redefine the 129-item paired set and silently change what every earlier number
# referred to.
CONFIGS = ("B0", "B1", "M")

# Day 5 caption ablation (generate.py:ABLATION). Same retrieval as M, [IMAGE n:] lines
# stripped, so B1->M_nocap isolates the retrieval channel and M_nocap->M the caption
# channel. Judged on the same 129 items, reported as an extra row + its own section.
ABLATION = "M_nocap"
ABLATION_SUMMARIES = ROOT / "results" / "summaries_ablation.csv"


# ============================================================== judge transport

_client = None


def _get_client():
    global _client
    if _client is None:
        from dotenv import load_dotenv
        from openai import OpenAI

        # load_dotenv() resolves against the CALLING FILE's directory, not cwd.
        load_dotenv(ROOT / ".env")
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                f"OPENAI_API_KEY not set — add the restricted project key to {ROOT}/.env"
            )
        _client = OpenAI(api_key=key)
    return _client


def _cache_key(system: str, prompt: str, schema: dict) -> str:
    blob = json.dumps([JUDGE_MODEL, JUDGE_MAX_TOKENS, system, prompt, schema], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def _judge_call(system: str, prompt: str, schema: dict, use_cache: bool = True) -> dict:
    """One structured judge call, cached by prompt hash. Returns parsed JSON.

    No temperature: GPT-5.6 Luna is used with reasoning disabled and a strict
    JSON schema. The generator remains GPT-4o-mini, so this is a different model
    family, though not a different provider; human validation remains required.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"judge_{_cache_key(system, prompt, schema)}.json"
    if use_cache and path.exists():
        return json.loads(path.read_text())["response"]

    from . import api_budget

    api_budget.assert_can_spend(estimated_max_usd=0.10)
    resp = _get_client().chat.completions.create(
        model=JUDGE_MODEL,
        max_completion_tokens=JUDGE_MAX_TOKENS,
        reasoning_effort="none",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "judge_response", "strict": True, "schema": schema},
        },
    )
    usage = api_budget.record(
        model=JUDGE_MODEL,
        input_tokens=resp.usage.prompt_tokens,
        output_tokens=resp.usage.completion_tokens,
        category="judge",
        cache_key=path.stem,
    )
    if resp.choices[0].finish_reason == "length":
        raise RuntimeError(f"judge hit max_tokens ({JUDGE_MAX_TOKENS}) — raise it and re-run")
    text = resp.choices[0].message.content or ""
    parsed = json.loads(text)

    path.write_text(json.dumps({
        "model": JUDGE_MODEL, "system": system, "prompt": prompt,
        "response": parsed,
        "usage": {"in": resp.usage.prompt_tokens, "out": resp.usage.completion_tokens},
        "estimated_cost_usd": usage["estimated_cost_usd"],
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
    return parsed


# ================================================================ judge prompts

# Byte-identical across B0/B1/M. Nothing below names a condition, an arm, a
# retrieval mode, or the word "multimodal" — _assert_blind() enforces that.

DECOMPOSE_SYSTEM = """\
You split a news summary into atomic factual claims. You are a parser, not a critic:
you never assess whether a claim is true, only what claims the text makes."""

DECOMPOSE_PROMPT = """\
Split the summary below into atomic factual claims.

Each claim must:
- assert exactly one fact
- be self-contained: resolve pronouns and vague references ("the incident", "he")
  using the rest of the summary, so the claim can be checked on its own
- use only what the summary states. Add no names, numbers, dates or context.

Skip pure discourse framing that asserts nothing checkable, such as "the coverage
addresses several issues" or "the reports provide context on the situation".

If the summary makes no checkable factual claim at all, return an empty list.

SUMMARY:
\"\"\"
{summary}
\"\"\"
"""

DECOMPOSE_SCHEMA = {
    "type": "object",
    "properties": {"claims": {"type": "array", "items": {"type": "string"}}},
    "required": ["claims"],
    "additionalProperties": False,
}

VERIFY_SYSTEM = """\
You check whether claims are grounded in a supplied body of evidence. You judge only
what the evidence in front of you states. You never use outside knowledge, and you
never reward a claim for being true in the world."""

VERIFY_PROMPT = """\
For each numbered claim, decide whether the EVIDENCE supports it.

- "supported" means someone who had read only the evidence would affirm the claim.
  A direct statement and an unambiguous paraphrase both count.
- Mark a claim NOT supported if the evidence does not mention it, contradicts it, or
  covers it only partly — for instance the claim adds a name, number, date, cause or
  outcome that the evidence does not give.
- A claim that is true in the world but absent from the evidence is NOT supported.
- Lines of the form [IMAGE n: "..."], where they appear, are part of the evidence.

Give a reason of at most 15 words for each verdict.
Return one verdict per claim, with `index` matching the claim's number.

EVIDENCE:
\"\"\"
{evidence}
\"\"\"

CLAIMS:
{claims}
"""

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "supported": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["index", "supported", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}

_LABEL_LEAK = re.compile(
    r"\b(B0|B1|multimodal|text-only|image-fused|fused retrieval|baseline arm|"
    r"method under test|which arm|the condition)\b", re.I)


def _assert_prompts_blind() -> None:
    """D4: the judge must never be TOLD which condition produced a summary.

    Checked against the prompt TEMPLATES — the text this file authors — and not
    against rendered prompts. That distinction is load-bearing and was found the
    expensive way: scanning the rendered prompt matched the word "condition" inside
    real BBC summaries ("the woman's condition", "in a critical condition") and threw
    on 11 of 387 summaries, which the runner then skipped. The run completed, wrote a
    plausible claims.csv, and reported n=125 instead of 129 with nothing saying so —
    the same silent-sample-shrink that cost Day 3 a whole 450-row run.

    Summaries and evidence are the MATERIAL under judgment, not instructions. A news
    article containing the word "condition" is not a label leak. What must be free of
    labels is the wrapper this module writes around it.
    """
    for name, text in (("DECOMPOSE_SYSTEM", DECOMPOSE_SYSTEM),
                       ("DECOMPOSE_PROMPT", DECOMPOSE_PROMPT),
                       ("VERIFY_SYSTEM", VERIFY_SYSTEM),
                       ("VERIFY_PROMPT", VERIFY_PROMPT)):
        hit = _LABEL_LEAK.search(text)
        assert not hit, (f"condition label {hit.group(0)!r} in {name} — the judge would "
                         f"know which arm it is grading, which is the one bias D4 exists "
                         f"to remove")


_assert_prompts_blind()


# ============================================================ evidence handling

_BLOCK_NUM = re.compile(r"^\[(\d+)\]\s*")
_IMG_NUM = re.compile(r"^\[IMAGE\s+\d+:")


def _split_blocks(evidence: str) -> list[str]:
    """Evidence block -> list of per-article chunks, leading [n] marker stripped."""
    if not isinstance(evidence, str) or not evidence.strip():
        return []
    return [b.strip() for b in evidence.split("\n\n") if b.strip()]


def merge_evidence(*blocks: str) -> str:
    """Union of several evidence blocks, deduped by headline and renumbered.

    B0 has no evidence of its own — it is the no-retrieval ceiling. To get a
    faithfulness number on the same scale as B1 and M, its claims are judged against
    the union of what B1 and M retrieved for the same query.

    The union rather than either arm alone, deliberately: judging B0 against B1's
    evidence would make the hallucination ceiling a function of B1's retrieval, and
    the ceiling would move if B1 changed. The union is symmetric between the arms.
    It is also generous to B0 — a larger grounding set can only raise B0's score —
    which is the safe direction for a ceiling claim: if B0 still hallucinates more
    than both arms on a superset of their evidence, the conclusion is stronger.
    """
    seen, out = set(), []
    for ev in blocks:
        for block in _split_blocks(ev):
            lines = block.split("\n")
            headline = _BLOCK_NUM.sub("", lines[0]).strip()
            if headline in seen:
                continue
            seen.add(headline)
            out.append([headline] + lines[1:])

    # Renumber both the article marker and its [IMAGE n:] line, which must agree.
    rendered = []
    for i, lines in enumerate(out, 1):
        body = [f"[{i}] {lines[0]}"]
        for ln in lines[1:]:
            body.append(_IMG_NUM.sub(f'[IMAGE {i}:', ln) if _IMG_NUM.match(ln) else ln)
        rendered.append("\n".join(body))
    return "\n\n".join(rendered)


# A refusal written as prose instead of as the token. Matches an opening sentence that
# disclaims the evidence ("The evidence does not provide information about X") rather
# than reporting it. Validated against the 5 summaries the decomposer independently
# reduced to zero claims: catches all 5, and every flagged example inspected by eye is a
# genuine refusal.
_SOFT_REFUSAL = re.compile(
    r"^.{0,40}?(evidence|coverage|articles?|information provided)\b[^.]{0,80}?"
    r"\b(does not|do not|doesn't|don't|lacks?|fails? to)\b[^.]{0,60}?"
    r"\b(provide|contain|include|mention|address|discuss|relate|specify|offer)", re.I)


def refusal_kind(summary) -> str:
    """-> "hard" | "soft" | "usable". There are THREE refusal modes, not two.

    D10 and the Day 3 handoff both warn that a refusal rate must read the summary text
    rather than the `abstained` flag — but the check they prescribe
    (`summary == INSUFFICIENT_EVIDENCE`) only catches the token. A third mode exists and
    was missed by every Day 3 and Day 4 check:

      hard    the literal INSUFFICIENT_EVIDENCE token          17 B1 / 19 M
      soft    prose refusal: "The evidence does not provide    36 B1 / 35 M
              information about X. It covers unrelated
              topics including A and B."
      usable  an actual summary of the retrieved coverage      97 B1 / 96 M

    So the true refusal rate is ~35%, not the ~12% Day 3 reported. Report the corrected
    figure and say the earlier one undercounted.

    Soft refusals are not merely miscounted — they CONTAMINATE faithfulness. The
    decomposer extracts their second sentence as claims ("The evidence covers a scandal
    involving the Post Office"), which are meta-statements about the evidence and are
    therefore supported almost by construction. Measured: soft refusals score 0.947 (B1)
    and 0.921 (M) against 0.900 / 0.889 for genuine summaries, inflating both arms.

    They inflate both arms nearly equally (28 B1 vs 32 M among judged items), so the
    headline B1-vs-M difference barely moves — but a metric whose value depends on how
    often the model declined is not a faithfulness metric, so report() cuts both ways.
    """
    if not isinstance(summary, str) or summary.strip().startswith(ABSTAIN):
        return "hard"
    return "soft" if _SOFT_REFUSAL.match(summary.strip()) else "usable"


def _is_refusal(summary) -> bool:
    """Hard refusals only — what `paired_ids()` uses to define the Day 3 paired set.

    Deliberately NOT widened to include soft refusals: the 129-item paired set is the
    published Day 3 interface, and redefining it here would silently change what every
    earlier number referred to. Soft refusals are handled as a reported cut in report(),
    not by quietly shrinking the sample.
    """
    return refusal_kind(summary) == "hard"


# ==================================================================== the run

def load_summaries() -> pd.DataFrame:
    df = pd.read_csv(SUMMARIES)
    if ABLATION_SUMMARIES.exists():
        df = pd.concat([df, pd.read_csv(ABLATION_SUMMARIES)], ignore_index=True)
    df["summary"] = df.summary.fillna("")
    df["kind"] = df.summary.map(refusal_kind)
    df["refused"] = df.kind == "hard"
    df["evidence"] = df.evidence.fillna("")
    return df


def paired_ids(df: pd.DataFrame) -> list[str]:
    """Test ids where every config produced a summary — the reporting set.

    Faithfulness is reported on this set and only this set. Dropping items post-hoc
    on any other criterion would be selection on the outcome.
    """
    ok = df[~df.refused].pivot_table(
        index="test_id", columns="config", values="summary", aggfunc="first")
    have = [c for c in CONFIGS if c in ok.columns]
    return sorted(ok.dropna(subset=have).index)


def _judge_one(row: dict, use_cache: bool = True) -> list[dict]:
    """One summary -> one row per atomic claim, with the verdict attached."""
    dp = DECOMPOSE_PROMPT.format(summary=row["summary"])
    claims = _judge_call(DECOMPOSE_SYSTEM, dp, DECOMPOSE_SCHEMA, use_cache)["claims"]
    claims = [c.strip() for c in claims if c and c.strip()]
    if not claims:
        return []

    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, 1))
    vp = VERIFY_PROMPT.format(evidence=row["judge_evidence"], claims=numbered)
    verdicts = _judge_call(VERIFY_SYSTEM, vp, VERIFY_SCHEMA, use_cache)["verdicts"]

    by_index = {int(v["index"]): v for v in verdicts}
    out = []
    for i, claim in enumerate(claims, 1):
        v = by_index.get(i)
        if v is None:                       # judge skipped one; count it, don't drop it
            out.append({"test_id": row["test_id"], "config": row["config"],
                        "claim_index": i, "claim": claim,
                        "supported": False, "reason": "NO_VERDICT_RETURNED"})
            continue
        out.append({"test_id": row["test_id"], "config": row["config"],
                    "claim_index": i, "claim": claim,
                    "supported": bool(v["supported"]), "reason": v["reason"]})
    return out


def _judge_rows(df: pd.DataFrame, ids: list[str]) -> list[dict]:
    """Attach the evidence each config's claims are judged against, then judge."""
    ev = {(r.test_id, r.config): r.evidence for r in df.itertuples()}
    present = [c for c in (*CONFIGS, ABLATION) if c in set(df.config)]
    rows = []
    for tid in ids:
        # Union is over the canonical arms only. M_nocap retrieves exactly what M does, so
        # it adds no articles — keeping it out means B0's prompts are unchanged and B0 is
        # served entirely from cache rather than re-billed at $1.96.
        union = merge_evidence(ev.get((tid, "B1"), ""), ev.get((tid, "M"), ""))
        for config in present:
            sub = df[(df.test_id == tid) & (df.config == config)]
            if sub.empty or sub.iloc[0].refused:
                continue
            rows.append({
                "test_id": tid,
                "config": config,
                "summary": sub.iloc[0].summary,
                # B0 has no evidence of its own — see merge_evidence().
                "judge_evidence": union if config == "B0" else sub.iloc[0].evidence,
            })
    return rows


def run_judge(limit: int | None = None, use_cache: bool = True) -> pd.DataFrame:
    df = load_summaries()
    ids = paired_ids(df)
    if limit:
        ids = ids[:limit]
    work = _judge_rows(df, ids)
    print(f"judging {len(work)} summaries over {len(ids)} paired test items "
          f"({JUDGE_MODEL}, thinking off, cached by prompt hash)")

    out, failed, done, t0 = [], [], 0, time.time()
    with futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(_judge_one, r, use_cache): r for r in work}
        for f in futures.as_completed(futs):
            r = futs[f]
            try:
                out.extend(f.result())
            except Exception as e:                      # noqa: BLE001
                failed.append((r["test_id"], r["config"], f"{type(e).__name__}: {e}"))
            done += 1
            print(f"  {done}/{len(work)} summaries, {len(out)} claims, "
                  f"{len(failed)} failed, {time.time() - t0:.0f}s", end="\r")
    print()

    claims = pd.DataFrame(out).sort_values(["test_id", "config", "claim_index"])
    CLAIMS_OUT.parent.mkdir(parents=True, exist_ok=True)
    claims.to_csv(CLAIMS_OUT, index=False)
    print(f"wrote {CLAIMS_OUT}  ({len(claims)} claims from "
          f"{len(work) - len(failed)}/{len(work)} summaries)")

    # A partial run must never be mistaken for a complete one. Every failure here is
    # one summary missing from a PAIRED comparison, so it silently shrinks n on one
    # arm only — the exact shape of the Day 3 bug where a broken 450-row run passed
    # every check. Stop hard; the prompt-hash cache makes the retry nearly free.
    if failed:
        print(f"\n{'=' * 72}\nFAIL  {len(failed)} of {len(work)} summaries did not judge\n"
              f"{'=' * 72}")
        for tid, cfg, err in failed[:10]:
            print(f"  {tid}/{cfg}: {err}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
        raise SystemExit(
            f"\nrefusing to report on a partial run. {CLAIMS_OUT.name} was written for "
            f"inspection but is INCOMPLETE — do not run --report against it. Fix the "
            f"errors above and re-run --judge: everything that already succeeded is "
            f"cached, so the retry re-bills only the failures.")
    return claims


# ==================================================================== reporting

def _bootstrap_ci(diff: np.ndarray, n: int = 10000, seed: int = SEED) -> tuple:
    rng = np.random.default_rng(seed)
    means = rng.choice(diff, size=(n, diff.size), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def report() -> pd.DataFrame:
    from scipy.stats import wilcoxon

    df = load_summaries()
    ids = paired_ids(df)
    claims = pd.read_csv(CLAIMS_OUT)

    # --- per-item faithfulness: supported / total, per (item, config)
    per_item = (claims.groupby(["test_id", "config"])
                .supported.agg(["sum", "count"]).reset_index()
                .rename(columns={"sum": "supported", "count": "claims"}))
    per_item["faithfulness"] = per_item.supported / per_item.claims

    per_item = per_item.merge(df[["test_id", "config", "kind"]], on=["test_id", "config"])

    present = [c for c in (*CONFIGS, ABLATION) if c in set(df.config)]
    rows = []
    for c in present:
        d = per_item[per_item.config == c]
        arm = df[df.config == c]
        rows.append({
            "config": c,
            "n_items": len(d),
            "n_claims": int(d.claims.sum()),
            # Macro: mean of per-item ratios, one vote per test item. This is the
            # headline — it matches the paired test below and does not let a verbose
            # summary outvote a terse one.
            "faithfulness_macro": round(d.faithfulness.mean(), 4),
            # Micro: pooled supported/total over all claims. Reported alongside
            # because the two diverge when claim counts differ between arms.
            "faithfulness_micro": round(d.supported.sum() / d.claims.sum(), 4),
            "hallucination_macro": round(1 - d.faithfulness.mean(), 4),
            "claims_per_summary": round(d.claims.mean(), 2),
            "tau_gated": int(arm.abstained.sum()),
            "hard_refusal": int((arm.kind == "hard").sum()),
            "soft_refusal": int((arm.kind == "soft").sum()),
            "true_refusal_rate": round((arm.kind != "usable").mean(), 4),
            # Faithfulness over genuine summaries only — soft refusals excluded.
            "faithfulness_usable": round(d[d.kind == "usable"].faithfulness.mean(), 4),
        })
    metrics = pd.DataFrame(rows)

    def paired(subset: pd.DataFrame, label: str, a: str = "B1", b: str = "M") -> dict:
        piv = subset.pivot(index="test_id", columns="config", values="faithfulness")
        piv = piv.loc[[i for i in ids if i in piv.index], [a, b]].dropna()
        lo_col, hi_col = piv[a], piv[b]
        d_bm = (hi_col - lo_col).to_numpy()
        lo, hi = _bootstrap_ci(d_bm)
        p = float(wilcoxon(hi_col, lo_col).pvalue) if d_bm[d_bm != 0].size else float("nan")
        print(f"\n{label}  (n={len(piv)})")
        print(f"  mean faithfulness  {a} {lo_col.mean():.4f}   {b} {hi_col.mean():.4f}")
        print(f"  difference ({b} - {a})  {d_bm.mean():+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
        print(f"  Wilcoxon signed-rank p = {p:.4f}   "
              f"({int((d_bm > 0).sum())} {b} better, {int((d_bm < 0).sum())} {a} better, "
              f"{int((d_bm == 0).sum())} tied)")
        return {"n": len(piv), "diff": d_bm.mean(), "lo": lo, "hi": hi, "p": p}

    print(f"\n{'=' * 78}\nFAITHFULNESS  (judge {JUDGE_MODEL}, blind to condition, D4)\n{'=' * 78}")
    print(metrics.to_string(index=False))
    n0 = len(ids)
    for c in CONFIGS:
        got = int((per_item.config == c).sum())
        if got < n0:
            print(f"\nnote: {c} judged {got}/{n0} items — {n0 - got} summaries decomposed to "
                  f"ZERO claims (soft refusals asserting nothing checkable). Excluded from "
                  f"faithfulness because 0/0 is undefined, not dropped silently.")

    # Both cuts, always. The all-items cut is contaminated upward by soft refusals, whose
    # "claims" are meta-statements about the evidence and so are supported by construction
    # (measured: 0.947 B1 / 0.921 M vs 0.900 / 0.889 for genuine summaries). The usable-only
    # cut is the honest faithfulness number. Report both; if they ever disagree in DIRECTION,
    # the result is an artifact of refusal rate, not of retrieval.
    a = paired(per_item, "ALL judged items (contaminated upward by soft refusals)")
    b = paired(per_item[per_item.kind == "usable"], "USABLE only — soft refusals excluded "
                                                    "(the honest number)")
    if (a["diff"] > 0) != (b["diff"] > 0):
        print("\n  WARNING: the two cuts disagree in SIGN. The apparent B1-vs-M difference "
              "tracks refusal rate, not faithfulness. Do not report it as a result.")
    elif b["lo"] <= 0 <= b["hi"]:
        print(f"\n  -> CI spans zero on both cuts: no significant B1-vs-M faithfulness "
              f"difference at n={b['n']}. Report the interval, not the point estimate.")

    # --- caption ablation: which of M's two channels, if either, did anything?
    extra_rows = []
    if ABLATION in present:
        print(f"\n{'=' * 78}\nABLATION — M differs from B1 in TWO ways; this separates them"
              f"\n{'=' * 78}")
        print("  B1        text-only retrieval,   no captions in prompt")
        print(f"  {ABLATION:<9} image-fused retrieval, no captions in prompt")
        print("  M         image-fused retrieval, captions in prompt")
        u = per_item[per_item.kind == "usable"]
        r1 = paired(u, "RETRIEVAL channel   B1 -> M_nocap  (different articles, same prompt "
                       "format)", "B1", ABLATION)
        r2 = paired(u, "CAPTION channel     M_nocap -> M   (same articles, captions added)",
                    ABLATION, "M")
        for label, r in (("retrieval_channel_B1_vs_Mnocap", r1),
                         ("caption_channel_Mnocap_vs_M", r2)):
            extra_rows.append((label, r))
        if r2["lo"] <= 0 <= r2["hi"]:
            print(f"\n  -> the caption channel is INERT: adding ~93 words of image caption to "
                  f"every\n     prompt moves faithfulness by {r2['diff']:+.4f} "
                  f"[{r2['lo']:+.4f}, {r2['hi']:+.4f}], p={r2['p']:.3f}. The visual evidence "
                  f"reaches the\n     model and changes nothing measurable.")

    for label, r in (("B1_vs_M_all", a), ("B1_vs_M_usable", b), *extra_rows):
        metrics = pd.concat([metrics, pd.DataFrame([{
            "config": label, "n_items": r["n"],
            "faithfulness_macro": round(r["diff"], 4),
            "faithfulness_micro": f"CI[{r['lo']:.4f},{r['hi']:.4f}]",
            "true_refusal_rate": f"wilcoxon_p={r['p']:.4f}",
        }])], ignore_index=True)
    metrics.to_csv(METRICS_OUT, index=False)
    print(f"\nwrote {METRICS_OUT}")
    return metrics


# ============================================== recall@k + alpha sweep (no API)

def _mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar on the discordant pairs only."""
    from scipy.stats import binomtest
    n = b + c
    return 1.0 if n == 0 else float(binomtest(b, n, 0.5).pvalue)


def recall(alphas=(0.0, 0.25, 0.5, 0.75, 0.9, 1.0), ks=(1, 5, 10)) -> pd.DataFrame:
    """Recall of the withheld gold article. Free — no API calls.

    Uses `query_qa` (the pinpoint question), not `query` (D8): a question naming the
    gold article's unique subject makes "did the retriever find it?" a sharp test.
    `include_test=True` because with the gold article unindexed recall@k is 0 by
    construction — this is the ONLY call site allowed to pass it (D7 Rule 3).
    """
    from .queries import load as load_queries
    from .retrieve import retrieve

    q = load_queries()
    kmax = max(ks)
    print(f"recall@k over {len(q)} test items, query_qa, include_test=True")

    got = {}   # (label) -> list of rank-or-None
    def sweep(label, **kw):
        ranks = []
        for r in q.itertuples():
            hits = retrieve(r.query_qa, k=kmax, include_test=True, **kw)
            ids = [h.article_id for h in hits]
            ranks.append(ids.index(r.test_id) + 1 if r.test_id in ids else None)
        got[label] = ranks
        print(f"  {label:<22} done")

    sweep("B1", mode="text")
    for a in alphas:
        sweep(f"M alpha={a}", mode="multimodal", alpha=a)

    rows = []
    base = got["B1"]
    for label, ranks in got.items():
        row = {"config": label}
        for k in ks:
            row[f"recall@{k}"] = round(np.mean([r is not None and r <= k for r in ranks]), 4)
        if label != "B1":
            hit_m = [r is not None and r <= 5 for r in ranks]
            hit_b = [r is not None and r <= 5 for r in base]
            b = sum(hb and not hm for hb, hm in zip(hit_b, hit_m))   # B1-only wins
            c = sum(hm and not hb for hb, hm in zip(hit_b, hit_m))   # M-only wins
            row.update({"B1_only@5": b, "M_only@5": c,
                        "mcnemar_p@5": round(_mcnemar_exact(b, c), 4)})
        rows.append(row)

    out = pd.DataFrame(rows)
    for col in ("B1_only@5", "M_only@5"):
        if col in out:
            out[col] = out[col].astype("Int64")
    RECALL_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(RECALL_OUT, index=False)
    print(f"\n{'=' * 72}\nRETRIEVAL RECALL — ground truth outside both score streams (D12)\n{'=' * 72}")
    print(out.to_string(index=False))
    print(f"\nwrote {RECALL_OUT}")

    # alpha=1.0 must land exactly on B1: mode="text" IS alpha=1.0 down the same code
    # path, not a second implementation. If these ever diverge, a B1-vs-M difference
    # could be an implementation difference rather than the images — the one failure
    # this project cannot detect by eye. Free to check, so check it every run.
    if 1.0 in alphas:
        same = got["B1"] == got["M alpha=1.0"]
        print(f"\n{'PASS' if same else 'FAIL'}  alpha=1.0 reproduces B1 exactly on all "
              f"{len(q)} queries — B1 and M share one code path, verified not assumed")
    print("\nThis is retrieval quality, NOT the research question. A lower-recall set can\n"
          "still produce more faithful summaries — fusion pulls in topically adjacent\n"
          "articles, and adjacency is exactly where hallucination shows up.")
    return out


# ========================================================================= CLI

def dry_run() -> None:
    """Print the exact judge request for one real item. Sends nothing, costs nothing."""
    df = load_summaries()
    ids = paired_ids(df)
    work = _judge_rows(df, ids[:1])
    r = work[-1]     # the M row: the one whose evidence carries [IMAGE n:] lines
    dp = DECOMPOSE_PROMPT.format(summary=r["summary"])
    print(f"model={JUDGE_MODEL}  max_tokens={JUDGE_MAX_TOKENS}  thinking=disabled  "
          f"temperature=<omitted: 400 on Claude 5>")
    print(f"\n--- PASS 1 system ---\n{DECOMPOSE_SYSTEM}\n\n--- PASS 1 user ---\n{dp}")
    print(f"--- PASS 1 schema ---\n{json.dumps(DECOMPOSE_SCHEMA)}")
    print(f"\n--- PASS 2 system ---\n{VERIFY_SYSTEM}\n\n--- PASS 2 user (evidence for "
          f"{r['test_id']}/{r['config']}) ---")
    print(VERIFY_PROMPT.format(evidence=r["judge_evidence"],
                               claims="1. <claims from pass 1>")[:2600])
    print(f"--- PASS 2 schema ---\n{json.dumps(VERIFY_SCHEMA)}")
    print(f"\nblindness check passed. {len(_judge_rows(df, ids))} summaries would be judged.")


def main():
    ap = argparse.ArgumentParser(description="Day 5 evaluation: faithfulness + recall@k")
    ap.add_argument("--recall", action="store_true", help="recall@k + alpha sweep (free)")
    ap.add_argument("--judge", action="store_true", help="run the faithfulness judge (API)")
    ap.add_argument("--report", action="store_true", help="aggregate -> results/metrics.csv")
    ap.add_argument("--dry-run", action="store_true", help="print one judge request, send nothing")
    ap.add_argument("--limit", type=int, help="first N test items only (smoke test)")
    ap.add_argument("--no-cache", action="store_true", help="bypass the prompt-hash cache")
    a = ap.parse_args()

    if a.dry_run:
        dry_run()
    if a.recall:
        recall()
    if a.judge:
        run_judge(limit=a.limit, use_cache=not a.no_cache)
    if a.report:
        report()
    if not any([a.recall, a.judge, a.report, a.dry_run]):
        ap.print_help()


if __name__ == "__main__":
    main()
