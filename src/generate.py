"""
Day 3: retrieved evidence -> prompt -> gpt-4o-mini -> grounded summary.

    python -m src.generate                      # demo: 1 query through B0/B1/M
    python -m src.generate --tau-scan           # pick tau (no API calls)
    python -m src.generate --run --limit 5      # 5 test items x 3 configs
    python -m src.generate --run                # full 450 rows -> results/summaries.csv

    from src.generate import summarize
    rec = summarize("who won the election in taiwan?", config="M")

Contract — docs/decisions.md D7, all three rules are load-bearing
-----------------------------------------------------------------
1. B1 and M share BYTE-IDENTICAL instruction text. Only the evidence block
   differs: B1 gets numbered passages, M gets the same passages plus
   [IMAGE k: "caption"] lines. A nicer prompt for the multimodal arm would
   confound the experiment — any faithfulness gain could be the prompt rather
   than the images. `_check_prompt_parity()` asserts this on every demo run
   rather than trusting the two templates to stay in sync by eye.

   B0 is the deliberate exception: with no evidence you cannot say "use ONLY the
   evidence". B0 is the hallucination ceiling, not a third arm.

2. Persist the EVIDENCE, not just the summary. The Day 5 judge scores each claim
   against the text that was actually in the prompt. Store only the summary and
   evaluation has nothing to judge against, and retrieval has to be re-run and
   hoped to match. Schema: test_id | config | query | evidence | summary (+ the
   retrieval diagnostics Day 5 wants for the stratified cut).

3. Assert no retrieved id is in the test set. retrieve() already defaults
   include_test=False, but evaluate.py's recall@k call passes True, and a leak
   into generation would push faithfulness to ceiling and LOOK LIKE A GREAT
   RESULT. Cheap guard, catastrophic failure mode.

Abstention gates on RAW s_text, never on `score`. `score` is min-max normalized
per query, so the top hit is ~1.0 on every query no matter how poor the match —
threshold on it and you never abstain. Same statistic in both arms so the gate
cannot itself become a B1-vs-M difference; see TAU below.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd

from .retrieve import retrieve

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "llm_cache"
TEST = "data/processed/test.parquet"
OUT = "results/summaries.csv"

# Generator is held CONSTANT across B0/B1/M — swapping it per arm would void the
# comparison. Not gpt-5.6-luna: it rejects temperature=0, and temperature=0 is
# load-bearing here (without it a B1-vs-M gap could be sampling noise).
MODEL = "gpt-4o-mini"
TEMPERATURE = 0
SEED = 42
MAX_TOKENS = 300

# Abstention threshold on raw cosine s_text, chosen from measurement, not taste.
# Over the 150 real test queries (--tau-scan) the gate statistic runs min 0.340,
# 1st pct 0.373, median 0.559. Over 8 deliberately off-topic probes (sourdough,
# sigmoid derivatives, flights to Osaka) it runs 0.19-0.365. The two populations
# overlap in a narrow 0.34-0.37 band, so no threshold separates them cleanly.
#
# 0.35 catches 6 of the 8 off-topic probes while falsely abstaining on 0.0% of B1
# and 0.7% of M test queries. Pushing to 0.40 catches all 8 but abstains on
# 4.7%/7.3% of real queries — and every abstention drops an item from the paired
# B1-vs-M faithfulness comparison, which is the entire result. Keeping false
# abstention near zero is worth more than catching two borderline probes.
# Re-derive with --tau-scan after any change to chunking or the encoders.
TAU = 0.35

CONFIGS = ("B0", "B1", "M")

# M_nocap is the Day 5 ablation, NOT part of the canonical three. M differs from B1 in two
# ways at once — image fusion changes WHICH articles are retrieved, and captions add text
# to the prompt — so a null B1-vs-M result cannot say which channel was inert. M_nocap
# retrieves EXACTLY as M does and drops only the [IMAGE n: "..."] lines, so:
#     B1 -> M_nocap   isolates the retrieval channel
#     M_nocap -> M    isolates the caption channel
# Because retrieval is identical to M's, B0's evidence union (D14) is unchanged and B0
# does not need re-judging.
ABLATION = "M_nocap"
_MODE = {"B1": "text", "M": "multimodal", ABLATION: "multimodal"}
_WITH_IMAGES = {"B1": False, "M": True, ABLATION: False}


# ---------------------------------------------------------------- LLM call

_client = None


def _get_client():
    global _client
    if _client is None:
        from dotenv import load_dotenv
        from openai import OpenAI

        # load_dotenv() resolves against the CALLING FILE's directory, not cwd —
        # from a script elsewhere it silently finds nothing and leaves the key None.
        load_dotenv(ROOT / ".env")
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(f"OPENAI_API_KEY not set — copy .env.example to {ROOT}/.env")
        _client = OpenAI(api_key=key)
    return _client


def _cache_key(prompt: str, model: str, temperature: float, max_tokens: int) -> str:
    blob = json.dumps([model, temperature, max_tokens, prompt], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def generate(prompt: str, model: str = MODEL, temperature: float = TEMPERATURE,
             max_tokens: int = MAX_TOKENS, use_cache: bool = True) -> str:
    """One chat-completion call, cached by prompt hash.

    The cache key covers the decoding parameters too, so changing the prompt or
    the model invalidates it rather than silently returning a stale summary.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{_cache_key(prompt, model, temperature, max_tokens)}.json"
    if use_cache and path.exists():
        return json.loads(path.read_text())["response"]

    # chat.completions, not the Responses API: separate key permission, and it is
    # the format Ollama/vLLM emulate for the future local-Llama swap.
    resp = _get_client().chat.completions.create(
        model=model,
        temperature=temperature,
        seed=SEED,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    text = (resp.choices[0].message.content or "").strip()
    path.write_text(json.dumps({
        "model": model, "temperature": temperature, "max_tokens": max_tokens,
        "prompt": prompt, "response": text,
        "usage": {"in": resp.usage.prompt_tokens, "out": resp.usage.completion_tokens},
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
    return text


# ---------------------------------------------------------------- prompts

ABSTAIN = "INSUFFICIENT_EVIDENCE"

# D7 Rule 1: this string is shared verbatim by B1 and M. Do not fork it.
#
# The task is SUMMARIZATION, not question answering — plan_7day.md:82. This
# distinction is not cosmetic, and getting it wrong cost a full 450-row run:
# phrased as "answer the question", gpt-4o-mini returned INSUFFICIENT_EVIDENCE on
# 46% of B1 and 44% of M items. That was correct behaviour, not a bug. D8's
# queries are pinpoint questions derived from a test article, and D1 deliberately
# withholds that article from the generation index — so the specific fact asked
# for is, by construction, never in the evidence. Retrieval was fine (the refused
# items had s_text up to 0.842 and returned visibly on-topic articles); the task
# was unanswerable as posed, and the paired comparison collapsed to 74/150 items.
#
# Framed as "summarize what this coverage reports", the task is well-posed for
# every query, and it is a STRICTER hallucination probe: the evidence is adjacent
# to the query but does not contain its answer, so the temptation to fill the gap
# from prior knowledge is exactly what B1-vs-M is measuring.
INSTRUCTIONS = """\
You are a news summarizer. Below is a search query and the numbered news coverage
retrieved for it. Summarize what that coverage reports on the topic of the query.

Rules:
- Use ONLY the numbered evidence. Every claim you write must be supported by it. Do not
  add names, numbers, dates, or events that are not in the evidence.
- Do not use prior knowledge, even if you believe the evidence is incomplete or wrong.
- The evidence was retrieved by a search engine, so it will rarely match the query
  exactly. If it covers the same subject, event, or theme, summarize it — partial or
  indirect relevance is enough. Do not refuse merely because the specific event named in
  the query is absent from the evidence.
- Report what the evidence does say, not what it fails to say. Do not comment on gaps,
  do not speculate, and do not fill them in.
- Reply with exactly INSUFFICIENT_EVIDENCE only if the evidence is about an entirely
  different subject.
- Write 3-4 sentences of plain prose. No bullet points, no preamble, no citation markers.
"""

# B0 is the no-retrieval ceiling. It CANNOT share the text above — "use only the
# evidence" is undefined with no evidence block. The length and format constraints
# are kept identical so summary length stays comparable across arms.
B0_INSTRUCTIONS = """\
You are a news summarizer. Summarize the news coverage on the topic of the query below,
from your own knowledge.

Rules:
- Write 3-4 sentences of plain prose. No bullet points, no preamble, no citation markers.
"""


def format_evidence(hits, with_images: bool) -> str:
    """Numbered evidence block. with_images appends the caption line (M only).

    Headlines appear in BOTH arms: the headline is indexed in neither stream, so
    including it adds the same context to each and cannot tilt the comparison.
    """
    blocks = []
    for i, h in enumerate(hits, 1):
        lines = [f"[{i}] {h.headline}"]
        lines += [p.strip() for p in h.passages]
        if with_images and h.caption:
            lines.append(f'[IMAGE {i}: "{h.caption.strip()}"]')
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_prompt(query: str, hits, config: str) -> tuple[str, str]:
    """-> (prompt, evidence). evidence is "" for B0 and is what Day 5 judges against."""
    if config == "B0":
        return f"{B0_INSTRUCTIONS}\nQUERY: {query}\n", ""
    # D7 Rule 1: INSTRUCTIONS is byte-identical for B1, M and M_nocap. Only the evidence
    # block may differ between them.
    evidence = format_evidence(hits, with_images=_WITH_IMAGES[config])
    return f"{INSTRUCTIONS}\nQUERY: {query}\n\nEVIDENCE:\n{evidence}\n", evidence


# ---------------------------------------------------------------- one item

_test_ids = None


def _get_test_ids() -> set:
    global _test_ids
    if _test_ids is None:
        _test_ids = set(pd.read_parquet(TEST)["id"])
    return _test_ids


def summarize(query: str, config: str = "M", k: int = 5, alpha: float = 0.5,
              tau: float = TAU, test_id: str = "", use_cache: bool = True) -> dict:
    """Run one (query, config) end to end. Returns the row that D7 Rule 2 persists."""
    if config not in (*CONFIGS, ABLATION):
        raise ValueError(f"config must be one of {(*CONFIGS, ABLATION)}, got {config!r}")

    hits = [] if config == "B0" else retrieve(
        query, mode=_MODE[config], k=k, alpha=alpha, include_test=False)

    # D7 Rule 3 — a test article in the generation evidence would push faithfulness
    # to ceiling and read as a great result. Assert rather than trust the default.
    leaked = {h.article_id for h in hits} & _get_test_ids()
    assert not leaked, f"test articles leaked into generation evidence: {sorted(leaked)}"

    # Gate on the strongest RAW text score in the returned set, not on hits[0].
    # Using the top-ranked hit would make the gate rank-order dependent, and M's
    # ranking is partly image-driven — the abstention rate would then differ
    # between arms for a reason that is not evidence quality.
    top_s_text = max((h.s_text for h in hits), default=0.0)
    abstained = config != "B0" and top_s_text < tau

    prompt, evidence = build_prompt(query, hits, config)
    summary = ABSTAIN if abstained else generate(prompt, use_cache=use_cache)

    return {
        "test_id": test_id,
        "config": config,
        "query": query,
        "evidence": evidence,
        "summary": summary,
        "abstained": bool(abstained),
        "retrieved_ids": "|".join(h.article_id for h in hits),
        "top_s_text": round(top_s_text, 4),
        "top_s_img": round(max((h.s_img for h in hits), default=0.0), 4),
        "n_hits": len(hits),
    }


def compare(query: str, configs=CONFIGS, **kw) -> dict:
    """One query -> {config: record}. What the Day 4 Streamlit toggle calls.

    Kept as a thin loop over summarize() rather than a fused implementation: B1 and
    M must go down identical code, and the cheapest way to guarantee that is to have
    only one path.
    """
    return {c: summarize(query, config=c, **kw) for c in configs}


# ---------------------------------------------------------------- batch run

def run_test_set(limit: int | None = None, k: int = 5, alpha: float = 0.5,
                 tau: float = TAU, configs=CONFIGS, out: str = OUT) -> pd.DataFrame:
    """Every test item x every config -> results/summaries.csv (D7 Rule 2).

    This file is the ENTIRE interface to Day 5: once it exists, evaluate.py runs
    standalone and can re-judge without re-generating.
    """
    from .queries import load as load_queries

    q = load_queries()
    if limit:
        q = q.head(limit)

    rows, t0 = [], time.time()
    for n, r in enumerate(q.itertuples(), 1):
        for config in configs:
            rows.append(summarize(r.query, config=config, k=k, alpha=alpha,
                                  tau=tau, test_id=r.test_id))
        print(f"  {n}/{len(q)} items ({len(rows)} rows, {time.time() - t0:.0f}s)", end="\r")
    print()

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}  ({len(df)} rows, {time.time() - t0:.0f}s)")

    print("\n--- checks ---")
    expect = len(q) * len(configs)
    print(f"{'PASS' if len(df) == expect else 'FAIL'}  {len(df)} rows (expected {expect})")
    empty = ((df.summary.str.strip() == "") | df.summary.isna()).sum()
    print(f"{'PASS' if empty == 0 else 'FAIL'}  no empty summaries ({empty})")
    ev = df[df.config != "B0"]
    print(f"{'PASS' if (ev.evidence.str.len() > 0).all() else 'FAIL'}  "
          f"evidence persisted for every B1/M row")
    ids = set(df.retrieved_ids.str.split("|").explode().dropna()) - {""}
    print(f"{'PASS' if not (ids & _get_test_ids()) else 'FAIL'}  no test id retrieved "
          f"({len(ids)} distinct pool articles used)")
    # Model-driven refusal is NOT the same as the tau gate, and the difference is
    # invisible in `abstained`. The first Day-3 run passed every check above while
    # 46% of B1 rows were the model replying INSUFFICIENT_EVIDENCE on its own — the
    # paired comparison had silently collapsed to 74/150 and nothing said so.
    # Anything below ~110 paired items means the task is mis-posed, not that the
    # corpus is hard; see D10.
    df["_refused"] = df.summary.str.strip() == ABSTAIN
    for c in configs:
        d = df[df.config == c]
        gate, model = d.abstained.sum(), (d._refused & ~d.abstained).sum()
        print(f"      {c}: {gate} gated by tau + {model} refused by the model = "
              f"{len(d) - gate - model}/{len(d)} usable, median "
              f"{d[~d._refused].summary.str.split().str.len().median():.0f} words")
    if {"B1", "M"} <= set(configs):
        p = df[df.config != "B0"].pivot(index="test_id", columns="config", values="_refused")
        paired = int((~p.B1 & ~p.M).sum())
        # Proportional, so --limit smoke runs are judged on the same standard as the
        # full 450. The full run sits at 129/150 = 86%; the broken first run was 49%.
        print(f"{'PASS' if paired >= 0.70 * len(p) else 'FAIL'}  {paired}/{len(p)} "
              f"({paired / len(p):.0%}) items where both arms produced a summary "
              f"— the paired B1-vs-M sample")
    return df.drop(columns="_refused")


# ---------------------------------------------------------------- tau scan

def tau_scan(limit: int | None = None, k: int = 5, alpha: float = 0.5):
    """Distribution of the gate statistic over the real test queries. No API calls."""
    from .queries import load as load_queries

    q = load_queries()
    if limit:
        q = q.head(limit)

    stats = {"B1": [], "M": []}
    for n, r in enumerate(q.itertuples(), 1):
        for c in ("B1", "M"):
            hits = retrieve(r.query, mode=_MODE[c], k=k, alpha=alpha, include_test=False)
            stats[c].append(max((h.s_text for h in hits), default=0.0))
        print(f"  {n}/{len(q)}", end="\r")
    print()

    s = pd.DataFrame(stats)
    print("\nmax raw s_text over the top-k, across test queries:")
    print(s.describe(percentiles=[.01, .05, .10, .25, .50]).round(3).to_string())
    print("\nabstention rate by tau:")
    print(f"  {'tau':>6}  {'B1':>7}  {'M':>7}")
    for t in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
        print(f"  {t:>6.2f}  {(s.B1 < t).mean():>6.1%}  {(s.M < t).mean():>6.1%}")
    print(f"\ncurrent TAU = {TAU}")


# ---------------------------------------------------------------- demo

def _check_prompt_parity(query: str, hits) -> bool:
    """D7 Rule 1 as an assert: B1 and M may differ only inside the evidence block."""
    p1, e1 = build_prompt(query, hits, "B1")
    pm, em = build_prompt(query, hits, "M")
    return p1.replace(e1, "<EVIDENCE>") == pm.replace(em, "<EVIDENCE>")


DEMO_QUERY = "what happened in the post office horizon scandal inquiry?"


def demo(query: str = DEMO_QUERY, k: int = 5, alpha: float = 0.5, tau: float = TAU):
    hits = retrieve(query, mode="multimodal", k=k, alpha=alpha)
    ok = _check_prompt_parity(query, hits)
    print(f"{'PASS' if ok else 'FAIL'}  B1/M instruction text byte-identical (D7 Rule 1)")
    assert ok, "B1 and M prompts differ outside the evidence block — comparison is void"

    for config in CONFIGS:
        t = time.time()
        rec = summarize(query, config=config, k=k, alpha=alpha, tau=tau)
        print("\n" + "=" * 88)
        print(f"{config}   ({time.time() - t:.1f}s"
              f"{', ABSTAINED' if rec['abstained'] else ''})")
        print("-" * 88)
        if rec["evidence"]:
            head = rec["evidence"].split("\n\n")[0]
            print(f"evidence: {len(rec['evidence'].split())} words, {rec['n_hits']} articles, "
                  f"top raw s_text {rec['top_s_text']:.3f}")
            print(f"  first block: {head[:300]}{'...' if len(head) > 300 else ''}")
        else:
            print("evidence: none (B0 is the no-retrieval ceiling)")
        print(f"\nsummary: {rec['summary']}")
    print("=" * 88)


def main():
    ap = argparse.ArgumentParser(description="Day 3 generation")
    ap.add_argument("--run", action="store_true", help="batch the test set -> " + OUT)
    ap.add_argument("--tau-scan", action="store_true", help="pick tau; makes no API calls")
    ap.add_argument("--limit", type=int, default=None, help="first N test items only")
    ap.add_argument("--query", default=DEMO_QUERY)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS),
                    help=f"which arms to run (default {' '.join(CONFIGS)}); "
                         f"'{ABLATION}' is the caption ablation")
    ap.add_argument("--out", default=OUT, help="output csv (use a separate file for "
                                               "the ablation — never overwrite " + OUT)
    args = ap.parse_args()

    if args.tau_scan:
        tau_scan(limit=args.limit, k=args.k, alpha=args.alpha)
    elif args.run:
        run_test_set(limit=args.limit, k=args.k, alpha=args.alpha, tau=args.tau,
                     configs=tuple(args.configs), out=args.out)
    else:
        demo(args.query, k=args.k, alpha=args.alpha, tau=args.tau)


if __name__ == "__main__":
    main()
