# Design decisions

Running log of decisions that shape results and are not obvious from the code.
Day-2-scoped decisions (index all 1023, query construction, pure image vector)
live in `day2_guide.md` § *Decisions to settle before writing code*.

---

## D4 — Claude judges faithfulness; GPT-4o-mini stays the generator

**Status:** decided 2026-08-01. Affects Day 5 (`src/evaluate.py`). Not blocking Days 2–4.

### The decision
`gpt-4o-mini` remains the generator for **all three configs** (B0/B1/M), unchanged.
The faithfulness judge in `evaluate.py` runs on **`claude-sonnet-5`** instead of
`gpt-4o-mini`.

### Why
`plan_7day.md:97` has GPT-4o-mini grading its own summaries. Models exhibit
**self-preference bias** — they rate their own output as supported more often than a
third party does. That bias sits directly on the headline B1-vs-M number, which is the
entire result. Different model families for generator and judge removes the confound.

Report line: *"Generator and judge are different model families, so faithfulness is not
self-graded."*

### Rules — treat these as load-bearing
| Rule | Reason |
|---|---|
| Same judge, same prompt, `temperature=0` across B0/B1/M | Same reason the generator is held constant |
| Judge sees only `(evidence, claim)` — never the condition label | A judge told "this is the multimodal one" is not blind |
| Cache by prompt hash, like the generator (`data/llm_cache/`) | Avoid re-billing on re-runs |
| Generator model is **not** changed | Swapping it would invalidate the Day-3 verification |

### How faithfulness is computed
Two passes, both automated — **no human labeling required in the loop**:

1. **Decompose** — summary → list of atomic claims.
2. **Verify** — for each claim, `(retrieved evidence, claim) → supported? yes/no`.

`faithfulness = supported / total`, `hallucination = 1 − faithfulness`.

"Supported" means **grounded in the retrieved evidence**, not true in the world. The
judge compares two texts in front of it; it needs no outside knowledge. This is a
verification task, not a generation task, which is why it automates reliably.

### Validation (do not skip — ~30 min)
Hand-label **50 random claims**, compare against the judge, report the agreement rate.
One line in the report — *"judge validated against 50 hand-labeled claims, 92%
agreement"* — converts the weakest methodological point into a strength. The proposal
already promises this (*"automatic factuality scores are validated against a small
human-annotated subset"*).

### Cost
Judge spend scales with **test-set size × configs × claims per summary** — it is
**independent of corpus size**. Growing the index from 1023 to the proposal's 50k
articles adds **$0** of judge cost.

Assumptions: 150 test items × 3 configs = 450 summaries; ~4 claims each; claims batched
one call per summary; k=5 passages ≈ 1400 input tok + 200 output tok per verify call,
plus a small decompose call. Total ≈ **0.8 MTok in / 0.16 MTok out**.

| Judge model | Rate (in/out per MTok) | Cost | With Batch API (−50%) |
|---|---|---|---|
| `claude-haiku-4-5` | $1 / $5 | ~$1.60 | ~$0.80 |
| **`claude-sonnet-5`** | **$2 / $10** (intro, through 2026-08-31) | **~$3.15** | **~$1.60** |
| `claude-sonnet-5` | $3 / $15 (standard) | ~$4.75 | ~$2.40 |
| `claude-opus-5` | $5 / $25 | ~$7.90 | ~$3.95 |

Sonnet 5's introductory rate runs through 2026-08-31 — the whole project week falls
inside it. **Batch API** (`client.messages.batches.create`) is 50% off and a natural fit:
Day 5 evaluation is offline, all 450 summaries exist before judging starts, and nothing
is latency-sensitive. Most batches finish within an hour.

At the proposal's scale the corpus grows but the test set need not. If it does grow to
~1500 items, multiply by 10: ~$16 unbatched, ~$8 batched. Still trivial.

Practical friction: needs `ANTHROPIC_API_KEY` in `.env` alongside `OPENAI_API_KEY`, plus
its own spend cap. A Claude Code subscription does **not** give scripts API access.

### Fallback
If the second API key is more setup than the week allows: keep `gpt-4o-mini` as judge and
**state the self-grading limitation in the report**. Honest, costs nothing, and the
comparison is still paired.

---

## D5 — No reference summaries (rejected)

**Status:** decided 2026-08-01. Rejected.

Considered generating reference summaries for the 150 test articles with an LLM (or by
hand) to enable reference-based scoring. **Rejected — not on cost or quality grounds.**

A model-written reference is competitive with a human-written one for single-article news
summarization, so provenance was never the issue. The problem is structural:

1. **The reference contains facts the system is correctly denied.** A reference is written
   from the test article's full body. The pipeline retrieves *related* pool articles, never
   the source (D1 / `day2_guide.md:57`). Low scores would measure how much the split
   withholds, not summary quality — and B1 and M are withheld the *same* facts, so it adds
   noise to the gap without adding signal.
2. **It rewards the exact failure the split prevents.** If reference = summary of the source
   article, the highest-scoring possible system is one that *retrieves the source and copies
   it*. Reference-based scoring would rank a leaking system above a correct one.

Neither is fixed by writing a better reference. Faithfulness-to-evidence (D4) needs no
reference at all, which is why it is feasible in a week.

The only leakage-safe reference design would derive references from the **retrievable**
evidence (summarize the pool articles per topic, then measure coverage). That is a
different metric answering a different question — completeness, not hallucination — and
requires defining topic clusters across 873 pool articles. Out of scope; list as future
work.

**Open item this raises:** `proposal.md` lists **ROUGE-L** among the metrics. ROUGE-L needs
a reference, so it inherits both problems above. Either drop it and report
faithfulness + recall@k only, or state explicitly in the report which reference it scores
against and why that is defensible. Decide before writing `evaluate.py`.

---

## D6 — The pool/test split stays as-is (random, unstratified)

**Status:** decided 2026-08-01. No action. Affects Day 5 reporting only.

`day1_split.py:32-35` is a seeded full shuffle (`random_state=42`) then a head/tail cut at
`min(150, len(df)//5)`. `story_type` is computed at line 29 but **never enters the split
logic** — it is attached as a column and printed, nothing more.

Measured consequence:

| `story_type` | test | pool |
|---|---:|---:|
| other | 61.3% | 69.8% |
| event_politics | **22.0%** | **15.9%** |
| sci_tech | 8.7% | 6.5% |
| markets_business | 8.0% | 7.7% |
| sport | **0.0%** (n=0) | 0.1% (n=1) |

`event_politics` is over-represented in test by ~6 points, and the single `sport` row landed
in the pool, so **test contains zero sport items**. Smallest usable test bucket is
`markets_business` at n=12.

**Decision: do not re-split.** The headline result is unaffected — B1 and M are evaluated on
the *same* 150 items, so the comparison is paired and test composition cancels. Re-splitting
would change `data/processed/*.parquet`, which is committed precisely so B1-vs-M numbers stay
comparable across teammates (see CLAUDE.md § Data), and would break anyone who has already
cloned. Stratification would only sharpen subgroup reporting, and it cannot rescue `sport`
(n=1 corpus-wide) regardless.

Report headline B1-vs-M only. For the stratified cut the hypothesis needs, use
`day2_guide.md:115` — hand-label the 150 test items into `event_centric` /
`abstract_topical` and delete `bucket()`, rather than re-splitting.

---

## D7 — Generation contract: identical prompts for B1 and M; persist the evidence

**Status:** decided 2026-08-01. Applies to Day 3 (`src/generate.py`). Record now, build later.

### Rule 1 — B1 and M share byte-identical instruction text
Only the `{evidence}` block may differ between B1 and M: B1 gets numbered passages, M gets the
same passages plus `[IMAGE k: "caption"]` lines (`plan_7day.md:68`). **A separate or nicer
prompt for the multimodal arm confounds the experiment** — any faithfulness gain could be the
prompt rather than the images, and the B1-vs-M comparison is void. Same principle as holding
the generator constant, one level up.

**B0 is the deliberate exception.** With no evidence you cannot instruct "summarize ONLY the
evidence", so B0's prompt is necessarily different (*"Answer from your own knowledge:
{query}"*). This is fine — B0 is the hallucination ceiling, not a third arm. Strict constancy
is required **between B1 and M only**.

### Rule 2 — `generate.py` must persist the evidence, not just the summary
The Day 5 judge scores each claim against **the evidence that was actually in the prompt**. If
`summaries.csv` stores only the summary, evaluation has nothing to judge against and retrieval
has to be re-run and hoped to match. Schema:

```
test_id | config | query | evidence | summary
```

One row per (test item × config) = 450 rows. This file is the entire interface between
generation and evaluation — once it exists, `evaluate.py` runs standalone and can re-judge
without re-generating.

### Rule 3 — assert no leakage into the generation path
`retrieve()` takes `include_test` defaulting to `False` (D1 / `day2_guide.md:57`). The one real
bug risk is `evaluate.py`'s permissive recall@k call leaking into generation, which would push
faithfulness to ceiling and *look like a great result*. Cheap guard: in `generate.py`, assert no
retrieved id appears in the test id set.
