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
| Same judge, same prompt, same decoding across B0/B1/M | Same reason the generator is held constant. **`temperature=0` is unavailable — see D13.** The Claude 5 family removed the parameter and returns 400; determinism comes from disabled thinking + a constrained JSON schema + prompt-hash caching. Do not add it back. |
| Judge sees only `(evidence, claim)` — never the condition label | A judge told "this is the multimodal one" is not blind. Enforced on the prompt **templates**, not rendered prompts (D13) |
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

### Resolved 2026-08-02 (Day 5): ROUGE-L is dropped
`evaluate.py` reports **faithfulness + recall@k only**. `rouge_score` stays installed but
unused. The deciding argument is reason 2 above, which no choice of reference repairs: a
reference derived from the test article makes "retrieve the source and copy it" the
highest-scoring behaviour, so ROUGE-L would rank a leaking system above a correct one —
and D1/D7 Rule 3 exist specifically to prevent that leak. Reporting a metric that rewards
what the rest of the design forbids is worse than reporting one metric fewer.

Report line: *"ROUGE-L was dropped rather than computed against a reference the retrieval
split deliberately withholds; see limitations."* Listed as future work in the
coverage-based form D5 sketches (references derived from the retrievable pool).

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

---

## D8 — Query text = LLM-rewritten headline, in two variants (closes `day2_guide.md:88`)

**Status:** decided 2026-08-02, Day 3. Built: `src/queries.py` →
`data/processed/queries.parquet` (committed). Used by generation now and by Day 5 recall@k.

### The decision
Each of the 150 test items gets **two** rewrites of its headline, both by `gpt-4o-mini`:

| Column | Form | Used by | Why that form |
|---|---|---|---|
| `query_qa` | pinpoint question | Day 5 recall@k | names the gold article's subject, so "did the retriever find it?" is sharp |
| `query` | broader topic phrase | **generation** | pool coverage is genuinely about it, so summarization is well-posed |

`mode="headline"` remains available as the documented zero-cost fallback.

### Why two columns and not one
The first build used the pinpoint question for both, and it failed measurably: `gpt-4o-mini`
returned `INSUFFICIENT_EVIDENCE` on **46% of B1 and 44% of M** items, collapsing the paired
comparison to 74/150. Not a retrieval bug — the refused items returned visibly on-topic
articles at `s_text` up to 0.842.

The cause is structural. D1 withholds the source article from the generation index, so a
question naming that article's unique subject asks for a fact the pool provably does not
contain. Generation and recall@k want opposite things from a query, and one column cannot
serve both.

Broadening is **subtractive only** — drop the private individual's name, keep the story
type, keep public figures, organisations and places. Nothing is added, so a query cannot
smuggle in information the retriever would otherwise have to find, and it stays neutral
between the arms. Verified: the B1-vs-M retrieval contrast is **unchanged at 55%** of M's
top-5 articles not appearing in B1's, so broadening did not wash out the independent
variable.

An early version of the topic prompt over-broadened — it dropped *Trump*, *Prince Andrew*
and *Met Police*, which would have wrecked both recall@k and the contrast. Fixed with
explicit keep/drop rules and three few-shot examples covering both cases. Worth knowing if
the prompt is ever retuned.

### Why rewrite the headline at all
`day2_guide.md:88` left this open and flagged both obvious options as rigged:

| Query = | Rigged toward | Mechanism |
|---|---|---|
| headline verbatim | **B1** | headline tokens recur through the body; SBERT saturates |
| `caption` field | **M** | the caption describes the image the M arm searches |

The headline is the one field indexed in **neither** stream — passages come from `body`,
image vectors from pixels alone (D3). Rewriting it keeps that property while removing the
verbatim token overlap that hands B1 a free win. Both arms bridge the same gap. 300 calls
total, about a cent, cached by prompt hash like everything else.

### Why this had to be settled on Day 3, not Day 5
The handoff files it as a Day 5 item because that is when *recall@k* needs it. But D7 Rule 2
requires `test_id | config | query | evidence | summary`, so generation needs a query per
test item too — and it turned out to need a *different* one, which is exactly the kind of
thing you want to discover on Day 3 rather than Day 5.

### Verification
150/150 non-empty, 0 duplicates, median 6 words for `query`. Spot-checked by eye: the
rewrites drop the `" - BBC News"` suffix, add no facts, and keep the entities that matter.

| Headline | `query` (generation) | `query_qa` (recall@k) |
|---|---|---|
| *Trump challenges his 'arbitrary' removal from Maine's ballot* | Trump challenges removal from Maine ballot | Why did Trump challenge his removal from Maine's ballot? |
| *Reed Wischhusen jailed for planning 'revenge' mass shooting* | man jailed for planning revenge mass shooting | Why was Reed Wischhusen jailed for planning a mass shooting? |

`queries.parquet` lands in `data/processed/`, which is **committed** — same reason as the
corpus. A teammate regenerating it would get different rewrites and non-comparable numbers.
`src/queries.py` refuses to overwrite without `--force`.

---

## D10 — The generation task is summarization, and the prompt is relevance-tolerant

**Status:** decided 2026-08-02, Day 3. `generate.py:INSTRUCTIONS`. Applies to B1 and M
identically, so it cannot confound the comparison (D7 Rule 1).

### The decision
Two properties of the shared instruction text, both arrived at by measurement:

1. **The task is summarization, not question answering** — `plan_7day.md:82` says
   *"Summarize ONLY the numbered evidence"*. Phrased as *"answer the question"*, the model
   refuses whenever the evidence lacks the one specific fact asked for.
2. **Partial relevance is explicitly sufficient.** The prompt states that retrieved evidence
   will rarely match the query exactly, that same-subject coverage should be summarized
   anyway, and that `INSUFFICIENT_EVIDENCE` is reserved for evidence about an *entirely
   different* subject.

### Why point 2 exists — measured, and not obvious
With a plain *"if the evidence is unrelated, reply INSUFFICIENT_EVIDENCE"* rule, the model
read "no article about *this exact event*" as "unrelated" and refused on ~48% of items —
including cases where retrieval had clearly succeeded. The worst example:

> query *"pro-Israel rally calling for hostages' release in London"*, top `s_text` **0.657**,
> evidence = five articles on the Gaza war, hostage negotiations, Israeli aid protests, and a
> London march. Refused.

That is over-refusal, not caution. Adding the relevance-tolerance rules took refusal to
**12% B1 / 15% M** with no other change.

### Refusal rate by configuration, measured on 40 items
| Query form | Instruction | B1 refused | M refused | paired usable |
|---|---|---|---|---|
| pinpoint question | answer the question | 46% | 44% | 74/150 |
| pinpoint question | summarize | 50% | 55% | 18/40 |
| topic phrase | summarize | 48% | 48% | 20/40 |
| **topic phrase** | **summarize + relevance-tolerant** | **12%** | **15%** | **34/40** |

Note the middle two rows: neither change alone did anything. Only the combination worked,
which is why the first two attempts looked like dead ends.

### The residual ~13% is real and should be reported, not tuned away
The remaining refusals are genuine topic misses — *Rochdale Labour MP dies*, *giraffe
relocation*, *Sea Eagle sighting in Gwynedd*. The pool is 873 articles from one month of BBC
news and simply contains no coverage of these stories. That is a **corpus-size limitation**,
and refusing is the correct behaviour. Report it as the honest floor; do not weaken the
prompt further to chase it, or the model starts inventing to fill gaps — the exact failure
this project measures.

### Risk accepted
A more permissive prompt could raise hallucination in all arms. That is acceptable and
arguably desirable: the instruction is byte-identical across B1 and M (asserted by
`_check_prompt_parity()`), so it shifts both arms equally, and a task where the evidence is
adjacent-but-not-identical to the withheld article is a *stricter* hallucination probe than
one where the answer sits in front of the model.

---

## D9 — Abstention threshold tau = 0.35 on raw `s_text`

**Status:** decided 2026-08-02, Day 3. `generate.py:TAU`. Re-derive with
`python -m src.generate --tau-scan`.

### The statistic
Gate on `max(h.s_text for h in hits)` — the strongest **raw** cosine in the returned set.

Two choices inside that, both deliberate:

- **Raw `s_text`, never `score`.** `score` is min-max normalized per query, so the top hit is
  ~1.0 on *every* query no matter how poor the match. Threshold on it and you never abstain.
- **Max over the top-k, not `hits[0]`.** Gating on the top-ranked hit makes the gate
  rank-order dependent, and M's ranking is partly image-driven. The two arms would then
  abstain at different rates for a reason that is not evidence quality — a confound
  introduced by the guardrail itself.

### Why 0.35 — measured, not chosen by taste
Two populations, both measured on 2026-08-02:

| Population | n | range | 1st pct | median |
|---|---:|---|---|---|
| Real test queries (B1) | 150 | 0.370–0.842 | 0.373 | 0.559 |
| Real test queries (M) | 150 | 0.340–0.842 | 0.356 | 0.546 |
| Deliberately off-topic probes | 8 | 0.19–0.365 | — | ~0.24 |

Off-topic probes were ordinary non-news questions (*sourdough*, *derivative of a sigmoid*,
*cheap flights to Osaka*). **The two populations overlap in a narrow 0.34–0.37 band, so no
threshold separates them cleanly** — report this, don't tune it away.

| tau | off-topic caught | B1 false abstain | M false abstain |
|---|---|---|---|
| 0.35 | 6/8 | **0.0%** | **0.7%** |
| 0.40 | 8/8 | 4.7% | 7.3% |
| 0.45 | 8/8 | 16.0% | 18.0% |

**0.35 chosen.** Every abstention drops an item from the paired B1-vs-M faithfulness
comparison, which is the entire result; keeping false abstention near zero is worth more than
catching two borderline probes. It also keeps B1 and M scored on effectively the same 150
items, which is what makes the comparison paired (D6).

### What this means for the report
Abstention is a **demo-time guardrail**, not a test-set phenomenon. Every test query is by
construction a real question about a real event in this corpus, so a well-set tau *should*
almost never fire there. Its value shows in the Day 4 Streamlit demo, where a user can type
anything. Report the off-topic-probe table as the evidence it works, not the test-set rate.

---

## D11 — alpha stays 0.5 for the committed run; sweep it on Day 5

**Status:** decided 2026-08-02 (Day 4). Affects Day 5 (`src/evaluate.py`).

### The decision
`alpha = 0.5` remains the default for `retrieve()`, the demo, and the committed
`results/summaries.csv`. Day 5 sweeps it (0.5 -> 0.9) and reports the curve.

### Why this needed deciding
alpha = 0.5 came from `plan_7day.md:63` as a round-number default. It was never measured.
Day 4 measured it, using recall of the withheld gold article — ground truth external to
both score streams, so neither arm is optimising for it (`query_qa`, `include_test=True`,
D8, n=150):

| config | @1 | @5 | @10 |
|---|---:|---:|---:|
| B1 text-only | 0.733 | **0.913** | 0.940 |
| M alpha=0.75 | 0.727 | **0.940** | 0.960 |
| **M alpha=0.5 (committed)** | 0.673 | **0.860** | 0.900 |
| M alpha=0.25 | 0.573 | 0.740 | 0.787 |

Paired, McNemar exact at k=5:

| comparison | B1-only wins | M-only wins | p |
|---|---:|---:|---:|
| M alpha=0.75 vs B1 | **0** | 4 | 0.125 |
| M alpha=0.5 vs B1 | 12 | 4 | 0.077 |

**alpha = 0.5 measurably costs recall. alpha = 0.75 strictly dominates B1** — a superset
of B1's hits, zero losses. But 4 discordant pairs is **underpowered**: p = 0.125, not
significant. The direction is unambiguous; the magnitude is not established. Report both
facts, not just the favourable one.

### Why not switch to 0.75 now
1. **Recall is not the research question.** The headline result is summary faithfulness.
   A lower-recall set can be *more* faithful — fusion pulls in topically adjacent
   articles, and that adjacency is exactly where hallucination shows up. Choosing alpha on
   recall would optimise a proxy for the thing being measured.
2. **`results/summaries.csv` (450 rows) is at alpha=0.5.** Switching invalidates it,
   costs ~$0.22 to regenerate, and mixing rows from two alpha values silently voids the
   paired comparison.
3. Picking alpha on the same test set the result is reported on is tuning on test. If
   Day 5 does select an alpha, say so explicitly as a limitation.

### Rules
- One alpha per reported table. Never mix.
- `--alpha` is already a flag on `python -m src.generate`.
- The sweep is cheap for recall (no API) and ~$0.22 per alpha for faithfulness.

---

## D12 — what the demo's retrieval statistics may and may not claim

**Status:** decided 2026-08-02 (Day 4). Affects `app/streamlit_app.py` and any B1-vs-M
metric anywhere.

### The decision
The demo shows two per-arm retrieval numbers, `gate score` and `mean text sim`, both
explicitly labelled as diagnostics. **Neither is presented as a quality comparison**, and
the app says so on screen.

### Why — both obvious statistics are rigged, in opposite directions
**`gate score` = `max s_text` over the top-k.** This is the tau statistic (D9); it is on
screen only to explain the abstention state. As an arm comparison it is near-useless:
B1 ranks purely by `s_text`, so its max is the **pool-wide maximum** regardless of k, and
M matches it whenever that article survives fusion. **Identical in both arms on 109/150
test queries (73%)**, and pinned across the whole slider range whenever one article is
the argmax of both streams. Side by side it reads as a broken UI.

**`mean s_text` over the retrieved set** was the fix, and it is worse. It moves with alpha
and separates the arms on 145/150 (97%), so it looks informative — but B1 selects the k
highest-`s_text` articles, so its mean is the **maximum achievable over any k-subset**:

| | count |
|---|---|
| M mean `s_text` **<** B1 | 145/150 |
| M mean `s_text` **=** B1 | 5/150 |
| M mean `s_text` **>** B1 | **0/150** |

M cannot win, at any alpha, on any query. It was briefly labelled **"evidence quality"**,
which turns a mathematical identity into an apparent finding that the method under test
is worse than the baseline. Renamed `mean text sim`.

### The general rule — applies well beyond the demo
**Any metric computed on the text stream alone is maximised by B1 by definition, because
B1 is the argmax of that stream.** A fair B1-vs-M comparison requires ground truth
*outside* both streams:

- recall of the withheld gold article (D11), or
- an external faithfulness judge (D4).

Nothing computed from `s_text` or `s_img` alone can answer the research question. If a
B1-vs-M number comes out clean and one-sided, check first whether the metric is defined
on the stream one arm ranks by.

### Related: the tau gate is mildly arm-asymmetric
Same root cause. B1's gate statistic is the pool-wide max, so its gate asks "does *any*
pool article clear tau" — a coverage test. M's asks "does any article *in M's fused
top-5* clear tau", which is stricter. Measured: M gates 5/150, B1 gates 2/150, disagreeing
on 3. **3 of M's 5 gates are this artifact, not evidence quality.** D9 chose max-over-top-k
rather than `hits[0]` to reduce rank-dependence and it does, but the retrieved *set* is
still arm-dependent. 2% of items; document it, do not change it — changing it invalidates
the committed run.

---

## D13 — the judge cannot run at temperature=0; determinism comes from elsewhere

**Status:** decided 2026-08-02 (Day 5). Amends D4. `src/evaluate.py`.

### The problem
D4 specifies the judge runs with `temperature=0`, for the same reason the generator does:
without it, a B1-vs-M difference could be sampling noise. **That is not available on
`claude-sonnet-5`.** The Claude 5 family removed `temperature`, `top_p` and `top_k`; a
non-default value returns **HTTP 400**. There is no flag to re-enable it.

This is not a reason to change judge model. It is a reason to get determinism another way.

### What replaces it
| Mechanism | What it removes |
|---|---|
| `thinking={"type": "disabled"}` | No sampled reasoning preamble. Also bounds output tokens, and therefore cost. |
| `output_config.format` = a json_schema | The verdict is a **boolean in a constrained schema**. The only free-text surface left is a <=15-word reason that no metric reads. Variance cannot reach the number. |
| Cache by prompt hash (`data/llm_cache/judge_*.json`) | A re-run reproduces the previous run exactly, for free. For a number that goes in a report, reproducibility on re-run is what temperature=0 was actually buying. |

The generator is **untouched** — `generate.py` still calls `gpt-4o-mini` at
`temperature=0`, where the parameter is still accepted and still load-bearing. Different
model families for generator and judge, which was D4's whole point, is preserved.

### Do not "fix" this
Adding `temperature=0` back to the judge call fails the entire run with a 400. The
constant is deliberately absent from `evaluate.py`, and the module docstring says why.

### What the report should say
*"The judge is claude-sonnet-5 with sampled reasoning disabled and verdicts constrained to
a JSON schema; the Claude 5 API does not accept a temperature parameter, so determinism is
enforced by output constraint and prompt-hash caching rather than by temperature=0. The
generator remains gpt-4o-mini at temperature=0."*

### Blindness is checked on the prompt templates, not on rendered prompts
D4 Rule 2 says the judge never sees the condition label. The check enforcing it
(`_assert_prompts_blind()`) runs against the **templates this module authors**, once at
import — not against rendered prompts. Found the expensive way: scanning the rendered
prompt matched the word *"condition"* inside real BBC summaries (*"the woman's
condition"*) and threw on **11 of 387 summaries**, which the runner skipped. The run
completed, wrote a plausible `claims.csv`, and reported n=125 instead of 129 with nothing
saying so — Day 3's silent-sample-shrink, reproduced exactly.

Summaries and evidence are the **material under judgment**, not instructions; a news
article containing the word "condition" is not a label leak. What must be free of labels
is the wrapper around it. Related hardening: `run_judge()` now counts failures and exits
non-zero rather than reporting on a partial run.

### The blindness that is not achievable, and must be disclosed
The judge is blind to the **label**, not to the evidence. M's evidence block carries
`[IMAGE n: "..."]` caption lines and B1's does not, so a judge could in principle infer
the arm from the evidence itself. Stripping the captions is strictly worse — the judge
must score claims against the evidence that was actually in the generator's prompt (D7
Rule 2), and a claim drawn from a caption would become unsupportable by construction.
**State this as a limitation; do not engineer around it.**

---

## D14 — B0's claims are judged against the union of B1's and M's evidence

**Status:** decided 2026-08-02 (Day 5). `evaluate.py:merge_evidence()`.

### The problem
Faithfulness is defined as grounding in the retrieved evidence. B0 has no retrieved
evidence — it is the no-retrieval ceiling. Claim-level faithfulness is therefore
undefined for B0 unless it is given something to be judged against.

### The decision
Judge B0's claims against `union(B1 evidence, M evidence)` for the same query, deduped by
headline and renumbered (measured: 5-10 articles, mean 7.6).

**Why the union and not one arm's evidence.** Judging B0 against B1's evidence alone makes
the hallucination *ceiling* a function of B1's retrieval — change B1 and the ceiling moves,
for a reason that has nothing to do with B0. The union is symmetric between the arms, so
neither arm's number is compared against a yardstick built from itself.

**The union is generous to B0, deliberately.** A larger grounding set can only raise B0's
faithfulness. That is the safe direction for a ceiling claim: if B0 still hallucinates more
than both arms while being scored against a **superset** of their evidence, the conclusion
is stronger, not weaker. If B0 instead scores *well*, that is a real finding about the
corpus (one month of BBC news the model may have seen in pretraining), not a bug.

### What the report must say
B0's number is not the same quantity as B1's and M's. B1 and M are scored against the
evidence that was in their own prompt; B0 is scored against evidence it never saw. Label
it explicitly as *"B0 claims scored against the pooled retrieved evidence for the same
query"* and never present the three as one homogeneous column without that note.

---

## D15 — there are THREE refusal modes; the third contaminates faithfulness

**Status:** found 2026-08-02 (Day 5), during the first real judge run. Amends D10 and the
Day 3 refusal figures. `evaluate.py:refusal_kind()`.

### The finding
D10 and both handoffs warn that a refusal rate must read the summary **text**, not the
`abstained` flag. Correct — but the check they prescribe (`summary == INSUFFICIENT_EVIDENCE`)
only catches the token. A third mode exists and passed every Day 3 and Day 4 check:

| mode | what it looks like | B1 | M |
|---|---|---:|---:|
| **hard** | the literal `INSUFFICIENT_EVIDENCE` token | 17 | 19 |
| **soft** | prose refusal: *"The evidence does not provide information about X. It covers unrelated topics including A and B."* | **36** | **35** |
| usable | an actual summary of the retrieved coverage | 97 | 96 |

**The true refusal rate is ~35%, not the ~12% reported on Day 3.** The earlier figure
undercounted by roughly 3x. Report the corrected number and say so.

This was invisible until Day 5 because nothing before it read the summaries *semantically*.
The decomposer did: 5 soft refusals reduced to **zero atomic claims**, which is what
surfaced the mode. `--report` now prints those explicitly rather than letting n shrink.

### Why it is not just a miscount
Soft refusals **inflate faithfulness in both arms**. The decomposer extracts their second
sentence as claims — *"The evidence covers a scandal involving the Post Office"* — which
are meta-statements **about** the evidence and therefore supported almost by construction:

| | faithfulness |
|---|---:|
| soft refusals, B1 / M | 0.947 / 0.921 |
| genuine summaries, B1 / M | 0.900 / 0.889 |

A metric whose value rises when the model declines more is not measuring faithfulness.

### The decision
`report()` prints **both cuts, always** — all judged items, and usable-only. The headline
is the usable-only number. If the two ever disagree in **sign**, the apparent B1-vs-M
difference tracks refusal rate rather than retrieval and must not be reported as a result;
`report()` prints a warning in that case.

Measured here, they agree and barely move (soft refusals hit both arms nearly equally,
28 B1 vs 32 M among judged items):

| cut | n | B1 | M | diff | 95% CI | p |
|---|---:|---:|---:|---:|---|---:|
| all judged | 125 | 0.9103 | 0.8957 | -0.0146 | [-0.0380, +0.0088] | 0.245 |
| **usable only** | **90** | **0.9010** | **0.8860** | **-0.0150** | **[-0.0431, +0.0125]** | **0.252** |

### What is NOT changed
`paired_ids()` still defines the paired set by **hard** refusals only, keeping the
published 129-item Day 3 interface intact. Widening it there would silently redefine what
every earlier number referred to. Soft refusals are handled as a reported cut, never as a
quiet sample shrink.

### Root cause, and why it is arguably correct behaviour
D10 made the prompt relevance-tolerant to stop over-refusal, and reserved the token for
evidence about "an entirely different subject". Faced with partially-relevant evidence the
model splits the difference: it neither emits the token nor invents content, and instead
describes what the evidence *does* cover. That is honest behaviour and the desired failure
mode — the fault is in the measurement, not the generator. Do **not** retune the prompt to
chase it; that is how a model starts inventing to fill gaps.

---

## D16 — the caption ablation (M_nocap), and what it establishes

**Status:** run 2026-08-02 (Day 5), $1.60. `generate.py:ABLATION`, `evaluate.py:ABLATION`.

### Why it was needed
M differs from B1 in **two** ways at once: image fusion changes *which* articles are
retrieved, and captions add *text* to the prompt. A null B1-vs-M result therefore cannot
say which channel was inert, or whether one helped and the other hurt and they cancelled.

### The design
A third arm, `M_nocap`: **identical retrieval to M** (same mode, same alpha — verified same
article ids) with only the `[IMAGE n: "..."]` lines removed. Asserted before running: same
5 articles, byte-identical `INSTRUCTIONS` (D7 Rule 1), and M_nocap's evidence equals M's
minus the caption lines and nothing else (~93 words per prompt).

Because retrieval is unchanged, M_nocap contributes no new articles to B0's evidence union
(D14), so B0's prompts are unchanged and B0 was served entirely from cache — the ablation
billed only its own 126 summaries.

`paired_ids()` deliberately still ranges over `CONFIGS = (B0, B1, M)` only. Adding the
ablation there would have redefined the 129-item paired set and silently changed what every
earlier number referred to.

### Result — both channels are independently null
| comparison | isolates | n | diff | 95% CI | p |
|---|---|---:|---:|---|---:|
| B1 → M_nocap | retrieval channel | 89 | +0.0046 | [−0.0236, +0.0321] | 0.640 |
| M_nocap → M | caption channel | 90 | −0.0118 | [−0.0393, +0.0135] | 0.530 |
| B1 → M | both | 90 | −0.0150 | [−0.0431, +0.0125] | 0.252 |

Not cancellation — both are individually ~zero and the combined effect is roughly their sum.

**The caption channel is inert.** ~93 words of human-written image description per prompt
moves faithfulness by −1.2 points, p=0.53. This is the load-bearing result: it rules out
"the captions never reached the model" as an explanation for the B1-vs-M null. They reached
it and did not matter.

One incidental effect the captions *do* have: M_nocap refuses slightly more (38.7%) than M
(36.0%), so caption text gives the model marginally more to hold onto before declining. It
is not a faithfulness effect.

### What this licenses in the report
The claim upgrades from "no effect observed" to **"no effect, localised to both channels"**.
It also sharpens the remaining limitation: the model never sees pixels, so what has been
falsified is *caption-mediated* multimodality, not multimodality as such. A vision-capable
generator arm is the natural next experiment (`docs/evaluation_report.md` §6, Tier 1.1).

---

## D17 — B0 rebuilt on the final-test split and judge, paired with E12b (E14)

**Status:** decided 2026-08-14. `src/research_b0.py`. Amends D14.

### The problem
The report's headline "retrieval removes hallucination" claim rested on the inherited B0
number (0.243 faithfulness, `docs/baseline_summary.md`) sitting next to E12b's B1/M/M_vision
numbers (0.864/0.874/0.898) as if they were one comparison. They are not: inherited B0 was
scored on `main`'s random pool/test split by a different judge pipeline than the one E12b
uses, before the group-safe roles (D5, E05) or the gpt-5.6-luna judge substitution (D4's
practical-friction note) existed. Reported side by side, this reads as a single controlled
result when it silently splices two different pipelines.

### The decision
Regenerate B0 with `gpt-4o-mini` on the identical 150 final-test queries E12b used, judge it
with the same `gpt-5.6-luna` judge already scoring B1/M/M_vision, against
`union(B1 evidence, M evidence)` per D14. Recorded as E14, paired with E12b rather than
replacing it.

### Result
Faithfulness 0.146 (150/150 usable — B0 has no retrieval-confidence gate to abstain against,
so unlike B1/M/M_vision it never refuses). Paired $B_1-B_0 = +0.586$, 95% CI
$[0.525, 0.643]$, $n=77$ — the largest, least ambiguous effect in the whole study, and now
computed on a pipeline consistent with the rest of the ladder. This *lowered* the reported
B0 faithfulness relative to the inherited 0.243, i.e. the fix went against the direction that
would have flattered the headline result.

### What the report must say
Table 2 (`report/report.tex`) now includes $B_0$ as a row scored on the same generator,
judge, and split as $B_1$/$M$/$M_{\text{vision}}$, with an explicit note that its usable
count is not comparable to theirs (no abstention gate). `docs/baseline_summary.md` and
`docs/evaluation_report.md` are untouched — they are dated, explicitly-frozen snapshots of
`main` before this branch existed, not claims about the current pipeline.
