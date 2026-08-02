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
