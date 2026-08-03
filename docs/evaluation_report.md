# Evaluation Report — does multimodal retrieval reduce hallucination?

**ECE 1508 project. Written 2026-08-02, end of Day 5.**
Covers what was built, what was measured, what the numbers mean, and what to do next.

> **One caveat applies to every faithfulness number below.** The LLM judge has **not yet
> been validated against human labels**. `results/validation_sample.csv` is exported and
> blind; until 50 claims are hand-labeled and `--validate` reports agreement, the results
> are *provisional*. If judge agreement turns out poor, the central null result becomes a
> statement about the judge rather than about the system. This is the single highest-value
> outstanding task and it costs $0.

---

## 1. Headline findings

**1. Retrieval is what removes hallucination. Images are not.**

| config | faithfulness | hallucination |
|---|---:|---:|
| **B0** no retrieval | 0.243 | **0.757** |
| **B1** text-only RAG | **0.901** | 0.099 |
| **M** multimodal RAG (α=0.5) | **0.886** | 0.114 |

Adding retrieval cuts hallucination from 76% to ~10% — a **3.7× reduction**. Adding image
fusion on top changes nothing measurable.

*(B0 is over all 129 items; B1 and M are the 90-item paired subset where both arms produced
a genuine summary — see §3.2 for the per-arm figures over all usable items, which differ in
the third decimal. B0 is also scored against evidence it never saw; see §2.2.)*

**2. The B1-vs-M null is bounded, not merely underpowered.**

Paired over 90 items where both arms produced a genuine summary:
difference **−0.0150** (M lower), 95% CI **[−0.0431, +0.0125]**, Wilcoxon **p = 0.252**.
The experiment was powered to detect any effect ≥ **4.0 points**; it measured 1.5. So the
claim is not "we could not tell" but **"if multimodal fusion helps, it helps by less than
4 points."**

**3. The ablation localises the null to both channels independently.**

M differs from B1 in two ways at once. Separating them:

| comparison | isolates | difference | 95% CI | p |
|---|---|---:|---|---:|
| B1 → M-nocap | **retrieval** (different articles) | +0.0046 | [−0.0236, +0.0321] | 0.640 |
| M-nocap → M | **captions** (text in prompt) | −0.0118 | [−0.0393, +0.0135] | 0.530 |
| B1 → M | both combined | −0.0150 | [−0.0431, +0.0125] | 0.252 |

Neither channel contributes, and they are not cancelling — both are individually ~zero.
In particular the **caption channel is inert**: injecting ~93 words of human-written image
description into every prompt moves faithfulness by −1.2 points, p=0.53. The visual
evidence reaches the model and does not matter once there.

**4. Retrieval confidence governs *whether* the model answers, not *how faithfully*.**

| retrieval confidence (top raw `s_text`) | faithfulness | refusal rate |
|---|---:|---:|
| Q1 worst (0.46) | 0.902 | **72.4%** |
| Q2 (0.55) | 0.870 | 41.9% |
| Q3 (0.65) | 0.880 | 26.7% |
| Q4 best (0.74) | 0.925 | **1.3%** |

Spearman(top `s_text`, faithfulness) = +0.146 (p=0.15) for B1 and +0.056 (p=0.59) for M —
neither significant. Refusal, by contrast, is almost perfectly determined by it.

This is the most useful mechanistic result in the project: **better evidence makes the
model willing to answer; conditional on answering it is ~90% faithful regardless.**

---

## 2. What was built

### 2.1 The pipeline (Days 1–4, context)

```
query ─► retrieve() ─► build_prompt() ─► gpt-4o-mini (T=0) ─► summary
             │
   switch:  [text-only]  vs  [text + image fusion]
```

- Corpus: 1,023 BBC news article/image pairs (`RealTimeData/bbc_news_alltime`, 2024-01),
  split 873 index pool / 150 test, frozen and committed.
- Text: SBERT `all-MiniLM-L6-v2` (384-d) over 11,052 passages. Images: CLIP
  `ViT-B-32-quickgelu` (512-d) over 1,023 images. Two FAISS `IndexFlatIP` indexes.
- Fusion: `score = α·s_text + (1−α)·s_img` on per-query min-max normalised streams.
- Generator held **constant** across all arms (`gpt-4o-mini`, `temperature=0`), so any
  difference is attributable to retrieval and not to the generator.
- Abstention guardrail: if `max(s_text)` over the top-k < τ=0.35, return
  `INSUFFICIENT_EVIDENCE` without calling the LLM.

### 2.2 The evaluation harness (Day 5, `src/evaluate.py`)

Two metrics, chosen because **any metric computed on the text stream alone is maximised by
B1 by definition** — B1 *is* the argmax of that stream. A fair comparison needs ground
truth outside both streams. There are exactly two available:

**(a) Recall@k of the withheld gold article.** Free, no API. Uses `query_qa` (a pinpoint
question) with `include_test=True`, since with the gold article unindexed recall is 0 by
construction. Neither arm can tune for it: the headline is indexed in neither stream.

**(b) Claim-level faithfulness via an external LLM judge.** Two passes:

1. **Decompose** — summary → atomic, self-contained claims. Deliberately **blind to the
   evidence**: a decomposer that can see the evidence shapes claim boundaries to fit it,
   and faithfulness is a ratio over that granularity.
2. **Verify** — (evidence, claims) → supported yes/no per claim.

`faithfulness = supported / total`, `hallucination = 1 − faithfulness`. "Supported" means
grounded in the retrieved evidence, **not true in the world** — the judge compares two
texts and needs no outside knowledge, which is why it automates reliably.

Judge design decisions:

| choice | reason |
|---|---|
| Judge = `claude-sonnet-5`, generator = `gpt-4o-mini` | Different model families. A model grading its own output exhibits self-preference bias, and that bias would sit directly on the headline number. |
| Judge blind to the condition label | A judge told "this is the multimodal arm" is not blind. Enforced programmatically on the prompt templates. |
| `thinking: disabled` + JSON-schema-constrained verdicts + prompt-hash caching | `temperature=0` is **unavailable** — the Claude 5 API removed sampling parameters and returns HTTP 400. Determinism recovered by constraining the output surface instead. |
| B0 judged against `union(B1 evidence, M evidence)` | B0 has no evidence of its own. The union is symmetric between arms, so the hallucination ceiling does not depend on either arm's retrieval. It is generous to B0, which is the safe direction for a ceiling claim. |

**Scale:** 513 summaries judged (508 yielded ≥1 claim; 5 were soft refusals that decomposed
to zero claims — see §4.1) → **5,146 atomic claims** → 1,015 judge calls. **Total spend
$7.24** (judge $6.95, generation $0.29).

### 2.3 The ablation

M differs from B1 in **two** ways simultaneously, so a null result cannot say which channel
was inert. A third arm separates them:

| arm | image-fused retrieval | captions in prompt |
|---|---|---|
| B1 | no | no |
| **M-nocap** | **yes** | **no** |
| M | yes | yes |

M-nocap retrieves **identically** to M (verified: same article IDs, same instruction text)
and differs only by removal of the `[IMAGE n: "..."]` lines. So `B1 → M-nocap` isolates the
retrieval channel and `M-nocap → M` isolates the caption channel. Because retrieval is
unchanged, B0's evidence union is unaffected and B0 was served entirely from cache.

---

## 3. Full results

### 3.1 Retrieval recall and the α curve (free, n=150)

| config | @1 | @5 | @10 | B1-only@5 | M-only@5 | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| B1 text-only | 0.733 | **0.913** | 0.940 | — | — | — |
| M α=0.9 | 0.733 | 0.933 | 0.953 | **0** | 3 | 0.250 |
| M α=0.75 | 0.727 | **0.940** | 0.960 | **0** | 4 | 0.125 |
| M α=0.5 *(committed default)* | 0.673 | 0.860 | 0.900 | 12 | 4 | 0.077 |
| M α=0.25 | 0.573 | 0.740 | 0.787 | 30 | 4 | <0.001 |
| M α=0.0 *(pure image)* | 0.353 | 0.600 | 0.693 | 50 | 3 | <0.001 |
| M α=1.0 | 0.733 | 0.913 | 0.940 | 0 | 0 | 1.000 |

**α=0.5 measurably costs retrieval quality** (0.860 vs 0.913 @5). The curve peaks in the
**0.75–0.9** band, where fusion strictly dominates the baseline — a superset of B1's hits
with zero losses — but on only 4 discordant pairs, so p=0.125: the direction is
unambiguous, the magnitude is not established.

**α=1.0 reproduces B1 exactly on all 150 queries.** This is the only empirical proof that
`mode="text"` and `mode="multimodal", α=1.0` really are one code path, so a B1-vs-M
difference cannot be an implementation artefact. Asserted on every run.

### 3.2 Faithfulness (129 paired test items)

| config | items | claims | faithfulness | hallucination | claims/summary | τ-gated | hard refusal | soft refusal | true refusal rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 129 | 1408 | 0.2430 | 0.7570 | 10.91 | 0 | 0 | 0 | 0% |
| B1 | 125 | 1219 | 0.8996 | 0.1004 | 9.75 | 2 | 17 | 36 | 35.3% |
| M | 128 | 1289 | 0.8885 | 0.1115 | 10.07 | 5 | 19 | 35 | 36.0% |
| M-nocap | 126 | 1230 | 0.8963 | 0.1037 | 9.76 | 5 | 20 | 38 | 38.7% |

(Faithfulness column is the usable-only cut — see §4.1 for why.)

### 3.3 The paired comparison

| cut | n | B1 | M | diff | 95% CI | p |
|---|---:|---:|---:|---:|---|---:|
| all judged items | 125 | 0.9103 | 0.8957 | −0.0146 | [−0.0380, +0.0088] | 0.245 |
| **usable only (headline)** | **90** | **0.9010** | **0.8860** | **−0.0150** | **[−0.0431, +0.0125]** | **0.252** |

Both cuts agree in sign and magnitude, which is why the null is trustworthy.

### 3.4 Why the null is credible: the independent variable genuinely moved

On the 90 scored items, M's top-5 overlaps B1's by only **52%** on average; just **5/90**
are identical and **41/90** are nearly disjoint (≤40% overlap). The two arms read
substantially different source material and produced statistically indistinguishable
faithfulness. This is much stronger evidence than a null from two arms that quietly
behaved the same.

### 3.5 Power analysis

| quantity | value |
|---|---:|
| SD of the paired per-item difference | 0.1351 |
| n (paired, usable) | 90 |
| Minimum detectable effect, 80% power, two-sided α=0.05 | **4.0 points** |
| Observed effect | 1.5 points |
| n required to detect an effect of 1.5 points | **636 pairs** |

The experiment was adequately powered for any effect a reader would care about, and the
observed effect is well inside the noise floor.

### 3.6 What a bigger corpus would buy

Projected from the measured scaling curve (median best `s_text` rises **+0.034 per
doubling** of pool size: 0.419 at n=100, 0.492 at n=400, 0.526 at n=873), with refusal
modelled as a logistic function of `s_text`:

| pool size | doublings | predicted refusal | usable pairs | detectable effect |
|---:|---:|---:|---:|---:|
| 873 *(today)* | 0 | 35.7% | 90 | 4.0 pts |
| 3,492 | 2 | 28.7% | 102 | 3.8 pts |
| 6,984 | 3 | 25.5% | 107 | 3.7 pts |
| 13,968 | 4 | 22.5% | 112 | 3.6 pts |
| **55,872** *(proposal scale)* | 6 | **17.3%** | 121 | **3.4 pts** |

**Scaling the index 64× moves the detectable effect only from 4.0 to 3.4 points.** It would
substantially fix the refusal rate, but it would not resolve the B1-vs-M question, because
statistical power comes from the size of the **test set**, not the index. Reaching 636
pairs needs ~1,000 test items — more than the entire current corpus.

---

## 4. Methodological findings

These cost real time to find and are worth reporting as contributions in their own right.

### 4.1 There are three refusal modes, and the third contaminates the metric

Every prior check tested `summary == "INSUFFICIENT_EVIDENCE"`. A third mode exists:

| mode | form | B1 | M |
|---|---|---:|---:|
| hard | the literal `INSUFFICIENT_EVIDENCE` token | 17 | 19 |
| **soft** | prose refusal: *"The evidence does not provide information about X. It covers unrelated topics including A and B."* | **36** | **35** |
| usable | an actual summary | 97 | 96 |

**The true refusal rate is ~35%, not the ~12% reported on Day 3** — a 3× undercount.

Worse, soft refusals **inflate faithfulness**: the decomposer extracts their second
sentence as claims ("The evidence covers a scandal involving the Post Office"), which are
meta-statements *about* the evidence and therefore supported almost by construction.

| | faithfulness |
|---|---:|
| soft refusals (B1 / M) | 0.947 / 0.921 |
| genuine summaries (B1 / M) | 0.900 / 0.889 |

A metric whose value rises when the model declines more is not measuring faithfulness. The
harness now reports both cuts always, and warns if they ever disagree in sign.

This was invisible until Day 5 because nothing before it read the summaries *semantically*.
The decomposer did — five soft refusals reduced to zero atomic claims, which is what
surfaced the mode.

### 4.2 A blindness check that silently shrank the sample

The first implementation scanned rendered judge prompts for condition labels. The word
*"condition"* appears in ordinary news prose (*"the woman's condition"*), so the check
threw on 11 of 387 summaries, which the runner skipped. The run completed, wrote a
plausible output file, and reported n=125 as if it were 129, with nothing saying so.

Fixed by checking the prompt **templates** rather than rendered prompts: summaries and
evidence are the *material under judgment*, not instructions. The runner now counts
failures and exits non-zero rather than reporting on a partial run.

### 4.3 Both obvious retrieval statistics are rigged

`max s_text` over the retrieved set is identical in both arms on 109/150 queries (73%),
because B1 ranks purely by `s_text` so its max is the pool-wide maximum. `mean s_text` over
the retrieved set separates the arms on 145/150 — but B1 selects the k highest-`s_text`
articles, so its mean is the **maximum achievable over any k-subset**, and M is ≤ B1 by
identity (measured: 145 lower, 5 tied, **0 higher**). Presented as "evidence quality" this
would make the method under test look worse by construction.

**General rule: any metric computed on the text stream alone is won by B1 by definition.**

### 4.4 `temperature=0` is not available on the judge

The Claude 5 family removed `temperature`/`top_p`/`top_k`; a non-default value returns HTTP
400. Determinism was recovered through disabled thinking, a JSON schema that constrains the
verdict to a boolean, and prompt-hash caching that makes re-runs bit-identical and free.
The generator is unaffected and still runs at `temperature=0`.

---

## 5. Limitations

1. **The judge is unvalidated.** Highest-priority open item. Everything is provisional.
2. **Faithfulness grades each arm against its own evidence.** It measures *"did the model
   stick to what it was given"*, not *"was what it was given any good"*. A system that
   retrieves irrelevant articles and faithfully summarises them still scores ~0.9. This is
   the deepest reason B1 and M land within 1.5 points of each other despite reading
   half-different articles — **the yardstick moves with the arm**. Recall@k is the metric
   sensitive to retrieval quality, and the two answer different halves of the question.
3. **The model never sees pixels.** "Multimodal" here means (a) image similarity influences
   *which* articles are retrieved and (b) human-written captions are injected as text. A
   vision-capable generator was never tested. §6.2 addresses this.
4. **Ceiling effect.** B1 already achieves recall@5 = 0.913, leaving ≤8.7 points of
   headroom. There is very little room for images to demonstrate value on this corpus.
5. **Faithfulness measured at α=0.5 only**, which the recall sweep shows is a poor operating
   point. The hypothesis is untested at α=0.75–0.9. Given that faithfulness is flat against
   retrieval quality (§1.4), a null is expected there too, but it is not measured.
6. **Recall@k and generation use different retrieval settings.** Recall uses
   `include_test=True` (the gold article must be reachable); generation uses
   `include_test=False` (the source article is deliberately withheld). They are not directly
   comparable.
7. **n=90 for the headline comparison**, after refusals. Bounded null, not a proven zero.
8. **Single corpus, single month, single language.** BBC, January 2024.
9. **`story_type` stratification was dropped.** BBC's `section` field is geographic, not
   topical; 68% maps to "other". Would need hand-labelling.

---

## 6. Proposed work — a 1.5 to 2 week programme

Ranked by information gained per unit of cost and effort. Estimated API spend assumes
current Sonnet 5 introductory pricing; the whole programme fits inside ~$50.

### Tier 0 — do these first, they are nearly free

**0.1 Validate the judge.** 50 blind hand-labelled claims, already exported. Report
agreement and Cohen's κ. **$0, ~30 min.** Every number in this report is conditional on it.

**0.2 Second-judge robustness.** Re-judge the same 5,146 claims with `claude-haiku-4-5` and
report inter-judge agreement. Turns "we trusted one judge" into a measured claim, and
hedges the risk that 0.1 comes back weak. **~$2.70, ~10 min.**

**0.3 Hallucination error taxonomy.** The ~10% unsupported claims already have written
judge rationales in `results/claims.csv`. Cluster them by failure type — entity
substitution, date/number error, unsupported causal link, world-knowledge intrusion. A
qualitative table of *what kind* of hallucination survives retrieval is a strong report
section. **$0–1, ~2 hours**, mostly reading.

### Tier 1 — the experiments that would most change the conclusion

**1.1 A genuinely multimodal generator.** *This is the most important gap.* The current
"multimodal" arm never shows the model an image; captions are the only visual channel and
the ablation proves they are inert. `gpt-4o-mini` is vision-capable, so a fourth arm — same
retrieval as M, but the top-k **images passed as image content blocks** rather than caption
text — tests the actual hypothesis rather than a caption-mediated proxy.

- Isolates: does visual information help when the model can actually see it?
- Keeps the generator family constant, so it remains comparable.
- **~$3–5, ~1 day** including prompt plumbing and a judge pass.
- Strongest possible outcome either way: a positive result rescues the hypothesis; a null
  makes "multimodal retrieval does not help news summarisation faithfulness" a much
  broader and better-supported claim.

**1.2 A retrieval regime where text is weak.** The null may be a ceiling effect — text
retrieval already succeeds 91% of the time, so images have almost nothing to add. Construct
conditions where text alone should struggle and see whether fusion rescues them:

- **Visual-first queries**: derive queries from the *image content* rather than the
  headline topic (e.g. "aerial photo of flooded farmland") — a regime where CLIP should
  outperform SBERT.
- **Degraded text**: truncate passages to the first sentence, or add noise, simulating
  low-text-signal retrieval.
- Report faithfulness and recall as a function of text-signal strength.
- **~$5, ~2–3 days.** This is the most intellectually interesting experiment available and
  directly probes *when* multimodality would matter, converting a flat null into a
  conditional finding.

**1.3 An evidence-coverage metric.** Faithfulness cannot detect better retrieval (§5.2).
Add a metric that can: judge `(query, evidence) → does this evidence contain what is needed
to address the query?` This is arm-comparable, sensitive to retrieval quality, and needs no
reference summary — so it avoids the leakage problem that killed ROUGE-L.

- Pairs naturally with faithfulness: *coverage* = did we retrieve the right thing;
  *faithfulness* = did we stay grounded in it.
- **~$1–2, ~half a day.**

### Tier 2 — scale, if time allows

**2.1 Enlarge the test set to reach real statistical power.** The bottleneck is n=90, not
index size. `RealTimeData/bbc_news_alltime` has many months available; using 6 months gives
roughly 6,000 articles → ~1,000 test items → ~600 usable pairs, which is the 636 needed to
resolve an effect the size actually observed.

- Simultaneously fixes the refusal rate (bigger pool) and the power problem (bigger test
  set) — the only intervention that addresses both.
- Cost scales with test items: ~600 pairs × 3 arms ≈ **$25–35 judge**, plus ~$1 generation.
- **~2–3 days**, mostly data plumbing and image downloads (the slow step, ~0.34 s/image
  serially — parallelise it).
- Note this invalidates the committed corpus and every number above, so it is a
  "do it properly and re-run everything" option, not an increment.

**2.2 Hybrid sparse+dense retrieval.** BM25 + SBERT fusion is a standard strong baseline and
would raise `s_text`, cutting the 35% refusal rate and increasing usable n at zero API cost.
It also makes B1 a *harder* baseline, which is the honest thing to do before claiming
anything about M. **$0 API + ~$7 to re-run everything, ~1 day.**

**2.3 Cross-encoder re-ranking.** Re-rank the top-50 fused candidates with a small
cross-encoder before taking the top-5. Cheap, local, and typically a large recall gain.
**$0 API, ~half a day.**

### Tier 3 — completeness, low priority given the findings

**3.1 Faithfulness at α=0.75 and 0.9.** ~$3.60, ~15 min. Closes the "untested at optimal α"
gap. Given faithfulness is flat against retrieval quality, expect a null. Note that
selecting α on the same test set you report on is tuning on test and must be declared.

**3.2 Story-type stratification.** Hand-label the 150 test items `event_centric` /
`abstract_topical` and report the comparison per stratum — the hypothesis is that images
help more for event-centric stories. n≈50/bucket is underpowered, so treat as exploratory.
**$0, ~1 hour of labelling.**

### Suggested two-week schedule

| days | work |
|---|---|
| 1 | Tier 0 in full (judge validation, second judge, error taxonomy) |
| 2–3 | 1.3 evidence-coverage metric + 1.1 vision-capable generator arm |
| 4–6 | 1.2 weak-text retrieval regime — the headline new experiment |
| 7–8 | 2.2 hybrid retrieval + 2.3 re-ranking; re-run the full evaluation |
| 9–11 | 2.1 corpus/test-set scale-up if the earlier results justify it |
| 12–14 | Write-up, plots, deploy, buffer |

**Recommended minimum** if time compresses: Tier 0 + 1.1 + 1.3. That is about **$8 and
three days**, and it upgrades the current result from "no effect observed" to "no effect
observed, with the mechanism identified and the obvious confound eliminated."

---

## 7. Reproducing these numbers

```bash
source .venv/bin/activate

python -m src.evaluate --recall               # free, no API      -> results/recall.csv
python -m src.evaluate --dry-run              # inspect judge request, sends nothing
python -m src.evaluate --judge                # cached; re-run costs $0
python -m src.evaluate --report               # -> results/metrics.csv
python -m src.evaluate --sample-validation    # 50 blind claims for hand-labelling
python -m src.evaluate --validate             # agreement + Cohen's kappa

# the ablation arm
python -m src.generate --run --configs M_nocap --out results/summaries_ablation.csv
```

| artefact | contents |
|---|---|
| `results/summaries.csv` | 450 rows — B0/B1/M generation output, the Day 3→5 interface |
| `results/summaries_ablation.csv` | 150 rows — the M-nocap arm |
| `results/claims.csv` | 5,146 per-claim judge verdicts with rationales |
| `results/metrics.csv` | the headline table |
| `results/recall.csv` | the α sweep |
| `results/validation_sample.csv` | 50 blind claims awaiting hand labels |

All LLM calls are cached by prompt hash in `data/llm_cache/` (gitignored), so every command
above re-runs for **$0** and reproduces identical numbers.

Design rationale for every decision referenced here is in `docs/decisions.md` (D4–D15).
