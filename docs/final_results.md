# Final research results

## Question and outcome

This project asks whether actual visual evidence improves grounded news
summarization beyond a strong text RAG system and a caption-mediated multimodal
system. It now implements end-to-end pixel multimodality: CLIP image embeddings
affect retrieval, and the vision-capable generator and evaluator receive the
retrieved JPEGs themselves.

The answer is conditional rather than a universal win. Actual pixels produced a
large improvement on a deliberately image-sensitive development sample, and on the
full 150-article final-test role every paired comparison favours the multimodal
systems. The complete pixel pipeline beats text-only RAG by 3.3-3.5 points at the
edge of significance (p=0.043 nonhard, 0.057 usable) — but that comparison is post
hoc, both prespecified comparisons are null, and no comparison survives correction
for the six tests reported. Pixels help some individual cases and can hurt under
image conflict.

The frozen protocol was run twice. E12 used 20 category-balanced articles and
yielded 12 paired cases, at which n every interval crossed zero and the sign of
M_vision - M depended on the usability cut. E12b re-ran the identical frozen
configuration on all 150 final-test articles. Nothing was re-selected between them;
both read `configs/final_research.json`. All figures below are E12b unless stated.

## Experimental validity

- Corpus: 1,023 BBC News articles from January 2024, each with text and an image.
- Duplicate audit: 102 candidate duplicate/near-duplicate edges; 20 crossed the
  inherited random split.
- Group-safe roles: 707 retrieval pool, 166 development, 150 final test, with zero
  audited duplicate edges crossing roles.
- Frozen final sample: all 150 final-test articles (E12b). E12's earlier 20-article
  cut took four from each of five declared section families, which over-weights
  `sports` (4 of 4 in the corpus) and under-weights `society_culture` (4 of 74), so
  the two runs estimate different quantities: a balanced-sample effect versus the
  average effect over the declared role.
- No training or fine-tuning: SBERT and CLIP are frozen encoders; generation and
  judging are hosted inference.
- M and M_vision use identical retrieval, passages, headlines, captions, ordering,
  prompt, model, and decoding. M_vision adds only the corresponding image pixels.

## Retrieval results

Development selected alpha=0.75 (75% dense text, 25% image): Recall@5 0.940
versus 0.927 for dense text, with three visual rescues and one harm. Equal fusion,
pure image, and RRF were rejected. A strong word/bigram TF-IDF baseline reached
Recall@5 1.00, revealing substantial lexical cues in the inherited questions.

On the frozen 20-query final sample, Recall@5 was 1.00 for dense text, selected
fusion, and lexical retrieval; image-only retrieval was 0.55. Thus final fusion
tied rather than improved on text retrieval.

## Generation and grounding

| frozen final system | usable faithfulness | usable items |
|---|---:|---:|
| B1 text RAG | 0.864 | 77 / 150 |
| M fused retrieval + captions | 0.874 | 80 / 150 |
| M_vision M + pixels | **0.898** | 84 / 150 |

Paired differences, 10,000-resample bootstrap intervals:

| comparison | status | cut | n | difference | 95% CI | Wilcoxon p | win/loss |
|---|---|---|---:|---:|---|---:|---:|
| M − B1 | prespecified | usable | 75 | +0.0188 | [−0.0158,+0.0594] | 0.642 | 28/24 |
| M − B1 | prespecified | nonhard | 101 | +0.0171 | [−0.0127,+0.0511] | 0.706 | 35/31 |
| M_vision − M | prespecified | usable | 80 | +0.0254 | [−0.0028,+0.0558] | 0.102 | 28/19 |
| M_vision − M | prespecified | nonhard | 104 | +0.0053 | [−0.0252,+0.0359] | 0.520 | 34/27 |
| M_vision − B1 | **post hoc** | usable | 75 | +0.0348 | [+0.0017,+0.0708] | 0.057 | 31/17 |
| M_vision − B1 | **post hoc** | nonhard | 98 | +0.0330 | [−0.0004,+0.0673] | 0.043 | 41/22 |

**Both prespecified comparisons are null.** The frozen config declares adjacent paired
comparisons as primary, which means the two rungs of the ladder — B1 to M, and M to
M_vision — are what this experiment committed to testing, and neither reaches
significance. M_vision − B1 skips a rung, was added after the data was seen, and is
reported because it is where the accumulated effect becomes visible, not because it
was predicted.

- Every comparison is positive and the two usability cuts agree in sign.
- The individual channel steps are each null and each carries roughly half the
  total, which is consistent with a small effect accumulating across both.
- 7.1% of supported M_vision claims received both text and pixel support; no
  supported claim required pixels alone.

Three caveats that must travel with these numbers:

1. **Six comparisons, uncorrected.** Bonferroni would require p < 0.008.
   `configs/final_research.json` freezes *adjacent* paired comparisons as primary;
   M_vision − B1 was added after seeing the data and is post hoc.
2. **Bootstrap and Wilcoxon disagree at the margin**, in opposite directions across
   the two cuts. That is what a borderline effect looks like, not a robust one.
3. **Still underpowered.** The detectable-effect floor at 80% power is 0.043-0.054
   against observed effects of 0.005-0.035. Confirming M_vision − M needs 224 pairs;
   the final-test role is exhausted at 150 articles, so more pairs require a larger
   corpus rather than another run.

Category cuts are reported in `per_item_by_category.csv`. They remain descriptive:
the role is unbalanced by construction (74 society_culture, 4 sports).

### What the 20-article sample cost

| comparison | E12 (n=12) | E12b (n=75) |
|---|---:|---:|
| M − B1 | −0.0405 | **+0.0188** |
| M_vision − M | +0.0366 | +0.0254 |

M − B1 flips sign. E12's apparent finding that fusion *hurts* faithfulness was an
artifact of the sample size, and E12's M_vision − M disagreed with itself across
usability cuts (+0.037 usable, −0.015 nonhard). Both symptoms disappear at the full
sample. Enlarging it improved the detectable-effect floor 1.3-2.2x (MDE 0.063-0.094
to 0.043-0.054) for $0.71 of hosted inference.

## Mechanism and stress findings

The targeted 12-case development diagnostic produced M_vision − M = +0.205 on
nine jointly usable cases, CI [+0.057,+0.427], while M − B1 remained null. This
sample deliberately concentrated visual rescues, harms, and strong rank shifts,
so it identifies possible benefit conditions but is not a population estimate.

For the five cases with pixel-supported claims, rotating another case's images
while keeping text fixed reduced clean-evidence faithfulness from 0.983 to 0.927:
mean −0.057, CI [−0.123,0.000]. Two cases degraded and three were unchanged.

## Null, negative, and invalid experiments retained

- Caption addition and caption-mediated fusion did not improve faithfulness.
- RRF and equal text/image weighting underperformed restrained fusion.
- TF-IDF exposed a lexical ceiling rather than a multimodal advantage.
- No model training occurred during any experiment.

## Efficiency

- Embedding build: about 40 seconds with two CPU threads.
- Median final retrieval: 35 ms/query after loading.
- Index plus embedding artifacts: about 38 MB.
- Research-extension API ledger: $0.922 across 1,008 successful calls, under the
  code-enforced $8 ceiling and $10 prepaid balance. E12b accounts for $0.714 of
  that across 794 calls.
- Measured provider limits, 2026-08-09: 10,000 RPM and 200,000 TPM. Tokens bind,
  not requests: a generation call carries ~6,205 tokens, so unpaced serial calls
  reach ~186k tokens/min. Pacing is therefore derived from the token budget
  (`research_scaleup.CALL_GAP`), not from a hand-picked delay.
- Generation latency was not systematically instrumented and is not reconstructed
  after the fact.

## Validation status

Two checks exist, and only the second satisfies the contract.

**E13a, AI-assisted audit on E12 claims.** An independent review completed all 50
blind rows: 39 supported, 11 unsupported, three supported by both text and pixels.
Agreement with the frozen judge was 88%, Cohen's kappa 0.672, modality agreement
86%. The contract states that an AI assistant's labels do not count as human
validation, so this is retained as secondary consistency evidence only.

**E13b, literal human validation on E12b claims.** 50 claims were sampled at random
from E12b's 2,838 and hand-labelled blind — no judge verdict and no system label
visible. Sampled from E12b rather than reusing E12's sheet: E12b is what the
conclusion rests on, and those rows never carried AI labels, so blindness is
structural rather than asserted.

| metric | value |
|---|---:|
| n | 50 |
| raw agreement | 0.80 (40/50) |
| **Cohen's kappa** | **0.057** |
| modality agreement | 0.80 |
| confusion (human rows, judge cols) | tn 1, fp 4, fn 6, tp 39 |

**The judge is not validated.** Raw agreement of 80% and a kappa near zero are not
in conflict: both raters answer "supported" for roughly 90% of claims, so chance
agreement is already high and kappa removes almost all of the observed agreement.
The verdicts that matter most are the negative ones, because the hallucination rate
rests entirely on them — and there the overlap is nearly absent. The human marked 5
claims unsupported, the judge marked 7, and exactly 1 is the same claim.

The disagreements are systematic rather than noisy, and they run in both directions:

- Where the judge was **stricter** (6 cases) it was usually right. The human passed
  three attribution errors — claims saying "the board" where the evidence named a
  professor, "unions" where it was the company, "fires" where it was damaged roads.
- Where the judge was **more lenient** (4 cases) it was usually wrong, and in 3 of
  those its own written reason argued against support while the verdict said
  supported. That is a structured-output consistency failure, not a comprehension
  failure.

**What this does not undermine.** The risk worth checking is arm-dependent judge
error: if the judge were more lenient toward the arm receiving pixels, the
M_vision result would reflect its preference rather than the system's capability.
The 50-item sample hints at it — judge-minus-human supported rate is -0.077 for B1,
-0.106 for M, and +0.056 for M_vision — but at n=13/19/18 per arm that cannot be
established. Measured instead across all 2,838 claims, the rate of supported
verdicts carrying a self-contradicting reason is flat across arms: 3.3% B1, 4.3% M,
3.8% M_vision. Errors of that kind cancel in a within-item paired difference, so the
paired comparison survives while the absolute level does not.

**Limitation of the validation itself.** Only 7 of the 50 sampled claims were
judge-negative, and kappa is unstable at that count — a single item flipping moves
it substantially. A stratified re-run, 25 judge-positive and 25 judge-negative,
would estimate the judge's precision on the negative verdicts directly. Random
sampling was kept here because it gives an unbiased estimate of overall agreement,
which the stratified design would not.

Reproduce with:

```bash
python -m src.final_validation --base E12b_full_final_sample              # export blind sheet
python -m src.final_validation --base E12b_full_final_sample --validate   # score it
```

Judge-derived faithfulness values remain provisional.
