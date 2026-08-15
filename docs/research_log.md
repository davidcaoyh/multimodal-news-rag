# End-to-end multimodal research log

This is the authoritative ledger for work on
`research/end-to-end-multimodal`. Every meaningful trial is recorded, including
null, negative, rejected, and invalid experiments. The presentation and report
may show only the experiments needed for the final argument; this file preserves
the complete research process.

## Rules

1. Never overwrite the committed baseline artifacts in `results/`.
2. Give every run an experiment ID and a distinct output directory.
3. Record the exact commit, data split, query set, prompt version, seed, model,
   retrieval settings, cost, and runtime.
4. Hyperparameter and architecture selection use development/validation data.
5. The final held-out test configuration is frozen before test execution.
6. A null or negative result is retained when the experiment is valid.
7. An invalid result is retained with the invalidating reason and is never mixed
   into a reported comparison.
8. API/model availability changes are recorded rather than silently substituted.

## Status vocabulary

| status | meaning |
|---|---|
| PLANNED | question and protocol defined, not run |
| RUNNING | execution in progress |
| COMPLETE | valid result available |
| NULL | valid experiment with no meaningful effect |
| REJECTED | valid result but component omitted from final system |
| INVALID | protocol or execution flaw prevents interpretation |
| BLOCKED | cannot proceed without an external dependency or decision |

## Experiment index

| ID | experiment | status | decision | main artifact |
|---|---|---|---|---|
| E00 | inherited B0/B1/M baseline | COMPLETE | preserve | `results/metrics.csv` |
| E01 | inherited retrieval alpha sweep | COMPLETE | exploratory only | `results/recall.csv` |
| E02 | inherited M_nocap caption ablation | NULL | preserve | `results/summaries_ablation.csv` |
| E03 | blind human validation of baseline judge | PLANNED | pending | TBD |
| E04 | text/image duplicate and split-leakage audit | COMPLETE | group duplicate components | `results/experiments/E04_leakage_audit/` |
| E05 | development/validation/final-test contract | COMPLETE | use grouped research roles | `data/research/split_manifest.parquet` |
| E06 | M_vision actual-image generation arm | COMPLETE | actual pixels operational; retain | `results/experiments/E09_development_generation/` |
| E07 | group-safe dense/image fusion selection | COMPLETE | select alpha=0.75 on development | `results/experiments/E07_research_retrieval/` |
| E08 | strong text alternatives and reranking | COMPLETE | retain lexical ceiling; do not overclaim visual retrieval | `results/experiments/E08_research_reranking/` |
| E09 | diagnostic four-arm development generation | COMPLETE | preserve targeted design | `results/experiments/E09_development_generation/` |
| E10 | image-aware claim support and modality attribution | COMPLETE | actual pixels promising on diagnostic sample | `results/experiments/E10_development_claim_evaluation/` |
| E11 | wrong-image stress test | COMPLETE | report modest case-dependent harm | `results/experiments/E11_wrong_image_stress/` |
| E12 | frozen final held-out comparison | COMPLETE | superseded by E12b; n=12 could not resolve any outcome | `results/experiments/E12_final_comparison/` |
| E12b | same frozen protocol, full 150-article final-test role | COMPLETE | direction consistently positive; M_vision-B1 borderline | `results/experiments/E12b_full_final_sample/` |
| E13a | independent 50-claim AI-assisted audit | COMPLETE | retain as secondary agreement evidence | `results/experiments/E12_final_comparison/evaluation/human_validation_metrics.json` |
| E13b | literal human judge validation | PLANNED | requires a freshly blinded human-review copy | `docs/future_work.md` §0.1 |
| E14 | B0 rebuilt on E12b's split/judge, paired | COMPLETE | keep; largest effect in the study (D17) | `results/experiments/E14_b0_openai/` |
| E15 | cross-model judge robustness check | PLANNED | second judge for self-preference-bias robustness | `docs/future_work.md` §0.2 |

## Inherited experiments

### E00 — B0/B1/M baseline

- **Source commit:** `57875634e9c36050ef195d123f81dc391ca7dff8`
- **Question:** Does retrieval reduce hallucination, and does the baseline
  caption-mediated multimodal arm outperform text RAG?
- **Data:** 873 retrieval-pool and 150 test BBC articles.
- **Configuration:** `k=5`, M `alpha=0.5`, `tau=0.35`.
- **Outcome:** B0 faithfulness 0.243, B1 0.901, M 0.886. Usable paired
  `M-B1=-0.015`, 95% CI `[-0.043,+0.013]`, `p=0.252`.
- **Decision:** Preserve as the historical baseline. Do not use it as the final
  end-to-end vision result.
- **Insight:** Retrieval is highly beneficial; caption-mediated multimodality is
  not measurably beneficial in this regime.
- **Limitation:** Judge is unvalidated and the generator never sees pixels.

### E01 — retrieval alpha sweep

- **Question:** How does text/image weighting affect gold-article recall?
- **Outcome:** Recall@5 was B1 0.913, alpha 0.5 0.860, alpha 0.75 0.940,
  alpha 0.9 0.933, and pure image 0.600.
- **Decision:** Preserve as exploratory evidence. Do not select final alpha from
  these test results.
- **Insight:** Image similarity is complementary only at low weight for ordinary
  text-derived queries; excessive image weight damages retrieval.

### E02 — M_nocap caption ablation

- **Question:** Does the changed retrieval set or added caption text explain M?
- **Outcome:** Both isolated differences were statistically null.
- **Decision:** Keep M_nocap as a causal bridge for later M_vision comparisons.
- **Insight:** Caption text did not add measurable grounding value once the
  retrieved articles were fixed.

### E04 — text/image duplicate and split-leakage audit

- **Status:** COMPLETE
- **Commit:** research branch after `0cae0cd`
- **Question:** Do exact or near-duplicate text/images cross the inherited random
  pool/test split?
- **Protocol:** character 3–5-gram TF-IDF cosine at `>=0.85`, exact image SHA-256,
  and 64-bit image dHash distance `<=5`. Candidate pairs are evidence for manual
  inspection, not automatic declarations that two stories are identical.
- **Outcome:** 102 candidate edges: 90 byte-identical image pairs, 9 near-image
  pairs, and 3 near-text pairs. Twenty edges crossed the inherited split: 18
  exact images, 1 near image, and 1 near-text pair, involving 29 articles.
- **Interpretation:** The inherited split is not group-safe. Many repeated images
  belong to follow-up coverage of the same event, which can inflate image recall
  and weaken claims of held-out generalization.
- **Decision:** KEEP. Build research roles by connected duplicate components and
  retain the inherited comparison only as a historical baseline.
- **Artifacts:** `results/experiments/E04_leakage_audit/candidate_pairs.csv` and
  `summary.json`.
- **Limitations:** dHash is a deliberately lightweight visual screen. Cropped or
  semantically equivalent but visually different images may remain undetected;
  all threshold candidates still require interpretation.

### E05 — group-safe research roles

- **Status:** COMPLETE
- **Question:** How can exploratory model selection and final evaluation be
  separated without discarding the inherited work?
- **Policy:** The inherited 150 test articles become development data because
  alpha and prompts were already explored on them. All connected duplicate
  components touching them follow into development. A new seeded 150-article
  final test is selected by whole groups; remaining articles form the pool.
- **Outcome:** 707 pool, 166 development, and 150 final-test articles. Sixteen
  formerly pooled articles moved to development with duplicate-linked inherited
  test articles. There are 44 multi-article duplicate components, largest size 6,
  and zero E04 candidate edges crossing the new research roles.
- **Decision:** KEEP. Use `data/research/split_manifest.parquet` for all new
  experiments. Do not modify the baseline processed parquets.
- **Artifacts:** `data/research/split_manifest.parquet` and
  `results/experiments/E05_research_split/summary.json`.
- **Limitations:** The 707-item pool is smaller than the inherited pool and may
  increase refusal. That is preferable to reporting leakage-contaminated gains.

### E06 — M_vision actual-image generation arm

- **Status:** COMPLETE; implementation and hosted inference validated.
- **Research question:** Do actual retrieved image pixels add grounding value
  beyond the captions and text evidence already supplied to M?
- **Controlled comparison:** M and M_vision use identical fused retrieval,
  captions, prompt instructions, generator, decoding parameters, and evidence
  ordering. M_vision adds one low-detail image block per retrieved article.
- **Implementation:** `src.generate.build_request_content()` converts retrieved
  JPEGs to OpenAI chat-completions image blocks and preserves article/image
  correspondence. Cache metadata stores image hashes rather than base64 payloads.
- **Verification:** Unit tests confirm byte-identical text prompts/evidence for M
  and M_vision, real JPEG data URLs, low-detail mode, image-path persistence, and
  structured generator input. All 18 current tests pass.
- **Outcome:** One cached GPT-4o-mini smoke request succeeded for M_vision and
  cost an estimated $0.000447. This validates transport and budget accounting,
  but is not a scientific sample. The first attempted smoke request
  was rejected before inference with HTTP 401 `missing_scope: model.request`;
  it consumed no tokens, created no cache entry, and added no budget-ledger row.
  A retry after editing permissions returned the identical pre-inference 401 and
  likewise cost $0. Replacing the project key resolved the scope issue.
- **Decision:** KEEP the arm and run controlled development comparisons within
  the $8 software budget. Do not claim a benefit until paired evaluation exists.

### E07 — group-safe dense/image fusion selection

- **Status:** COMPLETE.
- **Protocol:** Evaluate 150 inherited queries on development only, with the gold
  development article reachable among 873 pool+development candidates. Compare
  pure image, five score-fusion weights, text-only dense retrieval, and RRF.
- **Outcome:** alpha=0.75 achieved Recall@5 0.940 versus dense-text 0.927,
  alpha=0.5 0.880, RRF 0.787, and pure image 0.620. The selected fusion rescued
  three queries and harmed one at rank 5 relative to dense text.
- **Decision:** Use alpha=0.75 for subsequent dense multimodal experiments. Reject
  equal weighting and RRF for this corpus/query regime.
- **Insight:** Image similarity is complementary at low weight, but it is not a
  robust standalone retriever for ordinary text queries.

### E08 — strong text alternatives and reranking

- **Status:** COMPLETE.
- **Protocol:** Add an article-level word/bigram TF-IDF baseline over headline and
  body (captions excluded), plus dense+lexical fusion and top-30 lexical reranking.
  Tune and compare only on the same 150 development queries.
- **Outcome:** lexical retrieval reached Recall@5 1.000 and MRR 0.960; dense+
  lexical reached 0.993/0.949; text reranking reached 0.973/0.907; multimodal
  reranking reached 0.960/0.894; alpha=0.75 multimodal dense remained 0.940/0.830.
- **Decision:** Preserve lexical retrieval as the strongest text ceiling and use
  it to qualify the retrieval claim. Do not replace the dense multimodal path in
  the causal image-generation experiments, because doing so would remove the
  visual retrieval intervention being studied.
- **Insight:** These inherited article-derived questions carry unusually strong
  lexical cues. They can support retrieval engineering comparisons, but they are
  biased against demonstrating a visual retrieval advantage at Recall@5.

### E09/E10 — diagnostic actual-image generation and claim attribution

- **Status:** COMPLETE on development.
- **Protocol:** Twelve predeclared diagnostic cases (all rank-5 visual rescues and
  harms, strong up/down-ranks, and neutral controls), four arms: B1, M_nocap, M,
  and M_vision. M and M_vision have byte-identical retrieval and textual evidence;
  only M_vision receives the five corresponding low-detail pixels.
- **Evaluation:** Blind decomposition without evidence, physically separate
  text-only and image-enabled verification, atomic claim support, macro per-item
  faithfulness, refusal cuts, and 10,000 paired bootstrap resamples.
- **Outcome:** On nine jointly usable cases, B1=0.777, M=0.747, and M_vision=0.961.
  M_vision-M was +0.205, 95% bootstrap CI [+0.057,+0.427]. M-B1 was -0.021 with
  a CI crossing zero. Of supported claims in usable M_vision summaries, 8.6%
  received both text and pixel support; none required pixels alone.
- **Interpretation:** Pixels changed generation in a beneficial direction on this
  deliberately image-sensitive diagnostic set, but did not supply uniquely
  necessary facts. This is a mechanism/failure-analysis result, not a population
  estimate, and requires confirmation on the frozen final sample.
- **Coverage:** Each arm had one shared hard abstention. Soft refusals were B1=2,
  M_nocap=2, M=1, M_vision=2 and are reported separately.

### E11 — wrong-image stress test

- **Status:** COMPLETE on development.
- **Protocol:** For the five cases with pixel-supported claims, rotate another
  case's images into M_vision while holding query, retrieval, text, captions,
  prompt, and model fixed. Judge corrupted summaries against the original clean
  text-and-image bundle.
- **Outcome:** Mean clean faithfulness 0.983 versus wrong-image 0.927; paired
  difference -0.057, bootstrap CI [-0.123,0.000]. Two of five cases degraded and
  three were unchanged. Wrong-image unsupported-claim rate was 7.8%.
- **Interpretation:** Image conflict causes modest, case-dependent harm rather
  than catastrophic visual copying in this small stress set.

### E12 — frozen held-out comparison

- **Status:** COMPLETE; protocol was frozen before final outcomes.
- **Protocol:** Twenty seeded final-test articles, balanced four each across five
  declared section families. Headline-only natural queries; pool-only evidence;
  B1, M, and M_vision; alpha=0.75, k=5, tau=0.35; no further tuning.
- **Primary metric:** Macro claim faithfulness on jointly usable summaries with
  paired bootstrap intervals. Coverage, visual attribution, and category cuts are
  secondary.
- **Outcome:** On jointly usable cases, B1=0.861, M=0.834, M_vision=0.858.
  M_vision-M=+0.0366 over n=12, 95% bootstrap CI [-0.0155,+0.0888].
  M-B1=-0.0405, CI [-0.0892,+0.0076]. Usable coverage was 13/20,
  12/20, and 13/20 respectively. Pixel-supported claims were 4.0% of
  supported claims in usable M_vision summaries; none required pixels alone.
- **Retrieval:** Recall@5 was 1.00 for dense text, selected alpha=0.75 fusion,
  and lexical retrieval; image-only was 0.55. The selected fusion tied rather
  than improved on text in the small final sample.
- **Interpretation:** The large targeted development gain did not generalize.
  Actual pixels are sometimes useful, especially in individual visual cases,
  but no reliable average faithfulness improvement is established.
- **Efficiency:** median retrieval 35 ms/query after load; extension API ledger
  $0.208; embedding build about 40 seconds; index plus embedding files ~38 MB.
- **Artifacts:** `configs/final_research.json` and
  `results/experiments/E12_final_comparison/`.

### E13a/E13b — independent audit and pending human validation

- **AI-assisted audit:** The 50 blind rows were independently reviewed without
  exposing frozen judge verdicts during the first pass. The audit labeled 39
  claims supported and 11 unsupported, with three `both` modality labels.
- **Agreement:** 44/50 (88%), Cohen's kappa 0.672, and modality agreement 86%.
- **Artifacts:** `results/experiments/E12_final_comparison/human_validation_50.csv`,
  `evaluation/human_validation_metrics.json`, and
  `outputs/validation_audit/evidence_validation_50.xlsx`.
- **Rule:** This is secondary automated agreement evidence, not literal human
  validation. Before human review, make a fresh copy with `human_supported`,
  `human_modality`, and `notes` cleared so the reviewer cannot see the AI-assisted
  labels. Judge-derived faithfulness remains provisional until then.

### E12b — frozen protocol on the full final-test role

- **Status:** COMPLETE.
- **Research question:** Do E12's conclusions hold when the frozen configuration is
  executed on every article in the final-test role rather than a 20-article sample?
- **Hypothesis:** E12's intervals were too wide to interpret. At n=12 neither a null
  nor a positive result was distinguishable from noise, and M_vision-M reversed sign
  between usability cuts (+0.037 usable, -0.015 nonhard). Enlarging the sample should
  either sharpen the estimate or reveal the earlier one as an artifact.
- **Commit:** `6c07606` on `research/final-sample-scaleup`, branched from `5330d55`.
- **Dataset and split:** all 150 `final_test` articles from
  `data/research/split_manifest.parquet`. Evidence restricted to the 707-article pool;
  development and final-test ids forbidden and asserted per call.
- **Query artifact:** `results/experiments/E12b_full_final_sample/queries.csv`. E12's
  20 queries are copied verbatim; 130 new ones were generated in four batches with the
  byte-identical instruction E12 used. Regenerating the inherited 20 would have risked
  different wording and silently stopped them from being the same items.
- **Systems/configurations:** B1, M, M_vision. k=5, alpha=0.75, tau=0.35, gpt-4o-mini
  at temperature 0 / seed 42 / max_tokens 300, images at `detail: low`. Every value is
  read from `configs/final_research.json` rather than restated, so drift is impossible.
- **Controlled variables:** nothing was re-selected. This is E12's protocol with more
  items, which is why it carries E12's ID with a suffix rather than a new number.
- **Primary metrics:** macro claim faithfulness on jointly usable paired summaries,
  10,000-resample paired bootstrap intervals, Wilcoxon signed-rank.
- **Runtime/API cost:** $0.714 across 794 calls; ledger total $0.922 / 1,008 calls
  against the $8 ceiling. About 40 minutes wall clock.
- **Output artifacts:** `results/experiments/E12b_full_final_sample/` — `queries.csv`,
  `summaries.csv` (450 rows), `evaluation/claims.csv` (2,838 claims), `metrics.csv`,
  `paired_differences.csv`, `per_item.csv`, `per_item_by_category.csv`,
  `diagnostics.json`.
- **Outcome:**

  | comparison | status | cut | n | difference | 95% CI | p |
  |---|---|---|---:|---:|---|---:|
  | M - B1 | prespecified | usable | 75 | +0.0188 | [-0.0158,+0.0594] | 0.642 |
  | M - B1 | prespecified | nonhard | 101 | +0.0171 | [-0.0127,+0.0511] | 0.706 |
  | M_vision - M | prespecified | usable | 80 | +0.0254 | [-0.0028,+0.0558] | 0.102 |
  | M_vision - M | prespecified | nonhard | 104 | +0.0053 | [-0.0252,+0.0359] | 0.520 |
  | M_vision - B1 | **post hoc** | usable | 75 | +0.0348 | [+0.0017,+0.0708] | 0.057 |
  | M_vision - B1 | **post hoc** | nonhard | 98 | +0.0330 | [-0.0004,+0.0673] | 0.043 |

  Both prespecified comparisons are null. The frozen config commits to the two rungs
  of the ladder, B1->M and M->M_vision, and neither reaches significance.

  Usable faithfulness: B1 0.864 (n=77), M 0.874 (n=80), M_vision 0.898 (n=84).
  Visual contribution rate 7.1%, up from E12's 4.0%.
- **Interpretation:** E12's conclusions do not survive. M-B1 flips from -0.0405 to
  +0.0188, so "fusion hurts faithfulness" was an artifact of n=12, and the
  sign-disagreement between usability cuts disappears. All six comparisons are now
  positive. The only one approaching significance is the full pixel pipeline against
  text-only RAG; the two channel steps are individually null and each carries about
  half the total, consistent with a small effect accumulating across both.
- **Decision:** KEEP as the final result. Retain E12 in the ledger as a valid run whose
  sample was too small, not as a retracted one — it is the evidence for how much a
  20-item frozen sample can mislead.
- **Limitations or invalidating conditions:**
  - Six comparisons reported without multiplicity correction; Bonferroni requires
    p < 0.008. `configs/final_research.json` freezes *adjacent* paired comparisons as
    primary, so M_vision-B1 is post hoc and must be labelled as such.
  - Bootstrap interval and Wilcoxon test disagree at the margin in both cuts.
  - Detectable-effect floor is 0.043-0.054 against observed effects of 0.005-0.035.
    Confirming M_vision-M needs 224 pairs and the role is exhausted at 150 articles,
    so further power requires a larger corpus rather than another run.
  - The role is unbalanced (74 society_culture, 4 sports), so E12b and E12 estimate
    different quantities: the average over the declared role versus over a
    category-balanced sample. Category cuts stay descriptive.
  - Judge remains unvalidated by a human (E13b), so every faithfulness figure here is
    provisional in the same way E12's was.

### E12b-a — cross-machine reproduction and generator determinism

- **Status:** COMPLETE. Free; no API calls.
- **Research question:** Does a different machine reproduce the committed E12 retrieval,
  and does temperature-0 generation reproduce its summaries?
- **Protocol:** rebuild the index on Windows/conda, recompute retrieval for all 60
  committed E12 rows, and diff article ids, evidence text, per-stream scores and image
  paths. Then regenerate the 20 overlapping items in E12b and diff the summaries.
- **Outcome, retrieval:** initially 46/60 article sets matched, with image scores
  drifting up to 0.0088 while text scores were exact. Cause was **Pillow 10.4.0 against
  the pinned 12.3.0** — CLIP's preprocessing does a PIL bicubic resize, and the
  resampling implementation changed across those majors. `open_clip` versions already
  matched, and torch was ruled out because SBERT was bit-identical. After aligning
  Pillow: **60/60 on every field, max |delta| = 0.0 on both score streams.**
- **Outcome, generation:** with byte-identical prompts, temperature 0 and seed 42, only
  **17/60** summaries regenerated identically; mean character similarity 0.58, and 29 of
  the 43 differing rows were substantial rewrites. Refusal *classification* was stable at
  **58/60**, and the jointly-usable count on those 20 items was **12 in both runs**.
- **Interpretation:** retrieval is reproducible across machines once Pillow is pinned;
  generation is not reproducible at the level of wording, but is stable at the level that
  determines sample composition. Run-to-run variance is therefore a real noise component
  inside every per-item faithfulness score, and E12's exact figures cannot be reproduced
  by re-running it. This also explains a transient judge failure at case 141, which
  passed on retry without any configuration change.
- **Decision:** KEEP. `embed.py` now pins SBERT to CPU so the encoders are
  device-independent; the Pillow requirement is documented in the setup guide.
- **Limitations:** the determinism estimate rests on 60 paired observations from two
  runs. It bounds the noise but does not decompose it into generator and judge shares.

### E14 — B0 rebuilt on E12b's split and judge, paired (D17)

- **Status:** COMPLETE.
- **Research question:** The report's B0 number (0.243) came from the inherited pipeline —
  `main`'s random pool/test split, scored before the gpt-5.6-luna judge substitution existed.
  Does B0 still look this different from B1/M/M_vision once it is regenerated and judged on
  the *same* split, generator, and judge as the rest of the ladder?
- **Commit:** `src/research_b0.py`, on `main` post-branch-merge.
- **Dataset and split:** the identical 150 `final_test` queries E12b used
  (`E12b_full_final_sample/queries.csv`), so this is a paired addition to that role.
- **Systems/configurations:** B0 only (no retrieval). gpt-4o-mini generator, identical
  decoding to B1/M/M_vision. Judged with gpt-5.6-luna against
  `union(B1 evidence, M evidence)` for the same query (D14).
- **Controlled variables:** generator, judge, decoding, and item set all held identical to
  E12b; only the retrieval condition (none) differs, which is the point.
- **Primary metrics:** macro claim faithfulness, usable cut; paired bootstrap difference
  against E12b's B1.
- **Runtime/API cost:** $0.18 across 450 calls (150 generation + 150 decompose + 150
  verify), on top of E12b's $0.92/1,008-call running total — ledger total after this run:
  $1.10 / 1,458 calls.
- **Output artifacts:** `results/experiments/E14_b0_openai/` — `summaries.csv` (150 rows),
  `evaluation/claims.csv` (1,501 claims), `evaluation/per_item.csv`,
  `evaluation/diagnostics.json`.
- **Outcome:** faithfulness 0.146, 150/150 usable (B0 has no retrieval-confidence gate, so
  unlike B1/M/M_vision it never abstains — its usable count is not comparable to theirs).
  Paired $B_1 - B_0 = +0.586$, 95% CI $[0.525, 0.643]$, $n=77$.
- **Interpretation:** the largest, least ambiguous effect in the whole study, now on a
  pipeline consistent with the rest of the ladder rather than a number spliced in from a
  different one. The fix *lowered* B0's reported faithfulness relative to the inherited
  0.243 — i.e. it went against the direction that would have flattered the headline claim.
- **Decision:** KEEP as the final $B_0$ number; superseding the inherited 0.243 for any
  comparison against E12b's B1/M/M_vision (`docs/baseline_summary.md`'s 0.243 stays as-is,
  since that document is an explicitly dated snapshot of `main`, not a claim about the
  current pipeline).
- **Limitations or invalidating conditions:**
  - B0's 150/150 usable count reflects the absence of an abstention mechanism, not better
    grounding — do not read it as "B0 refuses less" in any positive sense.
  - Same-provider generation and judging: gpt-5.6-luna and gpt-4o-mini are both OpenAI
    models. A different-provider judge check is planned future work (`docs/future_work.md`
    §0.2) rather than completed.

## Experiment entry template

Copy this section for every new experiment.

```markdown
### EXX — title

- Status:
- Research question:
- Hypothesis:
- Commit:
- Dataset and split:
- Query artifact:
- Systems/configurations:
- Controlled variables:
- Primary metrics:
- Secondary metrics:
- Runtime/API cost:
- Output artifacts:
- Outcome:
- Interpretation:
- Decision: KEEP / REJECT / INVESTIGATE
- Limitations or invalidating conditions:
```
