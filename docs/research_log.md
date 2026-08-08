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
| E06a | local VLM feasibility screen | INVALID | aborted for thermal safety; use hosted inference | `results/experiments/E06_local_model_benchmark/` |
| E06b | M_vision actual-image generation arm | RUNNING | API smoke valid; proceed within $8 ledger | `results/api_usage.jsonl` |
| E07 | group-safe dense/image fusion selection | COMPLETE | select alpha=0.75 on development | `results/experiments/E07_research_retrieval/` |
| E08 | strong text alternatives and reranking | COMPLETE | retain lexical ceiling; do not overclaim visual retrieval | `results/experiments/E08_research_reranking/` |
| E09 | query-evidence coverage and claim modality attribution | PLANNED | pending | TBD |
| E10 | visual-first/weak-text benefit conditions | PLANNED | pending | TBD |
| E11 | conflicting and misleading image stress tests | PLANNED | pending | TBD |
| E12 | generation modality ablations | PLANNED | pending | TBD |
| E13 | frozen final held-out comparison | PLANNED | pending | TBD |

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

### E06a — local VLM feasibility screen

- **Status:** INVALID / ABORTED for model comparison; valid operational result.
- **Question:** Can 4B–12B local vision-language models support the experimental
  pipeline on the available M2 Max laptop without paid APIs?
- **Protocol:** Twelve seeded development images, forced-choice associated-caption
  discrimination, structured JSON output. Planned models were Qwen3-VL 4B,
  Qwen3-VL 8B, and Gemma 3 12B.
- **Outcome:** Both Qwen models completed 12 calls but returned empty answer
  content while consuming output tokens, so their apparent 0% accuracy is invalid
  and must not be reported as quality. Gemma returned valid JSON and selected the
  associated caption on its first 6/6 cases, but the laptop became unacceptably
  hot and the experiment was stopped before completion.
- **Decision:** REJECT sustained local VLM inference on this hardware. Preserve
  partial outputs as a feasibility/safety result and move all generator/judge
  inference to hosted APIs. No training or fine-tuning was performed.
- **Limitation:** Gemma's six cases are an incomplete, small, selected prefix and
  provide no defensible comparative accuracy estimate.

### E06b — M_vision actual-image generation arm

- **Status:** RUNNING; implementation complete and hosted inference validated.
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
