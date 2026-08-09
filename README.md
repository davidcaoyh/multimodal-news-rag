# Multimodal RAG for Factual News Summarization

Does grounding news summaries in **text + image** evidence reduce hallucination
compared to **text-only** retrieval? This repo builds a retrieval-augmented
summarizer with a text-vs-multimodal toggle and a small factuality-evaluation
harness.

ECE 1508 course project · 7-day scope · corpus: BBC News

## Headline result — frozen research extension

This is now an end-to-end multimodal RAG system: CLIP pixels affect retrieval and
GPT-4o-mini receives the five retrieved images themselves, not just captions.

| held-out system | usable faithfulness | usable coverage |
|---|---:|---:|
| **B1** text-only RAG | **0.861** | 13/20 |
| **M** image-fused retrieval + captions | 0.834 | 12/20 |
| **M_vision** M + actual pixels | 0.858 | **13/20** |

On the 20-item duplicate-group-safe frozen final sample, **M_vision − M = +0.0366**
over 12 jointly usable items, with paired bootstrap 95% CI **[−0.0155,+0.0888]**.
Actual pixels sometimes help, but the held-out result does not establish a reliable
average gain. Only 4.0% of supported M_vision claims received both text and pixel
support; none required pixels alone. Caption-mediated M likewise did not beat B1.

A targeted 12-case development diagnostic had shown a larger M_vision gain
(+0.205, CI [+0.057,+0.427]), illustrating why the final split was necessary. A
five-case wrong-image stress test reduced clean-evidence faithfulness by 0.057
(CI [−0.123,0.000]): modest, case-dependent harm rather than catastrophic copying.

An independent AI-assisted audit of 50 blind claims agreed with the frozen judge on
**44/50 claims (88%)**, with Cohen's kappa **0.672** and modality agreement **86%**.
This is a useful secondary robustness check, not a substitute for human annotation;
judge-based numbers remain provisional pending literal human review. Full experiment
history: **[docs/research_log.md](docs/research_log.md)**.

## Pipeline
```
query ─► text + image retrieval ─► text/captions + actual pixels ─► GPT-4o-mini
```
SBERT `all-MiniLM-L6-v2` (384-d, text) + CLIP ViT-B/32 (512-d, images) → two FAISS
`IndexFlatIP` indexes. Multimodal retrieval fuses the two score streams:
`score = α·s_text + (1−α)·s_img`, α=0.75, with per-query min-max normalization.

The frozen comparison uses **B1** text RAG · **M** image-fused retrieval plus captions ·
**M_vision** the same M evidence plus actual pixels. M versus M_vision isolates pixel input.

**Nothing is trained.** SBERT and CLIP are frozen pre-trained encoders used
forward-only; GPT-4o-mini is a hosted API. The corpus is a searchable index, not
training data.

## Research extension at a glance

1. Audited text and image duplicates and found 20 candidate leakage edges crossing
   the inherited random split.
2. Rebuilt group-safe research roles: 707 retrieval-pool, 166 development, and 150
   final-test articles, with zero audited duplicate edges crossing roles.
3. Added `M_vision`, which sends the retrieved JPEG pixels to the generator while
   holding M's retrieval, text, captions, prompt, model, and decoding fixed.
4. Selected restrained fusion (`alpha=0.75`) on development data and tested dense,
   image-only, reciprocal-rank, lexical, and reranked alternatives.
5. Ran image-sensitive development diagnostics and a controlled wrong-image stress
   test to identify when pixels help and when misleading pixels hurt.
6. Froze a balanced 20-case held-out comparison before final generation and judging,
   then completed a separate 50-claim AI-assisted evidence audit.

## Quickstart

```bash
bash scripts/setup.sh          # venv + pinned deps + macOS OpenMP fix + verify
source .venv/bin/activate

python src/embed.py && python src/index.py   # build the indexes (~35 s)
streamlit run app/streamlit_app.py           # the demo
python -m src.retrieve                       # 5 eyeball queries, both modes
python -m src.generate                       # 1 query through B0 / B1 / M
python src/inspect_data.py                   # explore the corpus
```

Needs **Python ≥ 3.10** (macOS system 3.9 will not work — `brew install python@3.12`).
No GPU required — a fused query is 21 ms on an M4 MacBook Air. Full instructions and
troubleshooting: **[docs/setup_guide.md](docs/setup_guide.md)**.

The corpus ships with the repo — **don't rebuild it**, or your split won't match your
teammates' and B1-vs-M numbers stop being comparable. Rationale in the setup guide.

One OpenAI API key is used for GPT-4o-mini generation and GPT-5.6 Luna judging (`.env`,
see `.env.example`). No Anthropic key is required. The extension's conservative ledger
records **$0.208** across 214 successful calls, under a code-enforced $8 ceiling.

## Status

**Implementation, frozen automated evaluation, and the AI-assisted evidence audit are
complete.** Literal blind human validation remains outstanding.

| Day | Deliverable | State |
|---|---|---|
| 1 | Environment + BBC corpus + split | ✅ done |
| 2 | `embed.py` → `index.py` → `retrieve.py` | ✅ done |
| 3 | `generate.py` — prompt, LLM, abstention | ✅ done |
| 4 | `app/streamlit_app.py` — the demo | ✅ done |
| 5 | `evaluate.py` — faithfulness + recall@k + ablation | ✅ done |
| 6 | Group-safe actual-image research extension | ✅ done |
| 7 | Frozen evaluation + report-ready results | ✅ done |
| — | Independent 50-claim AI-assisted audit | ✅ done |
| — | 50-claim human validation of the judge | ⬜ outstanding |

The historical inherited run recorded $7.24 under its former provider setup. The new
extension has its own auditable ledger: **$0.208 across 214 calls**. All successful LLM
calls are cached by prompt hash, so identical re-runs cost $0.

### Corpus (Day 1 output)

| | |
|---|---|
| Source | HF `RealTimeData/bbc_news_alltime`, config `2024-01` |
| Clean pairs | **1023** (from 1562 raw articles) |
| `index_pool` / `test` | **873 / 150** |
| Every row has an image | yes — 1023 jpgs, 38 MB |
| Body length | median 613 words |
| Integrity | 0 nulls, 0 duplicate ids, 0 id/headline overlap between pool and test |

### Retrieval (Day 2 output)

11,052 passages · `text_emb` (11052, 384) · `img_emb` (1023, 512), both unit-norm.
Self-retrieval rank 1 at score 1.00000. Rebuilt in ~35 s from committed data;
`data/index/` is gitignored on purpose.

Measured on an M4 MacBook Air, 24 GB — regenerate with `python -m src.bench`:

| Step | Measured |
|---|---|
| SBERT, 11,052 passages | 19.3 s (573/s) |
| CLIP, 1023 images | 10.2 s (101/s) |
| FAISS build, both indexes | 0.004 s |
| **Fused query, end-to-end** | **21 ms** (79% of it query encoding, 6% search) |

### Generation (Day 3 output)

`results/summaries.csv` — 450 rows (150 test items × 3 configs), the interface to the
Day 5 evaluation. 607 s, $0.22.

| config | tau-gated | model refused | usable | median words |
|---|---:|---:|---:|---:|
| B0 | 0 | 0 | 150 | 84 |
| B1 | 2 | 15 | **133** | 88 |
| M | 5 | 14 | **131** | 91 |

- **129/150 (86%)** items have a summary from both arms — the paired sample the
  B1-vs-M result rests on. Refusals are shared (15 both, 2 B1-only, 4 M-only), so they
  shrink the sample without biasing the comparison.
- **53%** of M's top-5 articles never appear in B1's; only 5/150 queries return an
  identical top-5. The independent variable is real.
- 0 test articles leaked into any evidence block.

### Demo (Day 4 output)

`streamlit run app/streamlit_app.py` — screenshots in [docs/screenshots/](docs/screenshots/).

![side-by-side B1 vs M](docs/screenshots/02_side_by_side.png)

One screen, both arms, the same query. Per arm: the summary, two retrieval diagnostics,
the literal prompt that was sent, and the retrieved articles with thumbnails, per-stream
scores and passages. Articles one arm found and the other missed are badged `only B1` /
`only M`. k, α and τ are live sliders (α=1.0 collapses M onto B1). Abstention renders as
**three** distinct states, not two — τ-gated (no LLM call made), model-refused (the call
happened and the model declined), and normal.

Verified headlessly with `streamlit.testing.v1.AppTest`: cold start, a real query
side-by-side, an off-topic query gating both arms, and the single-arm + B0 views.

### Retrieval recall — B1 vs M (free to compute, no API)

Recall of the withheld gold article, `query_qa` with `include_test=True`, n=150:

| config | @1 | @5 | @10 | B1-only@5 | M-only@5 | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| B1 text-only | 0.733 | **0.913** | 0.940 | — | — | — |
| M α=0.9 | 0.733 | 0.933 | 0.953 | **0** | 3 | 0.250 |
| M α=0.75 | 0.727 | **0.940** | 0.960 | **0** | 4 | 0.125 |
| M α=0.5 (the committed run) | 0.673 | 0.860 | 0.900 | 12 | 4 | 0.077 |
| M α=0.25 | 0.573 | 0.740 | 0.787 | 30 | 4 | <0.001 |
| M α=0.0 (pure image) | 0.353 | 0.600 | 0.693 | 50 | 3 | <0.001 |
| M α=1.0 | 0.733 | 0.913 | 0.940 | 0 | 0 | 1.000 |

At α=0.75 multimodal retrieval **strictly dominates** text-only — a superset of B1's
hits, zero losses — but on only 4 discordant pairs (McNemar p=0.125), so the direction is
unambiguous and the magnitude is **not** established. At the current default α=0.5 fusion
measurably *hurts* recall. α=0.5 was an inherited default, never measured; Day 5 sweeps
it. See [docs/decisions.md](docs/decisions.md) D11.

**α=1.0 reproduces B1 exactly on all 150 queries.** That is the only empirical proof that
`mode="text"` and `mode="multimodal", α=1.0` are one code path, so a B1-vs-M difference
cannot be an implementation artefact. Asserted on every `--recall` run.

Recall is retrieval quality, not the research question — summary faithfulness is (above).

### Historical inherited evaluation (before the research extension)

These preserved baseline numbers used 513 summaries and 5,146 claims. They are retained
for provenance, but the current extension uses GPT-5.6 Luna, a group-safe split, actual
image pixels, and the frozen E12 evaluation above.

| comparison | isolates | diff | 95% CI | p |
|---|---|---:|---|---:|
| B1 → M-nocap | image-fused **retrieval** | +0.0046 | [−0.024, +0.032] | 0.640 |
| M-nocap → M | **caption** text in prompt | −0.0118 | [−0.039, +0.014] | 0.530 |
| B1 → M | both combined | −0.0150 | [−0.043, +0.013] | 0.252 |

**Retrieval confidence gates *whether* the model answers, not *how faithfully*.** Across
retrieval-confidence quartiles, faithfulness is flat (0.902 / 0.870 / 0.880 / 0.925) while
refusal collapses (**72.4% → 41.9% → 26.7% → 1.3%**). Better evidence makes the model
willing to answer; conditional on answering it is ~90% faithful regardless.

### Independent evidence-label audit

The existing 50-row blind sheet was labeled in a separate AI-assisted evidence review,
without exposing the frozen judge verdicts during the first pass. The final audit has
39 supported and 11 unsupported claims; three supported claims were independently
grounded by both text and pixels. Against the frozen judge it reached 88% agreement,
kappa 0.672, and 86% modality agreement. Files are retained under the neutral
`outputs/validation_audit/` name, with provenance stated inside the workbook. These
numbers measure agreement between two automated reviews and are not reported as a
human-subject validation result.

## Limitation status after further development

### Resolved

- **[Resolved after further development] The generator previously saw captions but not
  pixels.** `M_vision` now sends the five retrieved JPEGs to GPT-4o-mini, and the evaluator
  can attribute support to text, pixels, both, or neither.
- **[Resolved in the inherited implementation] Video-player boilerplate.** `clean_body()`
  removes it before chunking; no detected boilerplate passages survive into the index.

### Partially addressed

- **[Partially addressed after further development] Split leakage.** The inherited random
  split contained 20 audited cross-role duplicate edges. Group-safe roles now have zero
  detected E04 edges across roles, although lightweight text similarity and image dHash
  cannot guarantee that every semantic duplicate was found.
- **[Partially addressed after further development] Moving evidence yardstick.** B1 and M
  still retrieve different evidence and are judged against their own contexts. The
  controlled M-to-M_vision comparison fixes the text evidence and changes only pixels,
  so the pixel ablation is cleaner, but the broader B1-to-M comparison retains this limit.
- **[Partially addressed after further development] Weak story categories.** The frozen
  sample balances five declared section families, but only one to four paired usable cases
  remain per family; category findings are descriptive rather than inferential.

### Still open

- **High refusal and unusable-summary rates.** The smaller group-safe pool and withheld
  source article leave only 12–13 usable summaries out of 20 final cases. Abstention may be
  correct, but it sharply reduces the paired evaluation sample.
- **No literal human validation.** The 50-claim AI-assisted audit checks consistency but
  does not satisfy a human-annotation claim. Faithfulness remains judge-derived.
- **Text-retrieval ceiling.** Dense text, selected fusion, and lexical retrieval all reach
  Recall@5=1.00 on the frozen 20 queries; TF-IDF also reached 1.00 on development. This
  leaves little retrieval headroom for images and suggests strong lexical cues.
- **Demo diagnostics are not causal evaluation.** On-screen retrieval scores cannot answer
  whether B1 or M is more faithful; that still requires external gold recall or claim-level
  evaluation.

### New limitations found after further development

- **Small frozen final sample.** Only 20 final queries and 12 jointly usable M/M_vision
  cases produce wide confidence intervals; the +0.0366 pixel gain is not conclusive.
- **Rare measurable pixel contribution.** Only 4.0% of supported M_vision claims were
  supported by both modalities and none required pixels alone, limiting the mechanism's
  practical effect in this corpus.
- **Development enrichment can exaggerate gains.** The targeted visual diagnostic found
  +0.205, while the frozen held-out estimate was +0.0366. Selection by visual sensitivity
  is useful for mechanism discovery but not for population claims.
- **Wrong-image robustness is based on five cases.** It reveals possible case-dependent
  harm but cannot estimate a general failure rate.
- **Single outlet and month.** All 1,023 items come from BBC News in January 2024, limiting
  temporal, publisher, cultural, and visual-domain generalization.
- **Same-provider automated generation and judging.** GPT-4o-mini generates and GPT-5.6
  Luna judges. They are different models, but shared provider/model-family biases may make
  their errors more correlated than a cross-provider evaluation.
- **Hosted inference dependence.** Sustained local vision-language inference was not
  evaluated on the available laptop because of thermal constraints; generation and judging
  therefore require a hosted API.
- **Generation latency was not systematically measured.** Retrieval and cost are logged,
  but end-to-end generation latency cannot be reconstructed reliably after the run.

## Repo layout

```
src/day1_build_dataset.py   DONE  fetch BBC + images -> data.parquet
src/day1_split.py           DONE  story_type labels + index_pool/test split
src/inspect_data.py         DONE  read-only data explorer (start here)
src/embed.py                DONE  clean_body + chunk + SBERT/CLIP encoders
src/index.py                DONE  build/save/load/search FAISS + smoke checks
src/retrieve.py             DONE  text-only + fused retrieval  (run with -m)
src/queries.py              DONE  test headlines -> topic + question queries
src/generate.py             DONE  prompt build + LLM + abstention + batch runner
src/bench.py                DONE  re-measure timings -> results/timings.csv
scripts/setup.sh            DONE  one-shot environment setup
scripts/fix_openmp.sh       DONE  macOS libomp conflict fix
app/streamlit_app.py        DONE  the demo — side-by-side B1 vs M + abstention UI
src/evaluate.py             DONE  faithfulness judge + recall@k + ablation (-m)
src/audit_leakage.py        DONE  exact/near text-image duplicate audit
src/build_research_split.py DONE  group-safe pool/development/final roles
src/research_generation.py  DONE  frozen B1/M/M_vision generation runner
src/research_evaluation.py  DONE  image-aware claim support + modality judge
src/research_final.py       DONE  frozen 20-case final comparison
src/research_stress.py      DONE  controlled wrong-image stress test
src/final_validation.py     DONE  blind 50-claim agreement scoring
data/processed/             committed — data/index_pool/test/queries .parquet
data/research/              committed — group-safe split manifest
data/images/                committed — 1023 jpgs
docs/screenshots/           committed — demo captures for the report
results/summaries.csv       450 rows — B0/B1/M generation, the Day 3 -> Day 5 interface
results/summaries_ablation.csv  150 rows — the M-nocap caption ablation
results/claims.csv          5,146 per-claim judge verdicts with rationales
results/metrics.csv         the headline table + ablation
results/recall.csv          the alpha sweep
results/experiments/E04-E12/    research-extension metrics and frozen outputs
results/experiments/E12_final_comparison/human_validation_50.csv
                              50-row AI-assisted audit in the inherited label schema
outputs/validation_audit/     reviewed workbook with explicit audit provenance
```

## Docs

- **[docs/setup_guide.md](docs/setup_guide.md)** — environment setup, troubleshooting (teammates start here)
- **[docs/architecture.md](docs/architecture.md)** — how the system fits together, invariants, data contracts
- **[docs/evaluation_report.md](docs/evaluation_report.md)** — Day 5 findings: what was built, measured, and what it means
- **[docs/future_work.md](docs/future_work.md)** — ranked improvement areas with cost/effort estimates
- [docs/plan_7day.md](docs/plan_7day.md) — the active plan
- [docs/decisions.md](docs/decisions.md) — design decision log (D4–D16) and rejected alternatives
- [docs/day1_guide.md](docs/day1_guide.md) · [docs/day2_guide.md](docs/day2_guide.md) — detailed per-day steps
- [docs/plan_full_2week.md](docs/plan_full_2week.md) — ambitious version + future-work list
- [CLAUDE.md](CLAUDE.md) — architecture, conventions, current status

## Notes

- **macOS:** re-run `bash scripts/fix_openmp.sh` after any pip install touching
  torch/faiss/sklearn, or FAISS searches will segfault — on the first search, not at
  import, so it surfaces looking like a retrieval bug.
- **CLIP must load as `ViT-B-32-quickgelu`**, never `ViT-B-32`. The wrong name runs and
  passes every sanity check while quietly degrading the embeddings; top-5 image
  retrieval overlap between the two is only 3.2/5. Pinned in `embed.py`.
- **`torch.no_grad()` in `embed.py` must stay scoped to the forward passes.** Torch's
  grad mode is thread-local; a single global call inside the cached model loader works
  from the CLI and breaks under Streamlit, which reruns in a new thread each time — the
  first query succeeds and the second raises. Forward values are unaffected either way.
- **API key:** OpenAI project key with Chat Completions access. Embeddings run locally,
  so no embeddings permission is needed. The judge is GPT-5.6 Luna; no Anthropic key is
  required. Keep auto-reload off and use the persistent software budget ledger.
