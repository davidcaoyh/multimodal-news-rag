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

All judge-based numbers remain provisional until a human fills the exported 50-claim
blind validation sheet. Full experiment history: **[docs/research_log.md](docs/research_log.md)**.

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

**Implementation and frozen automated evaluation are complete.** The only required
manual task is blind human validation of 50 judge labels.

| Day | Deliverable | State |
|---|---|---|
| 1 | Environment + BBC corpus + split | ✅ done |
| 2 | `embed.py` → `index.py` → `retrieve.py` | ✅ done |
| 3 | `generate.py` — prompt, LLM, abstention | ✅ done |
| 4 | `app/streamlit_app.py` — the demo | ✅ done |
| 5 | `evaluate.py` — faithfulness + recall@k + ablation | ✅ done |
| 6 | Group-safe actual-image research extension | ✅ done |
| 7 | Frozen evaluation + report-ready results | ✅ done |
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

## Known limitations

- **~35% of items are refusals**, in three distinct modes: the τ gate, the literal
  `INSUFFICIENT_EVIDENCE` token, and — found on Day 5 — **soft refusals** in prose
  (*"The evidence does not provide information about X"*). Earlier figures of ~12% counted
  only the token and undercounted by ~3× (D15). The pool is 873 articles from one month, so
  some test topics genuinely have no coverage and the source article is withheld by design;
  refusing is correct behaviour, not a defect.
- **The judge has not yet been validated against human labels.** `results/validation_sample.csv`
  is exported and blind. Until it is scored, every faithfulness number is provisional.
- **Faithfulness grades each arm against its own evidence.** It measures *"did the model
  stick to what it was given"*, not *"was what it was given any good"* — the yardstick moves
  with the arm. This is the deepest reason B1 and M land within 1.5 points of each other
  despite reading half-different articles. See [docs/future_work.md](docs/future_work.md) §1.3.
- **Text retrieval is near-saturated** (B1 recall@5 = 0.913), leaving ≤8.7 points of
  headroom, so the null may partly be a ceiling effect — see
  [docs/future_work.md](docs/future_work.md) §1.2.
- **`story_type` is 68% `other`** and `sport` has exactly 1 article, because BBC's
  `section` field is geographic (`Middle East`, `Wales`) rather than topical. The
  stratified analysis was dropped rather than run on it.
- **26.6% of raw bodies contain video-player boilerplate.** Stripped by `clean_body()`
  in `embed.py` before chunking; 0 boilerplate passages survive into the index.
- The inherited M arm sees image captions, not pixels. The new M_vision arm closes this
  gap by sending actual retrieved images; its frozen held-out advantage remains uncertain.
- **No retrieval statistic in the demo can answer B1 vs M.** Any metric computed on the
  text stream alone is maximised by B1 *by definition*, because B1 is the argmax of that
  stream — M's mean `s_text` is lower on 145/150 queries and higher on **0**. The two
  on-screen numbers are labelled as diagnostics for that reason. The comparison needs
  ground truth outside both streams: recall of the withheld gold article, or the
  faithfulness judge. See [docs/decisions.md](docs/decisions.md) D12.

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
data/processed/             committed — data/index_pool/test/queries .parquet
data/images/                committed — 1023 jpgs
docs/screenshots/           committed — demo captures for the report
results/summaries.csv       450 rows — B0/B1/M generation, the Day 3 -> Day 5 interface
results/summaries_ablation.csv  150 rows — the M-nocap caption ablation
results/claims.csv          5,146 per-claim judge verdicts with rationales
results/metrics.csv         the headline table + ablation
results/recall.csv          the alpha sweep
results/validation_sample.csv   50 blind claims awaiting hand labels
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
