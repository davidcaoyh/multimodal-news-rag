# Multimodal RAG for Factual News Summarization

Does grounding news summaries in **text + image** evidence reduce hallucination
compared to **text-only** retrieval? This repo builds a retrieval-augmented
summarizer with a text-vs-multimodal toggle and a small factuality-evaluation
harness.

ECE 1508 course project · 7-day scope · corpus: BBC News

## Pipeline
```
query ─► retrieve evidence ─► grounded prompt ─► GPT-4o-mini ─► summary
              │
    text-only  vs  text + image (late score fusion)
```
SBERT `all-MiniLM-L6-v2` (384-d, text) + CLIP ViT-B/32 (512-d, images) → two FAISS
`IndexFlatIP` indexes. Multimodal retrieval fuses the two score streams:
`score = α·s_text + (1−α)·s_img`, α=0.5, with per-query min-max normalization.

Three configs: **B0** no retrieval · **B1** text-only RAG · **M** multimodal RAG.
The whole project exists to compare **B1 vs M**. The generator, the prompt, and the
corpus are held constant across all three — retrieval is the only variable.

**Nothing is trained.** SBERT and CLIP are frozen pre-trained encoders used
forward-only; GPT-4o-mini is a hosted API. The corpus is a searchable index, not
training data.

## Quickstart

```bash
bash scripts/setup.sh          # venv + pinned deps + macOS OpenMP fix + verify
source .venv/bin/activate

python src/embed.py && python src/index.py   # build the indexes (~35 s)
python -m src.retrieve                       # 5 eyeball queries, both modes
python -m src.generate                       # 1 query through B0 / B1 / M
python src/inspect_data.py                   # explore the corpus
```

Needs **Python ≥ 3.10** (macOS system 3.9 will not work — `brew install python@3.12`).
No GPU required — a fused query is 21 ms on an M4 MacBook Air. Full instructions and
troubleshooting: **[docs/setup_guide.md](docs/setup_guide.md)**.

The corpus ships with the repo — **don't rebuild it**, or your split won't match your
teammates' and B1-vs-M numbers stop being comparable. Rationale in the setup guide.

An OpenAI API key is needed from Day 3 on (`.env`, see `.env.example`). Whole-project
usage so far is **$0.22**.

## Status

**Days 1–3 complete.** Corpus, retrieval, and generation all done and verified.
The headline faithfulness result lands on Day 5.

| Day | Deliverable | State |
|---|---|---|
| 1 | Environment + BBC corpus + split | ✅ done |
| 2 | `embed.py` → `index.py` → `retrieve.py` | ✅ done |
| 3 | `generate.py` — prompt, LLM, abstention | ✅ done |
| 4 | `app/streamlit_app.py` — the demo | ⬜ next |
| 5 | `evaluate.py` — faithfulness + recall@k | ⬜ |
| 6 | HF Spaces deploy + README result | ⬜ |
| 7 | Report + polish | ⬜ |

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

## Known limitations

- **~12% of items return `INSUFFICIENT_EVIDENCE`.** The index pool is 873 articles from
  a single month, so some test topics genuinely have no related coverage, and the source
  article is withheld by design. Refusing is correct behaviour, not a defect. Measured
  scaling curve: median best `s_text` rises **+0.034 per doubling** of pool size (0.419
  at n=100 → 0.526 at n=873), so a larger corpus would shrink this — see
  [docs/decisions.md](docs/decisions.md) D10.
- **`story_type` is 68% `other`** and `sport` has exactly 1 article, because BBC's
  `section` field is geographic (`Middle East`, `Wales`) rather than topical. Affects
  only the stratified analysis on Day 5, which uses hand-labels instead.
- **26.6% of raw bodies contain video-player boilerplate.** Stripped by `clean_body()`
  in `embed.py` before chunking; 0 boilerplate passages survive into the index.
- The generator sees **image captions, not pixels**. CLIP embeddings drive *retrieval*;
  the caption is what carries visual evidence into the prompt.

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
app/streamlit_app.py        TODO  Day 4 — the demo
src/evaluate.py             TODO  Day 5 — faithfulness + recall@k
data/processed/             committed — data/index_pool/test/queries .parquet
data/images/                committed — 1023 jpgs
results/                    summaries.csv, timings.csv, metrics.csv, plots
```

## Docs

- **[docs/setup_guide.md](docs/setup_guide.md)** — environment setup, troubleshooting (teammates start here)
- [docs/plan_7day.md](docs/plan_7day.md) — the active plan
- [docs/decisions.md](docs/decisions.md) — design decision log (D4–D10) and rejected alternatives
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
- **API key:** restricted OpenAI key, `Chat completions = Request` only. Embeddings run
  locally, so no embeddings permission is needed. Day 5 additionally needs an
  `ANTHROPIC_API_KEY` — the faithfulness judge is `claude-sonnet-5`, deliberately a
  different model family from the generator so summaries are not self-graded.
