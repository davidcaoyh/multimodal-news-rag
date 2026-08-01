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
The whole project exists to compare **B1 vs M**.

## Quickstart

```bash
bash scripts/setup.sh          # venv + pinned deps + macOS OpenMP fix + verify
source .venv/bin/activate
python src/inspect_data.py     # see what the corpus looks like
```

Needs **Python ≥ 3.10** (macOS system 3.9 will not work — `brew install python@3.12`).
No GPU required. Full instructions and troubleshooting: **[docs/setup_guide.md](docs/setup_guide.md)**.

The corpus ships with the repo — **don't rebuild it**, or your split won't match
your teammates'. Rationale in the setup guide.

## Status

**Day 1 complete.** Environment, corpus, and split are done and verified.

| Day | Deliverable | State |
|---|---|---|
| 1 | Environment + BBC corpus + split | ✅ done |
| 2 | `embed.py` → `index.py` → `retrieve.py` | ⬜ next |
| 3 | `generate.py` — prompt, LLM, abstention | ⬜ |
| 4 | `app/streamlit_app.py` — the demo | ⬜ |
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

Generation is verified working end-to-end against `gpt-4o-mini`.

### Known data caveats (carried into Day 2 / Day 5)

- **26.6% of article bodies contain video-player boilerplate**
  (`"This video can not be played…"`). Strip it in `embed.py` before chunking, or
  it becomes retrievable evidence. 31.7% have some junk overall.
- **`story_type` is 68% `other`** and `sport` has exactly 1 article, because BBC's
  `section` field is geographic (`Middle East`, `Wales`) rather than topical.
  Affects only the stratified analysis on Day 5.
- **recall@k needs a design decision.** `test` rows are excluded from the index,
  so "recall the gold article" is 0 by construction. Decide whether the retrieval
  target is the test article's own passages or related articles.

## Repo layout

```
src/day1_build_dataset.py   DONE  fetch BBC + images -> data.parquet
src/day1_split.py           DONE  story_type labels + index_pool/test split
src/inspect_data.py         DONE  read-only data explorer (start here)
scripts/setup.sh            DONE  one-shot environment setup
scripts/fix_openmp.sh       DONE  macOS libomp conflict fix
src/embed.py                TODO  Day 2
src/index.py                TODO  Day 2
src/retrieve.py             TODO  Day 2
src/generate.py             TODO  Day 3
app/streamlit_app.py        TODO  Day 4
src/evaluate.py             TODO  Day 5
data/processed/             committed — index_pool.parquet, test.parquet
data/images/                committed — 1023 jpgs
results/                    metrics.csv, plots
```

## Docs

- **[docs/setup_guide.md](docs/setup_guide.md)** — environment setup, troubleshooting (teammates start here)
- [docs/plan_7day.md](docs/plan_7day.md) — the active plan
- [docs/day1_guide.md](docs/day1_guide.md) — detailed Day 1 steps
- [docs/plan_full_2week.md](docs/plan_full_2week.md) — ambitious version + future-work list
- [CLAUDE.md](CLAUDE.md) — architecture, conventions, current status

## Notes

- **macOS:** re-run `bash scripts/fix_openmp.sh` after any pip install touching
  torch/faiss/sklearn, or FAISS searches will segfault.
- **API key:** restricted OpenAI key, `Chat completions = Request` only. Not needed
  before Day 3. Embeddings run locally, so no embeddings permission is required.
