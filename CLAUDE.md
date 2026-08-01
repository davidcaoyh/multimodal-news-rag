# CLAUDE.md — Project context for Claude Code

## What this project is
A course project (ECE 1508). One research question: **does multimodal (text+image)
retrieval reduce hallucination in news summarization vs text-only retrieval?**
Deliverables: (1) an interactive Streamlit demo with a text-only vs multimodal
toggle, (2) a small factuality-evaluation harness. **Timeline: 7 days, team project.**
An "okish demo" is the bar, not a perfect research system.

Answers should be concise; use plain mathematical language when relevant.

## The one comparison that matters
Build the pipeline once, run in three configs, score the outputs:
- **B0** LLM alone, no retrieval (hallucination upper bound)
- **B1** text-only RAG (baseline)
- **M**  multimodal RAG (method under test)
Everything serves a fair **B1 vs M** comparison. The generator must be held
**constant** across all three or the comparison means nothing.

## Architecture
```
query ─► RETRIEVE evidence ─► BUILD prompt ─► LLM ─► summary
                │
      switch: [text-only]  vs  [text + image]
```
- Text passages → SBERT `all-MiniLM-L6-v2` (384-d), normalized.
- Images + captions → CLIP ViT-B/32 (512-d).
- Two FAISS `IndexFlatIP` indexes (text, image).
- Multimodal retrieval = **late score fusion**: `score = alpha*s_text + (1-alpha)*s_img`,
  with per-query min-max normalization of each score stream. Default alpha=0.5.
- Generation: GPT-4o-mini behind a single `generate(prompt)` function (temperature=0),
  so a local Llama-3-8B can swap in later. Captions carry visual evidence into the
  text prompt (the model does not see raw pixels in this setup).
- Guardrail: retrieval-confidence abstention — if top score < tau, return
  `INSUFFICIENT_EVIDENCE`.

## Environment — read before running anything

**Python ≥ 3.10 required.** macOS system Python is 3.9 and will NOT work; torch,
faiss-cpu, sentence-transformers, datasets, and streamlit all require ≥3.10.
This project uses Homebrew **Python 3.12.13**. Setup is automated:

```bash
bash scripts/setup.sh          # venv + pinned deps + OpenMP fix + verification
source .venv/bin/activate
```

Install from **`requirements.lock.txt`** (pinned), not `requirements.txt` (unpinned).
Key versions: torch 2.13.0, faiss-cpu 1.14.3, sentence-transformers 5.6.1,
datasets 5.0.1, streamlit 1.60.0, openai 2.52.0, pandas 3.0.5, numpy 2.5.1.

### macOS OpenMP conflict — the biggest trap in this repo
torch, faiss, and sklearn each bundle their own `libomp.dylib`. Two OpenMP runtimes
in one process abort with `OMP: Error #15` or segfault (exit 134/139) **the first
time FAISS runs a search — not at import**. So imports look fine and the crash
appears later inside `index.py`.

Fix: `bash scripts/fix_openmp.sh` (symlinks faiss's + sklearn's copies to torch's).
**Re-run it after ANY pip install touching torch/faiss/sklearn** — pip restores the
bundled copies and the crash returns silently.

Do NOT waste time on these — all tested, none work: importing faiss before torch,
`KMP_DUPLICATE_LIB_OK=TRUE`, `OMP_NUM_THREADS=1`.

### Other environment gotchas
- `load_dotenv()` resolves relative to the **calling file's directory**, not cwd.
  From a script outside the repo it silently finds nothing and leaves the key
  `None`. Pass an explicit path in throwaway scripts.
- `torch.cuda.is_available()` is `False` on Mac — expected. MPS is available, but
  CPU is already fast enough (see benchmarks).
- The `datasets` library caches to `~/.cache/huggingface`, NOT `data/raw/`.
  `data/raw/` is vestigial (`RAW_DIR` in `day1_build_dataset.py:22` is defined but
  never used).

### Measured performance (M4 MacBook Air, 24 GB) — no GPU needed
| Step | Measured |
|---|---|
| SBERT, 20k passages | 4.5 s (4,485/s) |
| CLIP ViT-B/32, 1500 images | 13.3 s (113/s) |
| FAISS build, both indexes | 0.011 s |
| Fused query | 0.50 ms |
| Peak RSS | 3.56 GB |

Compute is a non-issue at this scale. The only slow step is image downloading
(0.34 s each, serial) — which is why the corpus is committed rather than rebuilt.

## LLM / API
- **Model: `gpt-4o-mini`.** Verified working end-to-end.
- **Do not switch to `gpt-5.6-luna`** (it exists, released after mid-2026). It
  rejects `temperature=0` (only default 1), costs more ($0.20/$1.20 per M vs
  $0.15/$0.60), and bills hidden reasoning tokens as output. temperature=0 is
  load-bearing here: without it, a B1-vs-M difference could be sampling noise.
  Also needs `max_completion_tokens` instead of `max_tokens`. Keep it as
  future work in the report, not as a swap.
- **API key scope:** restricted project key, **Model capabilities → Chat completions
  (`/v1/chat/completions`) = Request**. Everything else None. The dropdown says
  *Request*, not Write. **Embeddings permission is NOT needed** — SBERT and CLIP
  run locally. Set a $5–10 project spend cap; whole-week usage is under $1.
- Use `client.chat.completions.create`, not the Responses API — separate permission,
  and chat-completions is the format Ollama/vLLM emulate for the Llama swap.
- Key lives in `.env` (gitignored); `.env.example` is the committed template.

## Data
Corpus: Hugging Face `RealTimeData/bbc_news_alltime`, config `2024-01`.
Fields used: `content`→body, `description`→caption, `section`→story_type,
`top_image`→image URL. All auto-detected by the build script.

**Day 1 output (done, committed to git):**
- 1023 clean pairs from 1562 raw articles; every row has a downloaded image.
- `index_pool` 873 / `test` 150. Body median 613 words.
- 0 nulls, 0 duplicate ids, 0 id or headline overlap between pool and test.
- Images verified by eye against their articles — the join is correct.

**The corpus is committed on purpose** (`data/processed/*.parquet` 5 MB,
`data/images/` 38 MB). Rebuilding is non-deterministic — image download failures
vary by machine, producing different row counts and different splits, which makes
B1-vs-M numbers incomparable across teammates. **Clone, don't rebuild.**

### Known data caveats
- **26.6% of bodies contain video-player boilerplate** (`"This video can not be
  played To play this video you need to enable JavaScript…"`); 31.7% have some junk
  (that, `Follow BBC` footers, `Watch:` teasers). **Strip in `embed.py` before
  chunking** or it becomes retrievable evidence and near-identical junk across 232
  articles will cluster in embedding space.
- **`story_type` is 68% `other`; `sport` has n=1.** BBC's `section` is geographic
  (`Middle East`, `Wales`, `US & Canada`) not topical, and 142 rows have no section.
  `bucket()` in `day1_split.py` only matches UK/politics/business/sci-tech.
  Affects only the Day 5 stratified analysis.
- Caption text is duplicated into body in only 4.1% of rows — checked, small enough
  to ignore, so the text-only baseline is not contaminated by visual info.

## Repo layout
```
src/day1_build_dataset.py   DONE — fetch BBC, download images, write data.parquet
src/day1_split.py           DONE — story_type labels + index_pool/test split
src/inspect_data.py         DONE — read-only data explorer (CLI)
scripts/setup.sh            DONE — one-shot env setup, idempotent
scripts/fix_openmp.sh       DONE — macOS libomp fix
src/embed.py                TODO Day 2 — SBERT + CLIP encoders
src/index.py                TODO Day 2 — build/load/search FAISS
src/retrieve.py             TODO Day 2 — text-only + fused retrieval
src/generate.py             TODO Day 3 — prompt build + LLM + abstention
app/streamlit_app.py        TODO Day 4 — demo with text/multimodal toggle
src/evaluate.py             TODO Day 5 — faithfulness (LLM-judge) + recall@k
docs/setup_guide.md         teammate onboarding + troubleshooting
docs/plan_7day.md           the active plan  (READ THIS)
docs/day1_guide.md          detailed Day 1 steps
docs/plan_full_2week.md     ambitious version + future-work list
results/                    metrics.csv, plots
```
All plans live in `docs/`. A stale duplicate that once sat at the repo root
(`multimodal_rag_plan_7day.md`) has been deleted — `docs/plan_7day.md` is canonical.

## Conventions
- Run scripts from repo root. Fixed seed = 42.
- Normalize embeddings before FAISS (`IndexFlatIP` = cosine on normalized vectors).
- Watch dimension mismatches: MiniLM=384, CLIP=512. A silent dim/normalization bug
  produces garbage retrieval — sanity-check retrieved items by eye.
- Cache LLM outputs by prompt hash to avoid re-billing (`data/llm_cache/`, gitignored).
- `.venv/` and `.env` are gitignored; the corpus is NOT (deliberate, see Data).
- Explore data with `python src/inspect_data.py` (`--row N`, `--search TERM`, `--open`).

## Current status
**Day 1 complete and verified.** Environment installed, corpus built and split,
generation path tested against the live API. Nothing is blocking Day 2.

**Not yet done:** `git init` + first commit. The repo is not yet under version
control (`docs/day1_guide.md` Thing 3 Step 3). Suggested:
```bash
git init
git add -A && git commit -m "Day 1: env + BBC corpus + split + tooling"
```
`.gitignore` already protects `.env`, `.venv/`, and derived artifacts.

**Next: Day 2** — `embed.py` → `index.py` → `retrieve.py`. Fold the boilerplate
strip into `embed.py` before chunking.

## Open decisions (deferred, not blocking)
1. **recall@k design tension.** `plan_7day.md:83` wants recall@k on test items where
   "you know the gold article", but `day1_split.py` removes test rows from the index
   pool — so the gold article is unretrievable and recall@k is 0 by construction.
   Decide whether the retrieval target is the test article's own passages (which must
   then be indexed) or related articles. Settle before Day 5; it may change the split.
2. **story_type bucketing.** Widen `bucket()` so geographic sections map to real
   buckets, or drop stratification and report headline B1-vs-M only. The second is
   defensible and costs nothing.

## Nothing is trained
Common misconception worth restating: this is retrieval-augmented generation.
SBERT and CLIP are **frozen** pre-trained encoders used forward-only; GPT-4o-mini is
a hosted API. There is no training loop, no fine-tuning, no gradients anywhere. The
corpus is a **searchable index**, not training data. Scaling it would make retrieval
*harder and more convincing* (more distractors, tighter confidence intervals), not
the models better.

## Fallback ladder (if behind)
Drop in order: HF deploy → faithfulness eval → abstention → shrink corpus to ~500.
Floor to defend: a local Streamlit demo that retrieves + summarizes with the
text/multimodal toggle on a few hundred real pairs.
