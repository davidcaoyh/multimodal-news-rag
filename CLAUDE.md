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

### CLIP model name — a silent quality trap
Load CLIP as **`ViT-B-32-quickgelu`**, never `ViT-B-32`:

```python
open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
```

OpenAI trained these weights with QuickGELU (a fast approximation of the GELU
activation); open_clip's plain `ViT-B-32` config uses exact `nn.GELU`. Loading the
weights under the wrong config **runs and passes every sanity check** — same
(N, 512) shape, same unit norms, no exception, only a `UserWarning`.

Measured difference on this corpus: per-image cosine between the two versions is
**0.975 mean / 0.916 min**, and image-retrieval **top-5 overlap is only 3.2/5**
(worst query 1/5). So ~a third of the multimodal arm's retrievals change — but
*both* versions return plausible-looking articles, so eyeballing cannot tell them
apart and neither can any Day 2 acceptance check. The reason to use `-quickgelu` is
that it matches the config the weights were trained under, i.e. it reproduces
published CLIP; the other config is a subtly off-spec model. Pinned in
`embed.py:CLIP_MODEL` — don't "simplify" the name away.

### Other environment gotchas
- **`nltk` is installed but its `punkt`/`punkt_tab` data is NOT downloaded**, so
  `nltk.sent_tokenize` raises `LookupError` offline. `embed.py` uses a regex
  sentence splitter instead — no download step for teammates, and deterministic
  across machines, which matters because B1-vs-M numbers are compared across clones.
- `load_dotenv()` resolves relative to the **calling file's directory**, not cwd.
  From a script outside the repo it silently finds nothing and leaves the key
  `None`. Pass an explicit path in throwaway scripts.
- `torch.cuda.is_available()` is `False` on Mac — expected. MPS is available, but
  CPU is already fast enough (see benchmarks).
- The `datasets` library caches to `~/.cache/huggingface`, NOT `data/raw/`.
  `data/raw/` is vestigial (`RAW_DIR` in `day1_build_dataset.py:22` is defined but
  never used).

### Measured performance (M4 MacBook Air, 24 GB) — no GPU needed
**Regenerate with `python -m src.bench` → `results/timings.csv`** (tracked). Do not
hand-edit the numbers below; re-run and copy. Measured 2026-08-01 on the real corpus:

| Step | Measured |
|---|---|
| SBERT, 11,052 real passages | 19.3 s (**573/s**) |
| CLIP ViT-B/32, 1023 images | 10.2 s (101/s) |
| FAISS build, both indexes | 0.004 s |
| **Fused query, end-to-end** | **21 ms** |
| ├ query encode (SBERT + CLIP) | 16.7 ms — **79% of the cost** |
| └ FAISS search, both, k=ntotal | 1.25 ms |
| Peak RSS | 1.87 GB |

Two corrections to earlier planning figures: SBERT is **573/s on real ~100-word
passages, not 4,485/s** (that estimate came from short synthetic strings and
overstated throughput ~8×), and peak RSS is 1.87 GB, not 3.56 GB.

Compute is a non-issue at this scale, and the breakdown says where it isn't: search
is 6% of a query, encoding is 79%. If the Day 4 demo feels slow, cache query
embeddings — a faster index buys nothing. Note `mode="text"` still pays the CLIP
query encode (~11 ms) so every `Hit` carries `s_img` for analysis; skip it when
`alpha == 1.0` if B1 latency ever matters. The only genuinely slow step is image
downloading (0.34 s each, serial) — which is why the corpus is committed, not rebuilt.

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
- **Faithfulness judge = `claude-sonnet-5`, generator stays `gpt-4o-mini`** (Day 5 only).
  GPT-4o-mini grading its own summaries is self-preference bias sitting on the headline
  B1-vs-M number. Judge must be blind to condition, constant across B0/B1/M, temp 0.
  ~$1.60 batched. Needs `ANTHROPIC_API_KEY` in `.env` too. Full rationale, cost table,
  and the rejected reference-summary alternative: `docs/decisions.md`.

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
  articles will cluster in embedding space. **Done** — `clean_body()` in `embed.py`.
- **A second boilerplate family the guides don't list: embedded social-post consent
  blocks** (`This Twitter post cannot be displayed… We ask for your permission before
  anything is loaded…`, ~90 fixed words) in 1.3% of bodies, plus a `Sign up for our
  morning newsletter` footer in 2.2%. Both are stripped. The `Watch:` teasers in the
  guide's table are deliberately **kept** — they are the real video caption that
  follows the stub, i.e. genuine visual evidence.
- **`story_type` is 68% `other`; `sport` has n=1.** BBC's `section` is geographic
  (`Middle East`, `Wales`, `US & Canada`) not topical, and 142 rows have no section.
  `bucket()` in `day1_split.py` only matches UK/politics/business/sci-tech.
  Affects only the Day 5 stratified analysis.
- Caption text is duplicated into body in only 4.1% of rows — checked, small enough
  to ignore, so the text-only baseline is not contaminated by visual info.
- **Pool/test content leakage is 1/150 — measured, negligible.** The split is enforced
  on `id`, and BBC republishes, so id-disjointness alone doesn't guarantee no
  near-duplicate. Measured 2026-08-01: 8 of 150 test rows share a first paragraph with
  a pool row, but 7 are just the video boilerplate; after stripping it, exactly **1**
  shares real content. Second reason `clean_body()` matters — leaving the boilerplate
  in makes 232 unrelated articles look like near-duplicates of each other.

## Repo layout
```
src/day1_build_dataset.py   DONE — fetch BBC, download images, write data.parquet
src/day1_split.py           DONE — story_type labels + index_pool/test split
src/inspect_data.py         DONE — read-only data explorer (CLI)
scripts/setup.sh            DONE — one-shot env setup, idempotent
scripts/fix_openmp.sh       DONE — macOS libomp fix
src/embed.py                DONE — clean_body + chunk + SBERT/CLIP encoders
src/index.py                DONE — build/save/load/search FAISS + smoke checks
src/retrieve.py             DONE — text-only + fused retrieval (run with -m)
src/generate.py             TODO Day 3 — prompt build + LLM + abstention
app/streamlit_app.py        TODO Day 4 — demo with text/multimodal toggle
src/evaluate.py             TODO Day 5 — faithfulness (LLM-judge) + recall@k
docs/setup_guide.md         teammate onboarding + troubleshooting
docs/plan_7day.md           the active plan  (READ THIS)
docs/decisions.md           design decision log (judge model, rejected alternatives)
docs/day1_guide.md          detailed Day 1 steps
docs/plan_full_2week.md     ambitious version + future-work list
src/bench.py                DONE — re-measure timings -> results/timings.csv
results/timings.csv         generated by `python -m src.bench` (tracked)
results/                    metrics.csv, plots
```
All plans live in `docs/`. A stale duplicate that once sat at the repo root
(`multimodal_rag_plan_7day.md`) has been deleted — `docs/plan_7day.md` is canonical.

## Conventions
- Run scripts from repo root. Fixed seed = 42.
- `src/` is a package (`src/__init__.py`). Modules with sibling imports run as
  `python -m src.retrieve`; `embed.py` and `index.py` have none and run either way.
- **`retrieve()` is the experiment's only independent variable.** `mode="text"` is
  literally `alpha=1.0` down the same code path — do not fork it into two functions,
  or a B1-vs-M difference could be an implementation difference. Order inside is
  load-bearing: max-pool passages to article level → filter `include_test` →
  per-query min-max → fuse. Normalizing before filtering lets the excluded gold
  article set the range and silently changes what `alpha` means.
- Use the **raw** `s_text`/`s_img` on a `Hit` for the Day 3 abstention threshold, not
  `score` — `score` is min-max'd per query and so is relative, never absolute.
- Normalize embeddings before FAISS (`IndexFlatIP` = cosine on normalized vectors).
- Watch dimension mismatches: MiniLM=384, CLIP=512. A silent dim/normalization bug
  produces garbage retrieval — sanity-check retrieved items by eye.
- Cache LLM outputs by prompt hash to avoid re-billing (`data/llm_cache/`, gitignored).
- `.venv/` and `.env` are gitignored; the corpus is NOT (deliberate, see Data).
- Explore data with `python src/inspect_data.py` (`--row N`, `--search TERM`, `--open`).

## Current status
**Days 1 and 2 complete and verified.** Corpus built and split; embeddings, FAISS
indexes and both retrieval modes working and eyeballed.

Day 2 artifacts (all in `data/index/`, gitignored — rebuild in ~40 s):
`passages.parquet` **11,052 passages** from all 1023 articles · `text_emb.npy`
(11052, 384) · `img_emb.npy` (1023, 512) · `img_meta.parquet` · `text.faiss` ·
`img.faiss`. Rebuild with `python src/embed.py && python src/index.py`.

All acceptance checks in `day2_guide.md` pass: 0 boilerplate passages, both matrices
unit-norm to 1.2e-07, self-retrieval rank 1 at score 1.00000 (also 28/30 end-to-end
through `retrieve()`), no OpenMP fault, images visibly relevant. Measured:
SBERT 11k passages 19 s, CLIP 1023 images 10 s, FAISS build 0.004 s, **21 ms/query**
fused — 79% of which is encoding the query, not searching. Full breakdown in
`results/timings.csv` (`python -m src.bench`).

**Next: Day 3** — `src/generate.py`. Follow `docs/decisions.md` D7: B1 and M get
byte-identical instruction text (only the evidence block differs), and persist
`test_id | config | query | evidence | summary` — not just the summary, or Day 5 has
nothing to judge against. `retrieve()` already defaults to `include_test=False`;
D7 Rule 3 still wants the explicit assert in `generate.py`.

Still open, unchanged by Day 2: **query text for recall@k** (`day2_guide.md:88` —
headline verbatim is rigged toward B1, `caption` toward M; LLM-paraphrased headline
is the neutral option). Needed Day 5, not before. `retrieve()` takes a plain string
and stays agnostic.

## Decisions — where they live
Design decisions are **not** kept in this file. Two homes:
- `docs/day2_guide.md` § *Decisions to settle before writing code* — D1 index all 1023
  with a `split` column, D2 story_type deferred to Day 5, D3 index the pure image vector.
- `docs/decisions.md` — D4 judge model, D5 reference summaries rejected (+ ROUGE-L open
  item), D6 split stays unstratified, D7 generation contract for Day 3.

Both previously-open items in this file are now settled:
1. ~~**recall@k design tension.**~~ **Resolved** by `day2_guide.md:57`: embed all 1023 and
   carry a `split` column; `retrieve()` takes `include_test` (default `False`). Generation
   always excludes test rows; only the Day 5 recall@k call includes them, because with the
   gold article unindexed recall@k is 0 by definition. The split is unchanged.
2. ~~**story_type bucketing.**~~ **Resolved** by `day2_guide.md:115` + `decisions.md` D6:
   `section` is unfixable (geographic, 68% `other`, `sport` n=1) — do not widen `bucket()`,
   delete it. Day 5 fix is to hand-label only the 150 test items into
   `event_centric` / `abstract_topical`. Nothing in Day 2 reads `story_type`.

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
