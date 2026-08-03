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
- The generation task is **summarization of the retrieved evidence**, not question
  answering, and the prompt explicitly accepts partial relevance (D10). Posed as QA it
  refuses on ~46% of items, because D1 withholds the article that holds the answer.
- Guardrail: retrieval-confidence abstention — if `max(s_text)` over the top-k hits is
  below **tau=0.35**, return `INSUFFICIENT_EVIDENCE`. Raw cosine, never the fused
  `score`, which is min-max'd per query and so is relative (D9). Note the model can also
  emit `INSUFFICIENT_EVIDENCE` on its own, with `abstained=False` — any refusal rate must
  check the summary text, not just that flag.

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

### torch's grad mode is thread-local — cost an hour on Day 4
`embed.py` used to call `torch.set_grad_enabled(False)` **once**, inside the lazy
CLIP loader. That works from the CLI, where load and encode share a thread. It
breaks under Streamlit, which runs every rerun in a **new thread**: the thread that
loaded CLIP got grad disabled, then every later rerun found the model already cached,
skipped the loader, and encoded with grad on — `RuntimeError: Can't call numpy() on
Tensor that requires grad`. The first query worked and the second failed, so it looked
like a Streamlit caching bug rather than an encoder bug.

Fixed by scoping `with torch.no_grad():` to the forward passes in `embed_clip_text()`
and `embed_images()`. Grad mode does not affect forward values — verified the
recomputed embeddings are **bit-identical** (`max|Δ| = 0.0`) to the committed
`img_emb.npy` / `text_emb.npy`, so no Day 2 or Day 3 number moved. Do not "tidy" this
back into a single global call.

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
  B1-vs-M number. Judge must be blind to condition and constant across B0/B1/M.
  ~$2.80. Needs `ANTHROPIC_API_KEY` in `.env` too. Full rationale, cost table,
  and the rejected reference-summary alternative: `docs/decisions.md`.
- **The judge cannot use `temperature=0`** — the Claude 5 family removed `temperature`,
  `top_p`, `top_k` and returns **HTTP 400** on any non-default value. Adding it back fails
  the whole run. Determinism instead comes from `thinking={"type":"disabled"}` + a
  `output_config.format` json_schema (the verdict is a constrained boolean, so variance
  cannot reach the number) + prompt-hash caching. **The generator is unaffected** —
  `gpt-4o-mini` is an OpenAI chat-completion where `temperature=0` still works and is
  still load-bearing. See D13.

## Data
Corpus: Hugging Face `RealTimeData/bbc_news_alltime`, config `2024-01`.
Fields used: `content`→body, `description`→caption, `section`→story_type,
`top_image`→image URL. All auto-detected by the build script.

**Day 1 output (done, committed to git):**
- 1023 clean pairs from 1562 raw articles; every row has a downloaded image.
- `index_pool` 873 / `test` 150. Body median 613 words.
- 0 nulls, 0 duplicate ids, 0 id or headline overlap between pool and test.
- Images verified by eye against their articles — the join is correct.

**Day 3 addition:** `data/processed/queries.parquet` — two rewrites of each test
headline (`test_id, headline, query, query_qa, mode`), built once by
`python -m src.queries`. `query` is a broad topic phrase and drives **generation**;
`query_qa` is a pinpoint question for **Day 5 recall@k**. They have to differ — a
question naming the withheld article's unique subject asks for a fact the pool
provably lacks, which made the model refuse on ~45% of items. See D8. Frozen and
committed for the same reason as the corpus.

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
src/queries.py              DONE — test headlines -> topic + question queries (D8)
src/generate.py             DONE — prompt build + LLM + abstention + batch runner
app/streamlit_app.py        DONE — demo: side-by-side B1 vs M + abstention UI
src/evaluate.py             DONE — faithfulness judge + recall@k + alpha sweep (run with -m)
docs/setup_guide.md         teammate onboarding + troubleshooting
docs/plan_7day.md           the active plan  (READ THIS)
docs/decisions.md           design decision log D4-D12 + rejected alternatives
docs/day1_guide.md          detailed Day 1 steps
docs/day2_guide.md          detailed Day 2 steps + its own pre-code decisions
docs/plan_full_2week.md     ambitious version + future-work list
handoff/                    per-day handoffs (gitignored, local only)
src/bench.py                DONE — re-measure timings -> results/timings.csv
results/summaries.csv       450 rows, the Day 3 -> Day 5 interface (tracked)
results/timings.csv         generated by `python -m src.bench` (tracked)
results/recall.csv          alpha sweep, `python -m src.evaluate --recall` (free, tracked)
results/claims.csv          5,146 per-claim judge verdicts, Day 5 (tracked)
results/metrics.csv         the headline B1-vs-M table + ablation, Day 5 (tracked)
results/summaries_ablation.csv  150 rows, the M_nocap caption ablation (D16, tracked)
docs/evaluation_report.md   Day 5 findings report + proposed 2-week follow-on work
results/                    metrics.csv, plots
docs/screenshots/           Day 4 demo captures for the report (tracked)
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
- Use the **raw** `s_text`/`s_img` on a `Hit` for the abstention threshold, not `score` —
  `score` is min-max'd per query and so is relative, never absolute. Implemented as
  `max(h.s_text for h in hits) < tau`; max over the top-k rather than `hits[0]` so the
  gate is not rank-order dependent, since M's ranking is partly image-driven (D9).
- **`abstained` is the tau gate only.** The model can return `INSUFFICIENT_EVIDENCE`
  itself with `abstained=False`. Anything reporting a refusal rate must check the summary
  text. Conflating the two is how a broken 450-row run passed every check on Day 3.
- Normalize embeddings before FAISS (`IndexFlatIP` = cosine on normalized vectors).
- Watch dimension mismatches: MiniLM=384, CLIP=512. A silent dim/normalization bug
  produces garbage retrieval — sanity-check retrieved items by eye.
- Cache LLM outputs by prompt hash to avoid re-billing (`data/llm_cache/`, gitignored).
- `.venv/` and `.env` are gitignored; the corpus is NOT (deliberate, see Data).
- Explore data with `python src/inspect_data.py` (`--row N`, `--search TERM`, `--open`).

## Current status
**Days 1–4 complete and verified.** Corpus built and split; embeddings, FAISS
indexes and both retrieval modes working and eyeballed; generation running end to end
with `results/summaries.csv` (450 rows) written — the Day 5 interface exists; the
Streamlit demo runs.

Day 4 — `streamlit run app/streamlit_app.py`. Default view is **B1 and M side by side**
on one query (the "toggle" is a sidebar radio: side-by-side / B1 / M, plus an optional
B0 pane). Per-arm it shows the summary, two retrieval statistics, the literal prompt
that was sent, and the retrieved articles with thumbnails, per-stream scores and
passages. Articles one arm found and the other did not are badged `only B1` / `only M`
— the independent variable made visible. k, alpha and tau are live sliders (alpha=1.0
collapses M onto B1, which is a useful thing to show).

Verified headlessly with `streamlit.testing.v1.AppTest` — cold start, a real query
side-by-side, the off-topic query gating **both** arms, and the single-arm + B0 views.
Screenshots for the report are committed in `docs/screenshots/`.

Three things worth knowing before touching the app:
- **The app shows two retrieval statistics, and they do different jobs.** `gate score`
  = `max s_text` over the top-k — the tau statistic (D9), on screen only to explain the
  abstention state. It is a **bad** arm-comparison number: B1 ranks purely by `s_text`,
  so its max is the pool-wide maximum, and M matches it whenever that article survives
  fusion — **identical in both arms on 109/150 test queries (73%)**, and pinned across
  every k and alpha when one article is the argmax of both streams. Side by side that
  reads as a broken UI. `mean text sim` = **mean** `s_text` over the retrieved set moves
  monotonically with alpha and separates the arms on **145/150 (97%)**. But it is **not
  a quality score and must never be labelled as one**: B1 selects the k highest-`s_text`
  articles, so its mean is the maximum achievable over any k-subset and M is `<=` B1 by
  identity — measured **145 lower, 5 tied, 0 higher**. An earlier revision called it
  "evidence quality", which made the multimodal arm look worse by construction. Read the
  gap as how far image fusion moved the set. Neither on-screen number can answer
  B1-vs-M; that needs ground truth outside both streams.

### Retrieval recall — measured 2026-08-02, free (no API), a Day 5 preview
Recall of the withheld gold article, `query_qa` + `include_test=True` (D8), n=150:

Full sweep re-run on Day 5 (`python -m src.evaluate --recall`, free, ~90 s) →
`results/recall.csv`. Reproduces Day 4's independently-computed numbers exactly.

| config | @1 | @5 | @10 | B1-only@5 | M-only@5 | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| B1 text-only | 0.733 | **0.913** | 0.940 | — | — | — |
| M alpha=0.9 | 0.733 | 0.933 | 0.953 | **0** | 3 | 0.250 |
| M alpha=0.75 | 0.727 | **0.940** | 0.960 | **0** | 4 | 0.125 |
| M alpha=0.5 (current default) | 0.673 | **0.860** | 0.900 | 12 | 4 | 0.077 |
| M alpha=0.25 | 0.573 | 0.740 | 0.787 | 30 | 4 | <0.001 |
| M alpha=0.0 (pure image) | 0.353 | 0.600 | 0.693 | 50 | 3 | <0.001 |

**The default alpha=0.5 costs recall** (0.860 vs 0.913 @5; 12 B1-only wins vs 4 M-only,
p=0.077). **alpha=0.75 gains it and strictly dominates** — 4 M-only wins, **0 B1-only
losses** — but only 4 discordant pairs, so p=0.125: underpowered, not significant. The
curve peaks in the **0.75–0.9** band and falls off hard below 0.5. Direction is
unambiguous; magnitude is not established.

**alpha=1.0 reproduces B1 exactly on all 150 queries** (identical @1/@5/@10, 0 discordant
pairs). That is not a coincidence to note in passing — it is the only empirical proof that
`mode="text"` and `mode="multimodal", alpha=1.0` really are one code path, so a B1-vs-M
difference cannot be an implementation difference. `--recall` asserts it every run.

This is retrieval quality, NOT the research question — the headline result is summary
faithfulness, and a different retrieved set can be more faithful at lower recall. Do
**not** change the default: `results/summaries.csv` was generated at alpha=0.5, mixing
rows from two alphas silently voids the paired comparison (D11).
- Abstention has **three** UI states, not two: tau-gated (red, no LLM call made),
  model-refused (amber, the call happened and the model declined), and normal. This is
  the `abstained`-is-not-the-refusal-rate trap rendered as UI.
- The searched query lives in `st.session_state["active"]`, not a one-shot "was the
  button just clicked" flag. Streamlit reruns the whole script on every widget
  interaction, so a one-shot flag makes moving the alpha slider *clear* the results
  instead of recomputing them — which is the entire point of the sliders.

Day 3 results (`python -m src.generate --run`, 607 s, $0.22 all-in for the whole day):

| config | tau-gated | model refused | usable | median words |
|---|---:|---:|---:|---:|
| B0 | 0 | 0 | 150 | 84 |
| B1 | 2 | 15 | **133** | 88 |
| M | 5 | 14 | **131** | 91 |

**These refusal counts are wrong — corrected on Day 5 (D15).** They count only the literal
`INSUFFICIENT_EVIDENCE` token and miss a third mode: **soft refusals**, prose that declines
without the token (*"The evidence does not provide information about X. It covers unrelated
topics including A and B"*). Adding them, the true refusal rate is **35.3% B1 / 36.0% M**,
roughly 3x the figure in the table above. Soft refusals also *inflate* faithfulness, because
their claims are meta-statements about the evidence and so are supported by construction.
Any refusal or usability count in this file predating Day 5 undercounts for this reason.

**129/150 items have a summary from both arms** — that is the paired sample the Day 5
B1-vs-M number rests on, and refusals are tightly paired (15 shared, 2 B1-only, 4 M-only).
Retrieval contrast holds: **53% of M's top-5 articles never appear in B1's**, identical
top-5 on only 5/150 queries. 0 test ids in evidence (D7 Rule 3).

The residual ~12% refusal is a **corpus-coverage limit, not a bug** — the pool is 873
articles from one month, so some test topics genuinely have no coverage. Measured scaling
curve (subsampling the pool, no API cost): median best `s_text` rises **+0.034 per doubling**
of pool size — 0.419 at n=100, 0.492 at n=400, 0.526 at n=873. Report this as the
future-work scaling argument rather than rebuilding; see the Day 3 handoff.

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

### Day 5 results — the headline (judge run complete, 2026-08-02, $5.35, 769 calls)

`python -m src.evaluate --judge && --report` → `results/claims.csv` (3,916 claims from
387/387 summaries), `results/metrics.csv`.

| config | faithfulness (usable only) | hallucination | true refusal rate |
|---|---:|---:|---:|
| **B0** no retrieval | **0.243** | **0.757** | 0% |
| **B1** text-only RAG | **0.901** | 0.099 | 35.3% |
| **M** multimodal RAG | **0.886** | 0.114 | 36.0% |

**Retrieval is what kills hallucination: 0.24 → 0.90, a 3.7x gain.** That is the result
the project actually has.

**B1 vs M is a null result and must be reported as one.** Paired on 90 items where both
arms produced a genuine summary: difference **-0.0150** (M lower), 95% CI
**[-0.0431, +0.0125]**, Wilcoxon **p = 0.252**, 32 items M better vs 41 B1 better. The CI
spans zero. Multimodal fusion neither helps nor hurts faithfulness at this scale — and
the CI is tight enough to say the effect, if any, is smaller than ±4 points. Combined
with the recall sweep (alpha=0.5 *costs* recall), the honest conclusion is that **image
fusion at alpha=0.5 buys nothing measurable**, and the interesting finding is retrieval
itself plus the alpha curve.

Do not report the contaminated cut (n=125, diff -0.0146, p=0.245) as the headline; both
cuts agree, which is why the null is trustworthy. See D15.

**The caption ablation (D16) localises the null to both channels.** M differs from B1 in
two ways at once — image fusion changes *which* articles are retrieved, and captions add
*text* to the prompt. `M_nocap` (identical retrieval to M, `[IMAGE n:]` lines stripped)
separates them:

| comparison | isolates | diff | 95% CI | p |
|---|---|---:|---|---:|
| B1 -> M_nocap | retrieval channel | +0.0046 | [-0.0236, +0.0321] | 0.640 |
| M_nocap -> M | caption channel | -0.0118 | [-0.0393, +0.0135] | 0.530 |

Both individually null, not cancelling. **The caption channel is inert** — ~93 words of
image description per prompt moves faithfulness by -1.2 points. That rules out "the
captions never reached the model" as an explanation. What is falsified is *caption-mediated*
multimodality; the model never sees pixels, so a vision-capable generator arm is the natural
next experiment.

**Retrieval confidence gates answering, not accuracy.** Spearman(top `s_text`,
faithfulness) = +0.146 (p=0.15) B1 / +0.056 (p=0.59) M — flat. But refusal by
retrieval-confidence quartile runs **72.4% -> 41.9% -> 26.7% -> 1.3%**. Better evidence makes
the model willing to answer; conditional on answering it is ~90% faithful regardless. This
is why scaling the pool would fix the refusal rate and **not** the B1-vs-M question.

**Power:** SD of the paired difference 0.1351, n=90 -> minimum detectable effect **4.0
points** at 80% power; observed 1.5. Detecting an effect that small needs **636 pairs**,
which requires a bigger TEST set, not a bigger index. Projected from the measured scaling
curve, a 64x larger pool moves the detectable effect only 4.0 -> 3.4 points.

**Day 5 remaining:** the 50-claim hand validation (`results/validation_sample.csv` is
exported and blind — fill `your_label`, then `--validate`). Everything below is
provisional until that lands. Full write-up and a ranked 2-week follow-on plan:
`docs/evaluation_report.md`.

```bash
python -m src.evaluate --recall              # DONE — free, results/recall.csv
python -m src.evaluate --dry-run             # prints the exact judge request, sends nothing
python -m src.evaluate --judge --limit 3     # ~$0.07 smoke test, do this first
python -m src.evaluate --judge               # 387 summaries, ~$2.80, ~6 min
python -m src.evaluate --report              # -> results/metrics.csv
python -m src.evaluate --sample-validation   # 50 blind claims -> hand-label (D4, ~30 min)
python -m src.evaluate --validate            # agreement + Cohen's kappa
```

Scope settled for Day 5: faithfulness at **alpha=0.5 only** (the committed run), recall
swept for free across alpha, ROUGE-L dropped (D5), judge validated on 50 hand-labeled
claims (D4), story_type stratification dropped — `bucket()` stays in `day1_split.py`
because it writes the committed parquets, but nothing may read `story_type`.

The judge path was validated end-to-end against a mocked judge before spending anything:
387/387 summaries, 129 items in every config, stats and CSVs correct. That mock run caught
a real bug — the blindness check scanned rendered prompts and matched the word
*"condition"* inside real BBC summaries, silently dropping 11 summaries and reporting
n=125 as if it were 129. Now checked on the templates only, and `run_judge()` exits
non-zero rather than reporting a partial run (D13).

Day 5 must not use any metric computed on the text stream alone (D12).

Query text is **settled** (D8) and no longer open: `data/processed/queries.parquet` carries
both `query` (topic phrase → generation) and `query_qa` (pinpoint question → Day 5 recall@k).

## Decisions — where they live
Design decisions are **not** kept in this file. Two homes:
- `docs/day2_guide.md` § *Decisions to settle before writing code* — D1 index all 1023
  with a `split` column, D2 story_type deferred to Day 5, D3 index the pure image vector.
- `docs/decisions.md` — D4 judge model, D5 reference summaries rejected (+ ROUGE-L now
  resolved: dropped), D6 split stays unstratified, D7 generation contract for Day 3, D8
  query text = rewritten headline in two variants, D9 abstention threshold tau=0.35, D10
  summarization task + relevance-tolerant prompt, D11 alpha stays 0.5 for the committed run
  and is swept on Day 5, D12 what the demo's retrieval statistics may and may not claim,
  **D13 the judge cannot use temperature=0 (400 on Claude 5) and how determinism is
  recovered**, **D14 B0's claims are judged against the union of B1's and M's evidence**, **D15 three refusal modes, the third contaminates faithfulness**, **D16 the caption ablation and what it establishes**.

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
