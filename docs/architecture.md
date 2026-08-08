# Architecture

> **Current extension:** `M_vision` supplies actual retrieved image pixels to
> GPT-4o-mini, research roles are duplicate-group-safe (707 pool / 166 development /
> 150 final), selected fusion is alpha=0.75, and the judge is GPT-5.6 Luna. Diagrams
> below that name Claude or alpha=0.5 document the preserved inherited baseline.

How the system fits together, what each stage guarantees, and which parts are
load-bearing for the experiment. Read this before changing anything in `src/`.

Design *rationale* lives in [decisions.md](decisions.md) (D1–D16); this file describes the
system as built.

---

## 1. The experiment, in one line

Every component exists to make one comparison fair:

```
                    ┌──────────────────────────────────────┐
                    │  the ONLY thing that differs between │
   query ──────────►│  the two arms is retrieve()          │──────► summary
                    └──────────────────────────────────────┘
                       B1: text-only        M: text + image
```

Three configurations share one pipeline:

| config | retrieval | prompt | role |
|---|---|---|---|
| **B0** | none | "answer from your own knowledge" | hallucination ceiling |
| **B1** | text-only (`α = 1.0`) | numbered evidence | baseline |
| **M** | fused text+image (`α = 0.5`) | same + `[IMAGE n: "caption"]` lines | method under test |
| *M_nocap* | fused, identical to M | same as B1's format | ablation (D16) |

The generator, prompt template, corpus, decoding parameters and evaluation are **held
constant**. If any of them varied per arm, a B1-vs-M difference would be uninterpretable.

---

## 2. Data flow

```
 ┌─ BUILD (once, committed) ────────────────────────────────────────────────┐
 │                                                                          │
 │  HF RealTimeData/bbc_news_alltime 2024-01                                │
 │        │  day1_build_dataset.py   fetch + download 1023 images           │
 │        ▼                                                                 │
 │  data/processed/data.parquet        id, headline, body, image_path,      │
 │        │                            caption, section                     │
 │        │  day1_split.py   seeded shuffle, head/tail cut                  │
 │        ▼                                                                 │
 │  index_pool.parquet (873)   test.parquet (150)                           │
 │        │                          │                                      │
 │        │                          │  queries.py  (gpt-4o-mini, frozen)   │
 │        │                          ▼                                      │
 │        │                    queries.parquet    query      → generation    │
 │        │                                       query_qa   → recall@k     │
 └────────┼──────────────────────────────────────────────────────────────────┘
          │
 ┌─ INDEX (~35 s, gitignored, rebuildable) ────────────────────────────────┐
 │        ▼                                                                 │
 │  embed.py    clean_body() → regex sentence split → 3–5 sentence passages │
 │        ├──► SBERT all-MiniLM-L6-v2 ──► text_emb.npy  (11052, 384)        │
 │        └──► CLIP ViT-B-32-quickgelu ─► img_emb.npy   (1023, 512)         │
 │        │                                                                 │
 │  index.py  ──► text.faiss (IndexFlatIP)   img.faiss (IndexFlatIP)        │
 └──────────────────────────────────────────────────────────────────────────┘
          │
 ┌─ SERVE ──────────────────────────────────────────────────────────────────┐
 │        ▼                                                                 │
 │  retrieve.py   ← THE INDEPENDENT VARIABLE                                │
 │      encode query (SBERT + CLIP)                                         │
 │      text: per-passage scores → max-pool to article level                │
 │      filter by split  →  per-query min-max  →  α·s_text + (1−α)·s_img    │
 │      → list[Hit]                                                         │
 │        │                                                                 │
 │        ▼                                                                 │
 │  generate.py   τ gate → build_prompt() → gpt-4o-mini (T=0) → summary     │
 │        │                                                                 │
 │        ├──► app/streamlit_app.py        the interactive demo             │
 │        └──► results/summaries.csv       the Day 3 → Day 5 interface      │
 └──────────────────────────────────────────────────────────────────────────┘
          │
 ┌─ EVALUATE ───────────────────────────────────────────────────────────────┐
 │        ▼                                                                 │
 │  evaluate.py                                                             │
 │      recall@k  ── retrieve(query_qa, include_test=True) ─► recall.csv    │
 │      judge     ── decompose → verify (claude-sonnet-5)  ─► claims.csv    │
 │      report    ── paired stats + ablation                ─► metrics.csv  │
 └──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Stage detail

### 3.1 Data layer — `day1_build_dataset.py`, `day1_split.py`

Produces the frozen corpus. **Committed to git on purpose** (`data/processed/*.parquet`
5 MB, `data/images/` 38 MB): rebuilding is non-deterministic because image downloads fail
at different rates per machine, which changes row counts and therefore the split. Numbers
computed on different corpora are not comparable.

| guarantee | value |
|---|---|
| clean pairs | 1023, every row has a downloaded image |
| split | 873 index pool / 150 test, seeded (`random_state=42`) |
| integrity | 0 nulls, 0 duplicate ids, 0 id or headline overlap across the split |
| content leakage | 1/150 test rows share real content with a pool row (measured) |

`story_type` is written by `bucket()` but **nothing reads it** — BBC's `section` is
geographic, not topical (68% "other", `sport` n=1). Kept only because removing it would
drop a column from a frozen artefact.

### 3.2 Embedding layer — `embed.py`

Two frozen encoders, forward-only. **Nothing is trained anywhere in this project.**

| | text | image |
|---|---|---|
| model | SBERT `all-MiniLM-L6-v2` | CLIP `ViT-B-32-quickgelu` |
| dim | 384 | 512 |
| unit | passage (3–5 sentences) | article (one image each) |
| count | 11,052 | 1,023 |

Two things here are load-bearing:

- **`clean_body()` runs before chunking.** 26.6% of raw bodies carry video-player
  boilerplate and 31.7% carry some junk. Left in, ~90 near-identical words would cluster
  in embedding space across 232 articles and become retrievable "evidence".
- **The CLIP model name must stay `ViT-B-32-quickgelu`.** The plain `ViT-B-32` config
  loads the same weights under the wrong activation, passes every shape and norm check,
  and silently changes ~a third of the image retrievals.

`torch.no_grad()` is scoped to the forward passes, not applied globally in the loader —
torch's grad mode is thread-local, and Streamlit reruns in a new thread each time.

### 3.3 Index layer — `index.py`

Two `faiss.IndexFlatIP` indexes. Vectors are L2-normalised, so inner product **is** cosine.
Exact search, no approximation — at this scale the whole index is searched in 1.25 ms, so
ANN structures would add failure modes and buy nothing.

Artefacts live in `data/index/` and are **gitignored** — they rebuild deterministically
from the committed corpus in ~35 s.

### 3.4 Retrieval layer — `retrieve.py` — *the independent variable*

```python
retrieve(query, mode="multimodal", k=5, alpha=0.5, include_test=False) -> list[Hit]
```

Order of operations is load-bearing:

1. **Encode** the query with both towers (SBERT 384-d, CLIP text 512-d).
2. **Score exhaustively** on both indexes (`k = ntotal`), not two truncated top-k lists —
   otherwise an article strong on image but outside the text top-k gets fused against an
   implicit zero.
3. **Max-pool text to article level.** The text index is per-passage and the image index
   per-article; fusing them directly would compare different units.
   `s_text(article) = max over that article's passages`.
4. **Filter by split, then normalise.** With `include_test=False` the gold article would
   otherwise set the top of the min-max range and squash every surviving pool score — by a
   *different* amount in each stream, silently changing what α means.
5. **Fuse:** `score = α·s_text + (1−α)·s_img` on the normalised streams.

`mode="text"` is literally `α = 1.0` down the same code path — **not** a second
implementation. `evaluate.py --recall` asserts every run that α=1.0 reproduces B1 exactly
on all 150 queries, which is the only empirical proof that a B1-vs-M difference cannot be
an implementation difference.

Each `Hit` carries both the **fused** `score` and the **raw** cosines `s_text` / `s_img`.
The distinction matters: `score` is min-max'd per query so the top hit is ~1.0 on every
query however poor the match, and is therefore useless as an absolute threshold.

### 3.5 Generation layer — `generate.py`

```
query ─► retrieve() ─► τ gate ─► build_prompt() ─► generate() ─► summary
```

- **τ gate.** If `max(h.s_text for h in hits) < τ` (0.35), return `INSUFFICIENT_EVIDENCE`
  without calling the LLM. Gated on the **raw** cosine, and on the max over the top-k
  rather than `hits[0]` — gating on rank 1 would make the gate rank-order dependent, and
  M's ranking is partly image-driven, so the arms would abstain at different rates for a
  reason that is not evidence quality.
- **Prompt.** B1, M and M_nocap share **byte-identical** instruction text; only the
  evidence block differs (`_check_prompt_parity()` asserts it). B0 is the deliberate
  exception — "use only the evidence" is undefined with no evidence.
- **Generator.** `gpt-4o-mini`, `temperature=0`, `seed=42`, behind a single
  `generate(prompt)` function so a local Llama could swap in. Cached by prompt hash in
  `data/llm_cache/`.
- **Leakage assert.** No retrieved id may appear in the test id set. A test article in the
  generation evidence would push faithfulness to ceiling and *look like a great result*.

### 3.6 Evaluation layer — `evaluate.py`

Two metrics, because **any metric computed on the text stream alone is maximised by B1 by
definition** — B1 *is* the argmax of that stream. A fair comparison needs ground truth
outside both streams.

**recall@k** — free, no API. `query_qa` + `include_test=True`; the only call site permitted
to pass `include_test=True`, since with the gold article unindexed recall is 0 by
construction.

**faithfulness** — two judge passes, `claude-sonnet-5`, blind to the condition label:

```
summary ──decompose──► atomic claims ──verify(evidence, claims)──► supported y/n
                       (blind to evidence)
```

Pass 1 is blind to the evidence deliberately: a decomposer that can see the evidence shapes
claim boundaries to fit it, and faithfulness is a ratio over that granularity.

Determinism without `temperature=0` (removed from the Claude 5 API, returns 400): disabled
thinking + a JSON schema constraining the verdict to a boolean + prompt-hash caching.

### 3.7 Demo layer — `app/streamlit_app.py`

Side-by-side B1 and M on one query, with live `k` / `α` / `τ` sliders. Renders **three**
abstention states, not two: τ-gated (no LLM call), model-refused (call made, model
declined), and normal.

The two on-screen retrieval numbers are labelled **diagnostics** and cannot answer B1-vs-M —
see §4.

---

## 4. Invariants — break these and the experiment is void

| # | invariant | enforced by |
|---|---|---|
| 1 | `retrieve()` is the only thing that differs between B1 and M | one code path; α=1.0 identity check in `--recall` |
| 2 | Generator, decoding params and prompt text are constant across arms | `_check_prompt_parity()` |
| 3 | Normalise **after** filtering by split | order in `retrieve()`; changing it silently redefines α |
| 4 | Text max-pools to article level **before** fusion | `np.maximum.at` in `retrieve()` |
| 5 | τ gates on raw `s_text`, never on the fused `score` | `summarize()` |
| 6 | No test article may enter the generation evidence | assert in `summarize()` |
| 7 | `include_test=True` only in the recall@k call | default `False` |
| 8 | Judge is a different model family from the generator, and blind to condition | `_assert_prompts_blind()` |
| 9 | Never mix α values within one reported table | one α per run; `summaries.csv` is α=0.5 |
| 10 | A refusal is what the summary text says, not the `abstained` flag | `refusal_kind()` — three modes, not two |

**Invariant 10 is the most easily broken.** `abstained` is the τ gate only. The model can
also emit `INSUFFICIENT_EVIDENCE` itself, *and* refuse in prose without the token at all.
Counting only the flag understates refusals by ~3×.

### Why no text-stream metric can answer B1 vs M

B1 selects the k highest-`s_text` articles, so its mean `s_text` is the **maximum
achievable over any k-subset**. Measured: M's mean is lower on 145/150 queries, tied on 5,
**higher on 0**. Labelled "evidence quality", this turns a mathematical identity into an
apparent finding that the method under test is worse. The demo therefore labels its numbers
as diagnostics, and the real comparison uses recall of the withheld gold article or the
external judge.

---

## 5. Data contracts

Each stage's output is a stable interface the next stage reads. Changing a schema means
re-running everything downstream.

| artefact | key columns | committed |
|---|---|---|
| `data/processed/data.parquet` | `id, headline, body, image_path, caption, section` | ✅ |
| `index_pool.parquet` / `test.parquet` | + `story_type, split` | ✅ |
| `data/processed/queries.parquet` | `test_id, headline, query, query_qa, mode` | ✅ |
| `data/index/passages.parquet` | `article_id, text` | ❌ rebuildable |
| `data/index/img_meta.parquet` | `id, headline, caption, image_path, split` | ❌ rebuildable |
| `text_emb.npy` / `img_emb.npy` | `(11052, 384)` / `(1023, 512)`, unit-norm | ❌ rebuildable |
| `results/summaries.csv` | `test_id, config, query, evidence, summary, abstained, retrieved_ids, top_s_text, top_s_img, n_hits` | ✅ |
| `results/claims.csv` | `test_id, config, claim_index, claim, supported, reason` | ✅ |
| `results/recall.csv`, `metrics.csv` | see report | ✅ |

**`summaries.csv` persists the `evidence`, not just the summary.** The Day 5 judge scores
each claim against *the evidence that was actually in the prompt*. Storing only the summary
would force retrieval to be re-run and hoped to match.

---

## 6. Extension points

**Swap the generator** — replace the body of `generate()`. Chat-completions format was
chosen because Ollama and vLLM emulate it, so a local Llama needs only a `base_url` change.
Re-run all three arms; never mix generators across arms.

**Add an arm** — follow `M_nocap` (D16): add to `_MODE` and `_WITH_IMAGES` in `generate.py`,
write to a *separate* CSV, and add it to `ABLATION` in `evaluate.py`. Do **not** add it to
`CONFIGS`, which defines the paired set — that would silently redefine what every earlier
number referred to.

**Change α** — `--alpha` is already a flag. One α per reported table; mixing rows from two
α values voids the paired comparison.

**Add a metric** — it must use ground truth outside both score streams, or it is won by B1
by construction (§4).

**Scale the corpus** — `day1_build_dataset.py --target N`. Note this invalidates every
committed number and requires a full re-run. See [future_work.md](future_work.md) §2.1 for
what scaling would and would not buy.

---

## 7. Cost and performance

Measured on an M4 MacBook Air, 24 GB, no GPU (`python -m src.bench`):

| step | measured |
|---|---|
| SBERT, 11,052 passages | 19.3 s (573/s) |
| CLIP, 1,023 images | 10.2 s (101/s) |
| FAISS build, both indexes | 0.004 s |
| **fused query, end-to-end** | **21 ms** — 79% query encoding, 6% search |
| peak RSS | 1.87 GB |

Compute is a non-issue; the only genuinely slow step is image downloading (~0.34 s each,
serial), which is why the corpus is committed rather than rebuilt.

API spend to date: **$7.24** — generation $0.29, judge $6.95 (1,015 calls). All LLM calls
are cached by prompt hash, so re-runs cost $0.
