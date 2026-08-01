# Day 2 — Detailed Guide: Embeddings + FAISS + Retrieval

**Goal by end of today:** you can type a sentence and get back the most relevant
articles from the corpus — **twice**, once text-only and once text+image fused.
Nothing else. No LLM, no summaries, no demo.

If `retrieve("humza yousaf labour deal", mode="multimodal")` returns five
plausible articles with plausible images, Day 2 is a success.

---

## The big picture — why today is the whole project

Day 1 produced 1023 rows of body + image + caption. They are **inert**: readable,
not searchable. Today makes them searchable.

```
Day 1 ✅   1023 articles: body, image, caption  →  data/processed/*.parquet
           ↓
Day 2 ⬅    embed.py     text → vectors, images → vectors
           index.py     vectors → fast nearest-neighbour search
           retrieve.py  query → top-k articles   [switch: text-only | fused]
           ↓
Day 3      evidence → prompt → GPT-4o-mini → summary
Day 4      Streamlit demo with the toggle
Day 5      faithfulness + recall@k
```

The three configs differ in **exactly one place**:

| | Retrieval | Generator | Prompt | Temperature |
|---|---|---|---|---|
| B0 | none | gpt-4o-mini | same | 0 |
| B1 | text-only | gpt-4o-mini | same | 0 |
| M | text + image fused | gpt-4o-mini | same | 0 |

The generator is held constant on purpose — if it varied, a B1-vs-M difference
would mean nothing. So **the switch inside `retrieve.py` is the independent
variable of this entire project.** Today is not infrastructure you build before
the experiment. Today *is* the experiment; Days 3–7 read out the result.

That is why the decisions below matter more than they look.

---

## Decisions to settle before writing code

Mechanical test for "does this block me": **does `embed.py` refuse to run until
I answer it?**

| Decision | Blocks today? | Why |
|---|---|---|
| 1. What goes in the index | **Yes** | Line one of `embed.py` reads a dataframe — which one? |
| 3. What the image vector contains | **Yes** | Determines what `embed.py` computes and stores |
| 2. `story_type` buckets | **No** | Day 5 analysis label; nothing today touches it |

### Decision 1 — index all 1023, not just the 873 pool

`day1_split.py:35` removes test rows from the index pool. That was meant as a
leakage guard, but it makes retrieval evaluation impossible: for a query built
from test article X, X is not in the index, so it can never be returned.

recall@k over `n` queries, with `g_i` the gold (source) article for query `i` and
`R_i` the top-k returned:

```
recall@k = (1/n) · Σ 1[ g_i ∈ R_i ]
```

With test rows excluded, `1[g_i ∈ R_i] = 0` for every query. recall@k is exactly
0 regardless of retrieval quality. Not a bug in retrieval — a metric that cannot
fire.

**"Leakage" was the wrong frame.** Hallucination is scored against the evidence
the model was *shown*, not against hidden gold. Finding the source article is
the retriever's job; hiding it doesn't make the study rigorous, it makes it
unmeasurable. B0/B1/M stays fair either way — B0 doesn't retrieve, B1 and M
retrieve from the identical index.

**Do:** embed all 1023, carry a `split` column so the pool-only variant is a
post-hoc filter. FAISS builds in 0.011 s; there is no cost to indexing everything.

**Why it blocks:** the rebuild itself is cheap (~20 s). The cost is finding out
late — build on 873 today, discover on Day 5 that recall@k is structurally 0,
and you are redesigning the evaluation with two days left. Schedule risk, not
compute.

**Still open, but not until Day 5 — query text.** Both obvious choices are rigged:

| Query = | Rigged toward | Why |
|---|---|---|
| headline verbatim | text (B1) | headline tokens recur through the body; SBERT saturates, recall@5 ≈ 1.0 both arms, no visible difference |
| `caption` field | image (M) | caption describes the image; if captions enter the image index it is straight leakage |

Neutral option: **LLM-paraphrase the headline into a natural question.** The
headline is the one field indexed in *neither* stream, so both arms bridge the
same gap. 150 gpt-4o-mini calls, well under a cent, saved to
`data/processed/queries.parquet`. Fallback if behind: headline verbatim, but
report **recall@1 / MRR** so a saturating metric still discriminates.

Today this only means: `retrieve()` takes a plain query string. Stay agnostic.

### Decision 3 — index the pure image vector

`CLAUDE.md` says "images + captions → CLIP" without saying whether the *indexed
vector* is the image alone or blended with the caption's CLIP text embedding.

**Do:** index the image vector alone. The caption is a **payload**, carried
into the Day 3 prompt as `[IMAGE k: "caption"]`, never into the index vector.

Blend caption text into the index vector and M becomes partly a second text
retriever — "does the *image* help?" stops being answerable no matter how clean
the numbers look. Keep the blend as a one-line ablation knob for the report.

### Decision 2 — `story_type`: ignore today, fix Day 5

Not blocking. Nothing in `embed.py` / `index.py` / `retrieve.py` reads it.

For the record: `section` is unfixable for this purpose — it is geographic
(`Norfolk`, `Middle East`, `Wales`), 68% falls to `other`, `sport` is n=1, and
142 rows have no section. Widening `bucket()` just relabels geography as topic.

But the proposal's hypothesis is explicit — *multimodal helps most for
event-centric stories, least for abstract topics* — so dropping stratification
disconnects the report from its own claim. Day 5 fix: abandon `section`, label
**only the 150 test items** into `event_centric` / `abstract_topical` / `other`,
by hand (~20 min) or one gpt-4o-mini pass. Report as descriptive with n ≈ 50 per
bucket and state plainly that it is underpowered. Delete `bucket()` rather than
leave a misleading function in the repo.

---

## The 3 things to work on today

Do them in order. Don't start #2 until #1 writes clean arrays; don't start #3
until #2 survives a smoke search.

### Thing 1 — `src/embed.py` (target: ~1 hr)

Reads `data/processed/data.parquet` (all 1023), writes to `data/index/`.

**Step 1 — `clean_body(text)`.** Strip before chunking, or the junk becomes
retrievable evidence and near-identical boilerplate clusters in embedding space:

| Pattern | % of bodies |
|---|---|
| `This video can not be played` / `To play this video you need to enable JavaScript` | 26.1% |
| `Watch:` teasers | 8.6% |
| `Follow BBC…` footers | 7.2% |

Strip **only the fixed phrase** — the sentence following the video boilerplate is
a real video caption worth keeping.

**Step 2 — `chunk(body, n_sents=4, stride=3)`** → `(article_id, passage_idx, text)`.
Expect **~9,800 passages** (median 26 sentences/article, mean 672 words).

**Step 3 — encoders.**

| Function | Model | Output | Notes |
|---|---|---|---|
| `embed_text()` | SBERT `all-MiniLM-L6-v2` | (N, 384) | `normalize_embeddings=True`, float32 |
| `embed_images()` | CLIP ViT-B/32 image tower | (1023, 512) | normalize manually |
| `embed_clip_text()` | CLIP ViT-B/32 **text** tower | (·, 512) | queries; same space as images |

Exclude `headline` from indexed text — see Decision 1's query note.

**Step 4 — write** `passages.parquet`, `text_emb.npy`, `img_emb.npy`,
`img_meta.parquet`.

Checkpoint: shapes correct, no passage contains `enable JavaScript`.

### Thing 2 — `src/index.py` (target: ~30 min)

`build(emb)` → `faiss.IndexFlatIP`; `save()` / `load()`; `search(q, k)`.

**Run `bash scripts/fix_openmp.sh` before the first run.** Re-run it after *any*
pip install touching torch/faiss/sklearn.

**Put a smoke search in `__main__`.** The OpenMP abort fires on the first
*search*, not on import — imports look fine and the crash surfaces later inside
`retrieve.py`, where you will misdiagnose it as a retrieval bug. Force it here,
where you know what it is.

Checkpoint: a 1-query search returns without exit 134/139.

### Thing 3 — `src/retrieve.py` (target: ~1 hr)

`retrieve(query, mode, k=5, alpha=0.5)` → articles with best passage(s),
image path, caption. This is the file the rest of the week depends on.

**The trap: granularity mismatch.** The text index is **per-passage** (~9,800
entries); the image index is **per-article** (1023 entries). Fusing raw passage
scores against image scores compares different units. Aggregate first:

```
article_score_text = max(scores of that article's passages)     ← then fuse
```

Then, per query, min-max normalize each stream and combine:

```
score = alpha · s_text + (1 - alpha) · s_img          (default alpha = 0.5)
```

Per-query normalization matters — the two score streams have different scales
and a global normalization leaks cross-query information.

Checkpoint: 5 eyeball queries, text vs multimodal side by side.

---

## Acceptance checks before calling Day 2 done

- [ ] No passage contains `enable JavaScript`
- [ ] Shapes (N, 384) and (1023, 512); **every row unit-norm to < 1e-5**
- [ ] **Self-retrieval:** re-embed a known passage, search it, it returns rank 1
      at score ≈ 1.0
- [ ] FAISS smoke search does not segfault
- [ ] 5 eyeball queries, both modes, images visibly relevant
- [ ] Timings logged

The normalization and self-retrieval checks are the important ones. `IndexFlatIP`
is cosine similarity *only* on normalized vectors — un-normalized input still
returns k results, ranked wrong, with no error. Self-retrieval catches
dim/dtype/index-order bugs in one line.

---

## One data caveat to handle while you're in here

**89 of 1023 articles (8.7%) share an exact first paragraph** — BBC republishes
and updates stories. Near-duplicates will co-occupy top-k and shrink effective
evidence diversity.

Article-level max-pooling already mitigates it. If the eyeball check still shows
visible duplicates in top-5, dedupe by first-paragraph hash at retrieval time and
note it in the report.

Related, already checked: the body's first paragraph is *not* the `caption`
field (word-overlap Jaccard 0.12), so the caption remains genuinely distinct
visual evidence and the text-only baseline is not contaminated by it.

---

## Time budget

| | |
|---|---|
| `embed.py` | ~1 hr |
| `index.py` | ~30 min |
| `retrieve.py` | ~1 hr |
| Checks + eyeballing | ~45 min |
| **Total** | **~3–4 hrs** |

Compute is a non-issue: SBERT ~2 s, CLIP ~13 s, FAISS build 0.011 s. The time
goes to the fusion granularity and the normalization checks.
