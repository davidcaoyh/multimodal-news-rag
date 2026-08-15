# Improvement areas & future work

Ranked by information gained per unit of cost and effort, with concrete implementation
notes. Written 2026-08-02 after the Day 5 evaluation.

Findings this plan responds to are in [evaluation_report.md](evaluation_report.md); design
rationale is in [decisions.md](decisions.md).

**Budget context:** everything in Tiers 0–1 fits in ~$15 of API spend. Tier 2 is where cost
grows, and only 2.1 exceeds $10.

---

## 0. What the current results say about where to spend effort

Three measured facts should drive prioritisation, because each one closes off an obvious
direction:

| finding | consequence |
|---|---|
| Faithfulness is **flat** against retrieval confidence (ρ=+0.15 / +0.06, n.s.) while refusal runs **72% → 1%** across quartiles | Improving retrieval will cut refusals, **not** raise faithfulness. Do not expect retrieval work to move the headline metric. |
| Both multimodal channels are **independently null** — retrieval p=0.64, captions p=0.53 | The null is not a tuning problem. Changing α or the fusion formula is unlikely to help. |
| Detecting the observed 1.5-point effect needs **636 paired items**; a 64× bigger index buys only 4.0 → 3.4 points | Power comes from the **test set**, not the index. Scaling the pool alone does not resolve the research question. |

The two directions that remain genuinely open are **(a) the model never sees pixels** and
**(b) text retrieval is already near-saturated, so images have no headroom**. Tier 1
attacks both.

---

## Tier 0 — nearly free, do first

### 0.1 Validate the judge  ·  $0  ·  ~30 min

**Every faithfulness number is provisional until this is done.** If judge agreement is
poor, the central null result is a statement about the judge, not the system. Not yet
attempted with a fluent human rater — this needs a small export/score script (blind
sample of ~50 claims, `config` and the judge's verdict withheld, hand-labeled `y`/`n`,
then scored for agreement and Cohen's κ against the judge). `src/final_validation.py`'s
`export()`/`validate()` pair is a template for the shape of this, though it currently
serves the separate AI-assisted secondary check rather than a human rater.

Seeing the judge's answer before writing your own is confirmation, not validation.
Report agreement and κ; >0.6 is substantial, >0.8 near-perfect.

### 0.2 Second-judge robustness  ·  ~$2.70  ·  ~10 min

Re-judge the same 5,146 claims with `claude-haiku-4-5` and report inter-judge agreement.
Turns "we trusted one judge" into a measured claim, and hedges the risk that 0.1 comes back
weak. Implementation: `JUDGE_MODEL` is a module constant in `evaluate.py`; the prompt-hash
cache key already includes the model, so the two runs cannot collide.

Note Haiku 4.5 still accepts `temperature`, so it can run at `temperature=0` — which also
gives a free check on whether the D13 determinism workaround (disabled thinking + schema)
behaves like true greedy decoding.

### 0.3 Hallucination error taxonomy  ·  $0–1  ·  ~2 hours

The ~10% unsupported claims already carry written judge rationales in `results/claims.csv`.
Cluster them by failure type — entity substitution, date/number error, unsupported causal
link, world-knowledge intrusion, over-generalisation. A table of *what kind* of
hallucination survives retrieval is a strong qualitative section and costs only reading
time. Optionally have an LLM do a first-pass clustering, then verify by hand.

```python
c = pd.read_csv("results/claims.csv")
c[~c.supported].reason.value_counts()       # start here
```

---

## Tier 1 — the experiments that would most change the conclusion

### 1.1 A genuinely multimodal generator  ·  ~$3–5  ·  ~1 day  ·  **highest priority**

**The gap:** the current "multimodal" arm never shows the model an image. CLIP embeddings
influence *retrieval*, and human-written captions are injected as *text* — and the ablation
(D16) proved the caption channel is inert. So what has been falsified is *caption-mediated*
multimodality, not multimodality as such.

**The experiment:** a fourth arm, `M_vision` — identical retrieval to M, but the top-k
**images passed as image content blocks** instead of caption text.

- `gpt-4o-mini` is vision-capable, so the generator family stays constant and the
  comparison remains valid.
- Sits naturally alongside the existing ablation:
  `B1 → M_nocap → M → M_vision` walks from no visual signal, to visual signal in
  retrieval only, to captions, to actual pixels.
- Implementation: `format_evidence()` already branches on `with_images`; add a third mode
  that returns structured content blocks rather than a string. `generate()` needs to accept
  a message-content list instead of a bare string.
- Watch the cost: images at `detail: low` are ~85 tokens each (5 images ≈ 425 tokens);
  `detail: high` is ~2,800 each and would dominate the bill. Start low.

**Why it matters:** this is the single experiment that could still overturn the null. A
positive result rescues the hypothesis; a null makes "multimodal retrieval does not improve
faithfulness in news summarisation" a much broader and better-supported claim.

### 1.2 A retrieval regime where text is weak  ·  ~$5  ·  ~2–3 days

**The gap:** B1 already achieves recall@5 = 0.913. There is ≤8.7 points of headroom, so
images have almost nothing to add — the null may simply be a **ceiling effect** rather than
a fact about multimodality.

**The experiment:** construct conditions where text retrieval should struggle, and see
whether fusion rescues them.

| variant | construction | expectation if the hypothesis is right |
|---|---|---|
| **visual-first queries** | derive the query from *image content* ("aerial photo of flooded farmland") rather than the headline topic | CLIP should outperform SBERT; fusion should beat text-only |
| **degraded text** | truncate passages to their first sentence, or inject noise | as text signal falls, the optimal α should shift toward the image stream |
| **caption-free corpus** | strip captions from the index | isolates pure pixel-vs-text retrieval |

Report faithfulness *and* recall as a function of text-signal strength. This converts a flat
null into a **conditional** finding — "multimodality helps when text retrieval is weak,
which on ordinary news queries it is not" — which is a considerably more interesting result
than either a bare null or a bare positive.

Implementation note: `queries.py` already refuses to overwrite `queries.parquet` without
`--force`. Write new query variants to a *separate* file rather than regenerating the frozen
one.

### 1.3 An evidence-coverage metric  ·  ~$1–2  ·  ~half a day

**The gap:** faithfulness grades each arm against *its own* evidence, so it structurally
cannot detect better retrieval. A system that retrieves irrelevant articles and faithfully
summarises them still scores ~0.9. The yardstick moves with the arm.

**The metric:** judge `(query, evidence) → does this evidence contain what is needed to
address the query?` Arm-comparable, sensitive to retrieval quality, and needs no reference
summary — so it avoids the leakage problem that killed ROUGE-L (D5).

Pairs naturally with what exists:

```
coverage      did we retrieve the right thing?     ← currently unmeasured
faithfulness  did we stay grounded in it?          ← measured, ~0.90
```

Reuses the entire judge harness — same two-pass structure, same caching, same blindness
enforcement. Roughly 260 additional calls with short prompts.

---

## Tier 2 — scale and retrieval quality

### 2.1 Enlarge the test set to reach real statistical power  ·  ~$25–35  ·  ~2–3 days

The bottleneck is n=90, not index size. `RealTimeData/bbc_news_alltime` exposes many
months; using ~6 months gives roughly 6,000 articles → ~1,000 test items → ~600 usable
pairs, which is the 636 needed to resolve an effect the size actually observed.

This is the **only** intervention that fixes both the refusal rate (bigger pool) and the
power problem (bigger test set) at once.

Costs and caveats:

- Judge cost scales with test items: ~600 pairs × 3 arms ≈ **$25–35**, plus ~$1 generation.
- Most of the wall-clock is image downloading at ~0.34 s each, serial — **parallelise it**
  before starting; 6,000 images is ~35 min serial and a few minutes with a pool.
- **This invalidates the committed corpus and every number in the report.** It is a
  "re-run everything properly" option, not an increment. Do it only if the Tier 1 results
  justify the rebuild.

### 2.2 Hybrid sparse + dense retrieval  ·  $0 API + ~$7 re-run  ·  ~1 day

BM25 + SBERT fusion is a standard strong baseline. Expected effect: higher `s_text`, which
per §0 cuts the 35% refusal rate and increases usable n — without raising faithfulness.

It also makes B1 a **harder** baseline, which is the honest thing to do before claiming
anything about M. A multimodal method that only beats a weak text baseline has not shown
much.

`rank_bm25` is pure Python and needs no new heavy dependency.

### 2.3 Cross-encoder re-ranking  ·  $0 API  ·  ~half a day

Re-rank the top-50 fused candidates with a small cross-encoder (e.g.
`cross-encoder/ms-marco-MiniLM-L-6-v2`) before taking the top-5. Typically a large recall
gain for modest latency, and it runs locally. Same caveat as 2.2: expect it to move
refusals, not faithfulness.

---

## Tier 3 — completeness, low priority given the findings

### 3.1 Faithfulness at α = 0.75 and 0.9  ·  ~$3.60  ·  ~15 min

Closes the "untested at the retrieval-optimal α" gap. Given faithfulness is flat against
retrieval quality, **expect a null**. Only M needs regenerating and re-judging — B0 and B1
do not depend on α and are cached.

If you do run it: selecting α on the same test set you report on is **tuning on test** and
must be declared as such (D11).

### 3.2 Story-type stratification  ·  $0  ·  ~1 hour

Hand-label the 150 test items `event_centric` / `abstract_topical` and report the
comparison per stratum — the hypothesis being that images help more for event-centric
stories. Do **not** use `story_type` from `day1_split.py`; BBC's `section` is geographic and
68% "other" (D6). n≈50/bucket is underpowered, so report as exploratory.

### 3.3 Deploy to HF Spaces  ·  $0  ·  ~half a day

Day 6 of the original plan. Ship the FAISS index and corpus with the Space. Not a research
contribution, but it is the shareable artefact.

---

## Suggested two-week schedule

| days | work |
|---|---|
| 1 | Tier 0 in full — judge validation, second judge, error taxonomy |
| 2–3 | 1.3 coverage metric, then 1.1 vision-capable generator arm |
| 4–6 | 1.2 weak-text retrieval regime — the headline new experiment |
| 7–8 | 2.2 hybrid retrieval + 2.3 re-ranking; re-run the full evaluation |
| 9–11 | 2.1 corpus/test-set scale-up, **if** Tier 1 justifies the rebuild |
| 12–14 | Write-up, plots, deploy, buffer |

**Compressed minimum (3 days, ~$8):** Tier 0 + 1.1 + 1.3. That upgrades the result from
"no effect observed" to "no effect observed, mechanism identified, obvious confound
eliminated" — which is a defensible negative result rather than an inconclusive one.

---

## Explicitly rejected

Recorded so they are not re-proposed.

| idea | why not |
|---|---|
| **ROUGE-L / reference summaries** | The reference must come from the test article the split withholds, so the highest-scoring system is one that *leaks*. Rejected on structure, not cost (D5). |
| **Scaling the index alone for statistical power** | Measured: 64× the pool moves the detectable effect only 4.0 → 3.4 points. Power comes from the test set (§0). |
| **Tuning α to find a significant result** | Selecting on the reported test set. If done, must be declared as tuning on test (D11). |
| **Any B1-vs-M metric computed on `s_text` or `s_img` alone** | Won by B1 by construction — M's mean `s_text` is lower on 145/150 queries and higher on 0 (D12). |
| **Re-splitting the corpus to fix `story_type`** | Changes the committed parquets, breaks every clone's comparability, and cannot rescue `sport` (n=1 corpus-wide) (D6). |
| **Widening `bucket()`** | `section` is geographic and unfixable. Hand-label instead (D6). |
| **Weakening the generation prompt to cut refusals** | Refusal is the desired failure mode; a more permissive prompt makes the model invent to fill gaps — the exact failure this project measures (D10). |
