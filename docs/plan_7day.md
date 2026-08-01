# Multimodal RAG in 7 Days — Realistic Scope & Plan

## Bottom line

Yes, the *full* proposal is too much for 7 days solo — but not because coding is slow. With AI, writing the code is the fast part. What eats days is **data plumbing, environment/dimension debugging, and the evaluation harness**. So the winning move is to attack those three, not the code.

An "okish demo" is very achievable — roughly **3–4 focused days** of real work plus buffer. To get there, make one non-negotiable cut and one priority call:

- **Cut the 50k NYT scrape.** GoodNews only ships article URLs + a download script, and NYT image links rot — downloading tens of thousands of pairs can eat 2–3 days by itself and often half-fails. Use a **tiny corpus (~1–2k pairs) where images are already bundled**, or scrape only a few hundred that download cleanly. 1–2k is plenty to *demonstrate* the effect; you don't need scale to show text-vs-multimodal differs.
- **The demo is the deliverable, not the eval.** Prioritize a working Streamlit toggle (text-only vs multimodal) over a rigorous factuality study. A live demo + a small honest results table beats a half-finished harness.

Keep the ONE comparison that is the whole point — **text-only RAG vs multimodal RAG** — and drop almost everything else. Details below.

---

## What to keep vs cut (MoSCoW)

| | Item | Why |
|---|---|---|
| **MUST** | Tiny corpus (~1–2k) with bundled images | Removes the #1 time-sink |
| **MUST** | SBERT + CLIP embeddings, FAISS index | Core of retrieval |
| **MUST** | Text-only + fused multimodal retrieval | The experimental variable |
| **MUST** | GPT-4o-mini generation w/ grounding prompt | The summarizer |
| **MUST** | Streamlit demo with text-vs-multimodal toggle | The resume artifact |
| **SHOULD** | Small faithfulness eval (LLM-judge) + recall@k on ~100–200 items | Turns demo into a result |
| **SHOULD** | Retrieval-confidence abstention (one threshold) | Cheap, looks rigorous |
| **SHOULD** | Deploy to HF Spaces | Shareable link for resume |
| **WON'T (this week)** | 50k scale, Visual News OOD set, human-annotation subset, full ablation sweeps, local Llama, MinHash dedup | Each is a day+; none needed for an okish demo. List them as "future work." |

Reporting the cut items as explicit *future work* in your writeup is not a weakness — it reads as scoping judgment.

---

## Honest answer: "will this take super long even with AI?"

The parts AI makes genuinely fast: writing `embed.py`, `index.py`, `retrieve.py`, the prompt template, the Streamlit app. Hours, not days.

The parts that stay slow even with AI, and where your time actually goes:
1. **Getting data onto disk in a usable shape** — downloads, broken image links, matching images to articles. Budget a whole day; it's the #1 killer.
2. **Dimension / dtype / normalization bugs** — CLIP is 512-d, MiniLM is 384-d, FAISS wants float32 normalized. Mismatches cause silent garbage retrieval. Budget half a day of debugging.
3. **Eval harness** — parsing claims, LLM-judge calls, aggregating. Keep it minimal or it expands to fill all remaining time.

So: not "super long," but only if you ruthlessly scope. Unscoped, it's a 3-week project.

---

## The 7-day plan

### Day 1 — Environment + data (the make-or-break day)
1. Repo + venv + `pip install sentence-transformers faiss-cpu open_clip_torch pillow openai streamlit pandas rouge-score tqdm python-dotenv`. Put `OPENAI_API_KEY` in `.env`.
2. **Get a small multimodal news corpus with images already available.** We use `RealTimeData/bbc_news_alltime` (see `docs/day1_guide.md` + `src/day1_build_dataset.py`).
3. Land a single `data.parquet`: `id, headline, body, image_path, caption, section`.
4. Split off ~150 rows as `test`; the rest is the index pool. (Skip MinHash — with a tiny controlled corpus, a quick title-dedup check is enough.)

*If Day 1 slips, everything slips. Timebox data acquisition to one day and take whatever clean subset you have.*

### Day 2 — Embeddings + FAISS + retrieval
1. Chunk articles into ~3–5 sentence passages; embed with `all-MiniLM-L6-v2` (384-d), `normalize_embeddings=True`.
2. Embed images (and captions) with **CLIP ViT-B/32** (512-d).
3. Build two `faiss.IndexFlatIP` indexes (text passages, images). At 1–2k this builds in seconds.
4. `retrieve(query, mode, k, alpha)`:
   - text mode → search text index.
   - multimodal → also embed query with CLIP text tower, search image index, **min-max normalize each score stream**, combine `alpha*s_text + (1-alpha)*s_img`. Default `alpha=0.5`.
5. Eyeball 5 queries; confirm images look relevant.

### Day 3 — Generation, end-to-end
1. `generate(prompt)` wrapping GPT-4o-mini (`temperature=0`).
2. Grounding prompt: *"Summarize ONLY the numbered evidence. Every claim must be supported. If evidence is insufficient, reply INSUFFICIENT_EVIDENCE."* In multimodal mode, inject image captions as `[IMAGE k: "caption"]`.
3. Add abstention: if top retrieval score < τ, return INSUFFICIENT_EVIDENCE.
4. Wire configs B1 (text) and M (multimodal) so one call produces both for a query. Cache LLM outputs by prompt hash to avoid re-billing.
5. Get one query working fully, text and multimodal.

### Day 4 — Streamlit demo (priority deliverable)
1. Input box → show retrieved passages, retrieved images side-by-side, and the summary.
2. **Toggle: text-only vs multimodal.** This single screen *is* your project.
3. Show the abstention state when confidence is low.
4. Get it running locally and looking clean. Screenshot it — that goes in your report/resume even if deploy breaks.

### Day 5 — Minimal evaluation
1. **Faithfulness (LLM-judge):** split each summary into claims, ask GPT-4o-mini "supported by evidence? yes/no" per claim; faithfulness = supported/total, hallucination = 1 − that. (Use RAGAS if it installs cleanly; a 30-line hand-rolled judge is a fine fallback and less dependency risk.)
2. **recall@k** on your ~150 test items (you know the gold article): text vs multimodal at k=5.
3. One results table: B1 vs M on faithfulness, hallucination, recall@5. One bar chart. That's enough for an okish demo.

### Day 6 — Deploy + README (+ buffer)
1. Push demo to **Hugging Face Spaces** (Streamlit SDK, free tier); ship the small FAISS index + corpus with it.
2. README: pipeline diagram, headline result, run instructions, live link. This is the resume asset — make it skimmable in 30 seconds.
3. Reserve the afternoon for the deploy breakage that always happens.

### Day 7 — Report + polish (+ buffer)
1. Short report: hypothesis → data → method → the one result table → honest limitations → future work (list the cut items here).
2. Resume bullets (below).
3. Final buffer for anything that slipped.

---

## If you fall behind (fallback ladder)

Drop in this order and you still have something presentable:
1. Drop HF deploy → demo runs locally, show a screen recording.
2. Drop the faithfulness eval → keep recall@k + qualitative side-by-side examples.
3. Drop abstention.
4. Shrink corpus to ~500 → still demonstrates the toggle.

The floor you should defend: **a working local Streamlit demo that retrieves and summarizes, with a text-vs-multimodal toggle, on a few hundred real news pairs.** That alone is a legitimate project and a real resume line.

---

## Resume bullets (achievable in this scope)

- Built and deployed a multimodal retrieval-augmented news summarizer (SBERT + CLIP ViT-B/32 + FAISS + GPT-4o-mini) with an interactive Streamlit demo comparing text-only vs image-fused retrieval.
- Implemented cross-modal score fusion and retrieval-confidence-gated abstention to reduce ungrounded output; evaluated claim-level faithfulness and recall@k on a held-out test set.

---

## Sources
- [Good News, Everyone! (paper)](https://arxiv.org/pdf/1904.01475) — confirms dataset ships URLs + a download script, not images.
- [Transform and Tell: Entity-Aware News Image Captioning](https://arxiv.org/pdf/2004.08070) — GoodNews split details.
- [MiRAGeNews (HF paper page)](https://huggingface.co/papers/2410.09045) — example of a smaller (~12.5k) news image–caption set already hosted.
- [multi_news (HF dataset)](https://huggingface.co/datasets/alexfabbri/multi_news) — text-only news summarization fallback if image plumbing fails.

*Verify dataset access and library versions on Day 1 — availability and APIs drift.*
