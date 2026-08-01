# Multimodal RAG for Factual News Summarization — Full 2-Week Plan (reference)

> This is the original, ambitious version. The active plan for this project is `docs/plan_7day.md`.
> Keep this as the source for "future work" items and the rigorous version if scope expands.

**Goal:** Ship a working system that answers one question: *does adding image retrieval to text RAG reduce hallucination in news summaries?* Deliver (1) a deployed Streamlit demo and (2) a reproducible factuality-evaluation harness.

---

## 0. Mental model

```
query ──► RETRIEVE evidence ──► BUILD prompt ──► LLM ──► summary
                  │
        switch: [text-only]  vs  [text + image]
```

Three configs: **B0** LLM alone (no retrieval), **B1** text-only RAG (baseline), **M** multimodal RAG (method). Everything serves a fair B1-vs-M comparison.

**Model:** GPT-4o-mini via API behind a single `generate(prompt)` seam so a local Llama-3-8B can be swapped in later.

## 1. Setup
Repo scaffold: `src/{data_prep,embed,index,retrieve,generate,evaluate,config}.py`, `app/streamlit_app.py`, `notebooks/`, `results/`. Fixed `SEED=42`. `OPENAI_API_KEY` in `.env`.

## 2. Data (Days 1–2)
GoodNews (~50k stratified subset) + Visual News OOD subset. Store `id, headline, body, image_path, caption, section, story_type`. Split index_pool/test BEFORE indexing. **Near-duplicate filtering** with MinHash+LSH (datasketch) to prevent evidence leakage.

## 3. Retrieval & indexing (Days 3–5)
Passage chunking → SBERT text embeddings; CLIP ViT-B/32 image + caption embeddings; FAISS `IndexFlatIP`. Text-only retrieval (B1) and **late-fusion** multimodal retrieval `score = α·s_text + (1−α)·s_img` (M), per-query score normalization.

## 4. Generation & guardrails (Days 6–7)
Grounding prompt (summarize only numbered evidence; abstain with INSUFFICIENT_EVIDENCE). Captions carry visual evidence into the text prompt. Retrieval-confidence abstention gate. `temperature=0`. Cache by prompt hash.

## 5. Evaluation harness (Days 8–10)
Claim-level faithfulness (RAGAS) → hallucination rate. recall@k. ROUGE-L (secondary). **Human-annotated subset** (~50) to validate automatic scores (Cohen's κ). One tidy `results/metrics.csv`.

## 6. Experiments & analysis (Days 11–12)
Main hallucination bar chart (B0/B1/M). **Hypothesis test:** faithfulness gain (M−B1) stratified by story_type. Ablations: top-k, fusion weight α, caption availability. Domain shift NYT vs Visual News. Qualitative failure cases (misleading images).

## 7. Demo (Day 13)
Streamlit: query → retrieved passages + images + summary + per-claim support highlighting + text/multimodal toggle. Deploy to HF Spaces.

## 8. Report & README (Day 14)
Report + resume-ready README + reproducibility check.

## Future-work items (cut from the 7-day version)
50k scale, Visual News OOD, human-annotation subset, full ablation sweeps, local Llama-3-8B, MinHash dedup, early-fusion retrieval, per-claim highlighting in the demo.

## Key references
GoodNews (Biten et al. 2019); Visual News (Liu et al.); CLIP (Radford et al. 2021) / OpenCLIP; Sentence-Transformers; FAISS; RAGAS (Es et al. 2023); rouge-score; datasketch; HF Spaces Streamlit.
