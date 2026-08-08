# Baseline system before end-to-end multimodal work

This document freezes the state inherited from `main` before the Level-4
multimodal research extension begins. It is descriptive: none of the numbers
below were produced by the new research branch.

## Provenance

| field | value |
|---|---|
| baseline branch | `main` |
| baseline commit | `57875634e9c36050ef195d123f81dc391ca7dff8` |
| extension branch | `research/end-to-end-multimodal` |
| corpus | `RealTimeData/bbc_news_alltime`, January 2024 |
| generator | `gpt-4o-mini`, temperature 0, seed 42 |
| judge | `claude-sonnet-5` |

The committed baseline artifacts under `data/processed/`, `data/images/`, and
`results/` must not be overwritten by exploratory runs. New runs use distinct
experiment IDs and output paths.

## Implemented baseline

The repository contains an operational query-driven news summarization pipeline:

```text
BBC articles and images
  -> body cleaning and overlapping sentence-window passages
  -> SBERT text embeddings + CLIP image embeddings
  -> separate FAISS IndexFlatIP indexes
  -> text-only or weighted text/image article retrieval
  -> confidence gate
  -> grounded GPT-4o-mini summary
  -> external claim decomposition and support judgment
```

### Data

- 1,562 raw January 2024 BBC records were considered.
- 1,023 article-image pairs were retained and committed.
- A seeded random split produced 873 retrieval-pool and 150 test articles.
- The committed corpus has 1,023 JPEG files and reports no null IDs or exact
  test/pool headline overlap.
- Embedding produces 11,052 overlapping passages.
- Semantic text near-duplicates and perceptual image near-duplicates were not
  used to form the split.

### Encoders and retrieval

- Text: `all-MiniLM-L6-v2`, 384-dimensional unit-normalized embeddings.
- Image: OpenCLIP `ViT-B-32-quickgelu`, 512-dimensional unit-normalized pixel
  embeddings.
- Text passages are max-pooled to article scores before fusion.
- Scores are filtered by split, min-max normalized per query, then combined as
  `alpha * text + (1 - alpha) * image`.
- B1 is the same code path as multimodal retrieval with `alpha=1.0`.
- There is no BM25, RRF, caption index, or reranker in the baseline.

### Generation arms

| arm | retrieval | generator evidence | causal role |
|---|---|---|---|
| B0 | none | query only | no-retrieval hallucination ceiling |
| B1 | text only | headlines and passages | text-RAG baseline |
| M | text/image score fusion | headlines, passages, captions | baseline multimodal arm |
| M_nocap | same fused retrieval as M | headlines and passages | caption ablation |

Images affect retrieval in M, but image pixels are never sent to the generator.
The baseline is therefore Level 2 multimodality: pixel-informed retrieval with
text-mediated generation.

### Abstention and grounding

- The generator is instructed to use only retrieved evidence.
- A raw-text-similarity gate returns `INSUFFICIENT_EVIDENCE` below `tau=0.35`.
- The model can also return the token itself.
- Evaluation later identified soft prose refusals as a third refusal mode.
- Summaries do not contain evidence citations; the prompt explicitly forbids
  citation markers.

### Evaluation

- Recall@1, Recall@5, and Recall@10 of the withheld gold article.
- Exact McNemar comparison for paired retrieval successes.
- Two-pass external judge: atomic claim decomposition, then evidence support.
- Macro/micro faithfulness and hallucination.
- Paired Wilcoxon tests and 10,000-resample bootstrap confidence intervals.
- Caption ablation, confidence/refusal analysis, and efficiency measurements.
- A 50-claim blind human-validation file exists, but all human labels were blank
  at the branch point. Baseline faithfulness results are therefore provisional.

## Committed baseline outcomes

### Main generation comparison

| system | faithfulness | hallucination |
|---|---:|---:|
| B0 | 0.243 | 0.757 |
| B1 | 0.901 | 0.099 |
| M, alpha=0.5 | 0.886 | 0.114 |

For usable paired items, `M - B1 = -0.015`, bootstrap 95% CI
`[-0.043, +0.013]`, Wilcoxon `p=0.252`. The baseline found no measurable
faithfulness improvement from the committed multimodal configuration.

### Retrieval alpha sweep

| system | Recall@1 | Recall@5 | Recall@10 |
|---|---:|---:|---:|
| B1 | 0.733 | 0.913 | 0.940 |
| M alpha=0.0 | 0.353 | 0.600 | 0.693 |
| M alpha=0.25 | 0.573 | 0.740 | 0.787 |
| M alpha=0.5 | 0.673 | 0.860 | 0.900 |
| M alpha=0.75 | 0.727 | 0.940 | 0.960 |
| M alpha=0.9 | 0.733 | 0.933 | 0.953 |
| M alpha=1.0 | 0.733 | 0.913 | 0.940 |

Pure image retrieval was weak for headline-derived queries. A small image
contribution at alpha 0.75-0.9 improved observed recall, while the committed
alpha 0.5 reduced it. Because the sweep used the existing test set, it is
exploratory and cannot be used as unbiased final model selection.

### Caption ablation

| comparison | difference | 95% CI | p |
|---|---:|---|---:|
| B1 -> M_nocap | +0.0046 | [-0.024, +0.032] | 0.640 |
| M_nocap -> M | -0.0118 | [-0.039, +0.014] | 0.530 |
| B1 -> M | -0.0150 | [-0.043, +0.013] | 0.252 |

Changing the retrieved articles and adding captions were independently null.
This does not test whether a vision-language generator can extract useful facts
from actual image pixels.

### Confidence and refusal

Faithfulness was approximately flat across retrieval-confidence quartiles
(`0.902 / 0.870 / 0.880 / 0.925`), while refusal fell from roughly 72% to 1%.
The evidence supports the mechanism that retrieval confidence mainly changes
answer coverage, not conditional claim faithfulness.

The later evaluator detected approximately 35% true refusals after including
soft prose refusals, rather than the approximately 12% obtained by counting the
formal token alone. Soft refusals can inflate faithfulness through supported
meta-claims about the evidence.

### Efficiency

The committed benchmark reports approximately 19.3 seconds for 11,052 SBERT
passages, 10.2 seconds for 1,023 CLIP images, 21 ms for fused retrieval, and
1.87 GB peak RSS on an M4 MacBook Air. The reported API spend was $7.24.

## Valid conclusions at the branch point

1. Retrieval greatly improves evidence-grounded generation relative to B0.
2. At alpha 0.5, image-fused retrieval plus captions does not measurably improve
   faithfulness over text RAG on ordinary headline-derived BBC queries.
3. The tested image contribution is caption-mediated at generation time.
4. Text retrieval is near a recall ceiling in this regime.
5. Retrieval confidence strongly affects refusal/coverage.
6. Human validation, true image-conditioned generation, evidence coverage,
   duplicate-safe splitting, and conditional visual help/harm testing remain
   necessary before making a broad multimodal conclusion.

