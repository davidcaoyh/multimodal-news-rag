# Final research results

## Question and outcome

This project asks whether actual visual evidence improves grounded news
summarization beyond a strong text RAG system and a caption-mediated multimodal
system. It now implements end-to-end pixel multimodality: CLIP image embeddings
affect retrieval, and the vision-capable generator and evaluator receive the
retrieved JPEGs themselves.

The answer is conditional rather than a universal win. Actual pixels produced a
large improvement on a deliberately image-sensitive development sample, but only
a small, statistically uncertain gain on the frozen held-out sample. They help
some individual cases and can hurt under image conflict.

## Experimental validity

- Corpus: 1,023 BBC News articles from January 2024, each with text and an image.
- Duplicate audit: 102 candidate duplicate/near-duplicate edges; 20 crossed the
  inherited random split.
- Group-safe roles: 707 retrieval pool, 166 development, 150 final test, with zero
  audited duplicate edges crossing roles.
- Frozen final sample: 20 articles, four from each of five declared section families.
- No training or fine-tuning: SBERT and CLIP are frozen encoders; generation and
  judging are hosted inference.
- M and M_vision use identical retrieval, passages, headlines, captions, ordering,
  prompt, model, and decoding. M_vision adds only the corresponding image pixels.

## Retrieval results

Development selected alpha=0.75 (75% dense text, 25% image): Recall@5 0.940
versus 0.927 for dense text, with three visual rescues and one harm. Equal fusion,
pure image, and RRF were rejected. A strong word/bigram TF-IDF baseline reached
Recall@5 1.00, revealing substantial lexical cues in the inherited questions.

On the frozen 20-query final sample, Recall@5 was 1.00 for dense text, selected
fusion, and lexical retrieval; image-only retrieval was 0.55. Thus final fusion
tied rather than improved on text retrieval.

## Generation and grounding

| frozen final system | usable faithfulness | usable coverage |
|---|---:|---:|
| B1 text RAG | 0.861 | 13/20 |
| M fused retrieval + captions | 0.834 | 12/20 |
| M_vision M + pixels | 0.858 | 13/20 |

On 12 jointly usable final cases:

- M − B1 = −0.0405, paired bootstrap 95% CI [−0.0892,+0.0076].
- M_vision − M = +0.0366, CI [−0.0155,+0.0888].
- 4.0% of supported M_vision claims received both text and pixel support; no
  supported claim required pixels alone.

The prespecified category cuts contain only one to four paired cases each, so
they are descriptive. Pixel gains were largest in two business/technology/science
cases and one sports case; society/culture was negative.

## Mechanism and stress findings

The targeted 12-case development diagnostic produced M_vision − M = +0.205 on
nine jointly usable cases, CI [+0.057,+0.427], while M − B1 remained null. This
sample deliberately concentrated visual rescues, harms, and strong rank shifts,
so it identifies possible benefit conditions but is not a population estimate.

For the five cases with pixel-supported claims, rotating another case's images
while keeping text fixed reduced clean-evidence faithfulness from 0.983 to 0.927:
mean −0.057, CI [−0.123,0.000]. Two cases degraded and three were unchanged.

## Null, negative, and invalid experiments retained

- Caption addition and caption-mediated fusion did not improve faithfulness.
- RRF and equal text/image weighting underperformed restrained fusion.
- TF-IDF exposed a lexical ceiling rather than a multimodal advantage.
- No model training occurred during any experiment.

## Efficiency

- Embedding build: about 40 seconds with two CPU threads.
- Median final retrieval: 35 ms/query after loading.
- Index plus embedding artifacts: about 38 MB.
- Research-extension API ledger: $0.208 across 214 successful calls, under the
  code-enforced $8 ceiling and $10 prepaid balance.
- Generation latency was not systematically instrumented and is not reconstructed
  after the fact.

## Validation status

An independent AI-assisted review completed all 50 blind rows: 39 supported,
11 unsupported, and three supported by both text and pixels. Agreement with the
frozen judge was 88%, Cohen's kappa 0.672, and modality agreement 86%. This is a
secondary automated consistency check, not literal human validation.

For a human-validation claim, first create a fresh copy of
`results/experiments/E12_final_comparison/human_validation_50.csv` with the three
label fields cleared. A person must independently label that blind copy without
seeing the AI-assisted values; after replacing the reference labels, run:

```bash
python -m src.final_validation --validate
```

This reports agreement, Cohen's kappa, and a confusion matrix. Until human review
is completed, judge-derived faithfulness values remain provisional.
