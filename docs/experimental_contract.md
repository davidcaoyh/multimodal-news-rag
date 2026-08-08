# Experimental contract for the research extension

This contract separates exploration from final evidence. It prevents the final
architecture from being selected on the same examples used to report its result.

## Research questions

1. Does retrieval reduce unsupported claims relative to no retrieval?
2. Do actual image pixels improve evidence coverage or factual generation beyond
   text retrieval and caption-mediated retrieval?
3. Under which visually informative conditions do images help?
4. When do misleading or conflicting images harm grounding?

## System ladder

| system | retrieval | generator evidence | interpretation |
|---|---|---|---|
| B0 | none | query | hallucination ceiling |
| B1 | strongest validated text retriever | text | text-RAG baseline |
| M_nocap | validated text/image retrieval | text | image retrieval contribution |
| M_caption | same as M_nocap | text + captions | caption contribution |
| M_vision | same as M_caption | text + image pixels | pixel contribution |
| M_conflict | same text as M_vision, controlled wrong image | text + wrong pixels | visual misgrounding risk |

Each adjacent comparison changes one intended factor. Models, decoding, query,
retrieval depth, and text evidence stay fixed unless the experiment explicitly
states otherwise.

## Data roles

The inherited 150-item test set has already been used for alpha exploration and
is therefore a historical/exploratory benchmark, not an untouched final test.

Before final model selection:

1. Audit exact and near-duplicate text and images across the committed split.
2. Construct grouped splits so duplicate/near-duplicate clusters cannot cross.
3. Reserve explicit development, validation, and final-test roles.
4. Generate and freeze queries separately for each role.
5. Select alpha, k, fusion, reranking, and prompt changes on development and
   validation data only.
6. Freeze the final configuration before executing the final test.

If the small corpus cannot support three statistically useful partitions, use
development plus final test, with cross-validation inside development. The final
test still remains untouched.

## Metrics

### Primary

- Query-evidence coverage/relevance.
- Claim faithfulness.
- Hallucination rate, separating unsupported and contradicted claims.
- Paired difference with 95% bootstrap confidence interval.

### Multimodal mechanism

- Claim support: text only, image only, both, unsupported, contradicted.
- Visual Contribution Rate.
- Change in coverage and faithfulness from M_caption to M_vision.
- Change under correct versus misleading images.

### Secondary

- Recall@1/5/10, MRR, and optionally nDCG.
- Answer coverage, hard/soft refusal, and abstention correctness.
- Retrieval/generation latency, memory, tokens, and API cost.

## Human validation

Automated judge outputs are provisional until compared with blind human labels.
The existing 50-claim sample is a minimum validation set. Human annotators must
not see the system label or automated verdict before labeling. Report agreement,
Cohen's kappa, confusion counts, and disagreement categories.

An AI assistant's labels do not count as human validation.

## Selection and reporting rules

- Do not choose an architecture because it has the best final-test number.
- Record every attempted configuration in `docs/research_log.md`.
- A component enters the final system only if it improves a primary metric,
  reveals a robust mechanism, improves safety, or simplifies the system without
  material loss.
- Null and negative results remain in the ledger.
- Invalid runs remain documented but are excluded from aggregate claims.
- The report presents the smallest system ladder needed to support the conclusion.

