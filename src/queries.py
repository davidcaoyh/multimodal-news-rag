"""
Day 3 support: one query string per test article.

    python -m src.queries                 # build (paraphrase mode), write parquet
    python -m src.queries --mode headline # zero-cost fallback
    python -m src.queries --show 10

Writes data/processed/queries.parquet
    test_id, headline, query, query_qa, mode
(data/processed/ is COMMITTED — every teammate must generate and evaluate on the
identical 150 queries or B1-vs-M numbers are not comparable across clones.)

Why this file exists at all
---------------------------
Both obvious choices for "what is the query?" are rigged (day2_guide.md:88):

    headline verbatim -> rigged toward B1. Headline tokens recur through the body,
                         SBERT saturates, and the text arm gets a free win.
    caption field     -> rigged toward M. It describes the image the M arm searches.

Neutral option, used here: LLM-rewrite the headline. The headline is the one field
indexed in NEITHER stream (passages come from `body`, image vectors from the pixels
— D3), so both arms bridge the same gap. Both prompts below forbid adding facts, so
a query cannot smuggle in detail the retriever would otherwise have to find.

TWO columns, because generation and recall@k want opposite things
------------------------------------------------------------------
`query_qa` — the pinpoint question ("Why was X jailed for planning a shooting?").
    Ideal for recall@k: it names the gold article's subject, so "did the retriever
    find it?" is a sharp question.

`query` — the broader topic phrase ("man jailed over a planned mass shooting").
    What generation uses.

The first attempt used the pinpoint question for BOTH and it failed measurably:
gpt-4o-mini returned INSUFFICIENT_EVIDENCE on ~50% of B1 and M items, collapsing
the paired comparison to 74/150. Not a retrieval bug — retrieval returned visibly
on-topic articles at s_text up to 0.842. D1 withholds the source article from the
generation index, so a question naming that article's unique subject (a specific
person, a specific court case) asks for a fact the pool provably does not contain.
The model was right to refuse.

Broadening drops the unique name and keeps the story type, so pool coverage is
genuinely about the query and the summarization task is well-posed. The evidence is
still adjacent-but-not-identical to the withheld article, which is the point: that
gap is exactly where hallucination shows up, and it is where B1 and M can differ.

`mode="headline"` stays available as the documented zero-cost fallback.
"""
import argparse

import pandas as pd

from .generate import generate

TEST = "data/processed/test.parquet"
OUT = "data/processed/queries.parquet"

QUESTION_PROMPT = """\
Rewrite this news headline as the natural question a reader would type into a news \
search box.

Rules:
- Use only information already in the headline. Add no names, numbers, or facts.
- Do not answer the question or add context.
- One sentence, under 20 words.
- Output only the question, nothing else.

Headline: {headline}
Question:"""

# The broadening rules are all SUBTRACTIVE — drop the unique identifier, keep the
# story type. Nothing is added, so this cannot smuggle in information the retriever
# would otherwise have to find, and it stays neutral between the two arms.
TOPIC_PROMPT = """\
Rewrite this news headline as a short topic phrase, of the kind someone would type \
into a news search box to find this story AND other coverage like it.

Rules:
- Use only information already in the headline. Add no names, numbers, or facts.
- KEEP public figures, politicians, companies, institutions and place names. These are
  the subject and must survive.
- Drop only the name of a private individual who is in the news for this one incident
  (a defendant, a victim, a local resident), replacing it with what they are.
- Generalise the single specific detail the story reveals; keep everything else.
- A noun phrase, not a question. No question mark.
- 4 to 10 words. Output only the phrase, nothing else.

Headline: Reed Wischhusen jailed for planning 'revenge' mass shooting - BBC News
Topic: man jailed over a planned revenge mass shooting

Headline: Trump challenges his 'arbitrary' removal from Maine's ballot - BBC News
Topic: Trump removal from the Maine ballot

Headline: No investigation into Prince Andrew - Met Police - BBC News
Topic: Met Police investigation into Prince Andrew

Headline: {headline}
Topic:"""


def _rewrite(prompt: str, headline: str, floor: int = 3) -> str:
    """One cached rewrite. Never returns empty — falls back to the headline."""
    q = generate(prompt.format(headline=headline), max_tokens=60).strip()
    q = q.strip('"').strip().rstrip(".")
    # a refusal or an empty completion must not become a silent empty query
    return q if len(q.split()) >= floor else headline


def question(headline: str) -> str:
    """Headline -> pinpoint question. Used by Day 5 recall@k."""
    return _rewrite(QUESTION_PROMPT, headline)


def topic(headline: str) -> str:
    """Headline -> broader topic phrase. Used by generation."""
    return _rewrite(TOPIC_PROMPT, headline)


def build(mode: str = "topic") -> pd.DataFrame:
    test = pd.read_parquet(TEST)[["id", "headline"]]
    if mode == "headline":
        return pd.DataFrame({
            "test_id": test["id"], "headline": test["headline"],
            "query": test["headline"], "query_qa": test["headline"], "mode": mode,
        })
    if mode != "topic":
        raise ValueError(f"mode must be 'topic' or 'headline', got {mode!r}")

    topics, questions = [], []
    for i, h in enumerate(test["headline"], 1):
        topics.append(topic(h))
        questions.append(question(h))
        print(f"  rewrote {i}/{len(test)}", end="\r")
    print()

    return pd.DataFrame({
        "test_id": test["id"],
        "headline": test["headline"],
        "query": topics,
        "query_qa": questions,
        "mode": mode,
    })


def load() -> pd.DataFrame:
    """Read the frozen query set. Raises if it has not been built."""
    try:
        return pd.read_parquet(OUT)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{OUT} not found — build it once with `python -m src.queries`. "
            "It is committed so every teammate evaluates on identical queries."
        ) from None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="topic", choices=["topic", "headline"])
    ap.add_argument("--show", type=int, default=5, help="print N examples")
    ap.add_argument("--force", action="store_true", help="rebuild even if OUT exists")
    args = ap.parse_args()

    import os

    if os.path.exists(OUT) and not args.force:
        df = load()
        print(f"{OUT} already exists ({len(df)} rows, mode={df['mode'].iloc[0]}). "
              "Use --force to rebuild.")
    else:
        df = build(args.mode)
        df.to_parquet(OUT, index=False)
        print(f"wrote {OUT}  ({len(df)} rows, mode={args.mode})")

    dup = df["query"].duplicated().sum()
    empty = (df["query"].str.strip() == "").sum()
    print(f"{'PASS' if empty == 0 else 'FAIL'}  no empty queries")
    print(f"{'PASS' if dup == 0 else 'WARN'}  {dup} duplicate query strings")
    print(f"      median query length: {df['query'].str.split().str.len().median():.0f} words")

    print()
    for r in df.head(args.show).itertuples():
        print(f"  {r.headline}\n   query    (generation): {r.query}"
              f"\n   query_qa (recall@k):   {r.query_qa}\n")


if __name__ == "__main__":
    main()
