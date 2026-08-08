"""Lightweight article-level TF-IDF retrieval used as a text reranker."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .research_data import articles


@lru_cache(maxsize=1)
def _state():
    frame = articles().reset_index(drop=True)
    # Captions are deliberately excluded: this is the strong text arm, not an
    # indirect visual condition. Headlines plus bodies are legitimate text.
    docs = (frame.headline.fillna("") + " " + frame.body.fillna("")).tolist()
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(docs)
    return frame[["id"]], vectorizer, matrix


def scores(query: str, candidate_ids: set[str]) -> pd.DataFrame:
    """Cosine TF-IDF scores in canonical article order for an explicit set."""
    frame, vectorizer, matrix = _state()
    mask = frame.id.isin(candidate_ids).to_numpy()
    idx = np.flatnonzero(mask)
    values = (matrix[idx] @ vectorizer.transform([query]).T).toarray().ravel()
    return pd.DataFrame({"id": frame.iloc[idx].id.to_numpy(), "s_lex": values})

