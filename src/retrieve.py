"""
Day 2, step 3: query -> top-k articles, text-only or text+image fused.

    python -m src.retrieve                      # 5 eyeball queries, both modes

    from src.retrieve import retrieve           # from repo root
    hits = retrieve("humza yousaf labour deal", mode="multimodal", k=5)

retrieve.py imports its siblings, so it runs as a module (-m) rather than as a
path. embed.py and index.py have no sibling imports and run either way.

This is the independent variable of the whole project. B1 and M differ HERE and
nowhere else — same corpus, same generator, same prompt, same temperature. Three
things in this file are load-bearing:

1. Granularity. The text index is per-passage (11k entries), the image index is
   per-article (1023). Fusing raw passage scores against image scores compares
   different units, so text collapses to article level by max-pool FIRST:
       s_text(article) = max over that article's passages
   then the two streams fuse.

2. Filter BEFORE normalizing. Queries are derived from test articles, so with
   include_test=False the gold article would otherwise set the top of the min-max
   range and squash every surviving pool score — by a DIFFERENT amount in each
   stream, since the gold article's dominance differs between text and image.
   That silently changes what alpha means. Drop test rows, then normalize.

3. mode="text" is literally alpha=1.0 down one code path. Not a separate branch.
   If the two arms ran different code, a B1-vs-M difference could be an
   implementation difference rather than the images.

Scoring is exhaustive (k=ntotal on both indexes) rather than fusing two truncated
top-k lists. At this scale that costs ~2 ms and removes the failure where an
article ranks high on image but falls outside the text top-k and is fused
against an implicit zero.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import index as faiss_index
from .embed import embed_clip_text, embed_text

INDEX_DIR = "data/index"
EPS = 1e-9

_state = None


@dataclass
class Hit:
    """One retrieved article. Carries everything Day 3 puts in the prompt."""
    article_id: str
    headline: str
    caption: str
    image_path: str
    split: str
    score: float          # fused, from per-query min-max normalized streams
    s_text: float         # RAW cosine (max over passages) — use this for tau
    s_img: float          # RAW cosine
    passages: list = field(default_factory=list)

    def __repr__(self):
        return (f"Hit({self.score:.3f} | txt {self.s_text:.3f} img {self.s_img:.3f} | "
                f"{self.headline[:60]!r})")


class _State:
    """Loaded once per process: indexes, metadata, and the passage->article map."""

    def __init__(self):
        self.passages = pd.read_parquet(f"{INDEX_DIR}/passages.parquet")
        self.img_meta = pd.read_parquet(f"{INDEX_DIR}/img_meta.parquet")
        self.text_index = faiss_index.load("text")
        self.img_index = faiss_index.load("img")

        if self.text_index.ntotal != len(self.passages):
            raise ValueError("text.faiss and passages.parquet are out of sync — "
                             "re-run embed.py then index.py")
        if self.img_index.ntotal != len(self.img_meta):
            raise ValueError("img.faiss and img_meta.parquet are out of sync — "
                             "re-run embed.py then index.py")

        # article_id -> row position in img_meta (the canonical article ordering)
        self.n_articles = len(self.img_meta)
        pos = {a: i for i, a in enumerate(self.img_meta["id"])}
        self.passage_article_pos = self.passages["article_id"].map(pos).to_numpy()
        self.is_test = (self.img_meta["split"] == "test").to_numpy()


def _get_state() -> _State:
    global _state
    if _state is None:
        _state = _State()
    return _state


def _dense_scores(index, q: np.ndarray, n: int) -> np.ndarray:
    """Exhaustive search -> dense score vector of length n (index row order)."""
    D, I = faiss_index.search(index, q, k=index.ntotal)
    out = np.full(n, -np.inf, dtype="float32")
    out[I[0]] = D[0]
    return out


def _minmax(x: np.ndarray) -> np.ndarray:
    """Per-query min-max to [0, 1]. Degenerate (all-equal) range -> all zeros."""
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < EPS:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def encode_query(query: str):
    """-> (sbert_vec (384,), clip_text_vec (512,)), both unit-norm float32."""
    return embed_text([query])[0], embed_clip_text([query])[0]


def component_scores(
    query: str,
    *,
    include_test: bool = False,
    candidate_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Return one row per candidate with raw and normalized text/image scores.

    This is the efficient research interface for comparing many fusion rules:
    the query encoders run once, then alpha sweeps and rank fusion are pure NumPy.
    It intentionally returns no passages and therefore cannot be used as
    generation evidence.
    """
    st = _get_state()
    q_text, q_img = encode_query(query)
    passage_scores = _dense_scores(st.text_index, q_text, len(st.passages))
    s_text = np.full(st.n_articles, -np.inf, dtype="float32")
    np.maximum.at(s_text, st.passage_article_pos, passage_scores)
    s_img = _dense_scores(st.img_index, q_img, st.n_articles)

    if candidate_ids is not None:
        cand = st.img_meta["id"].isin(candidate_ids).to_numpy(copy=True)
    else:
        cand = np.ones(st.n_articles, dtype=bool) if include_test else ~st.is_test
    cand &= np.isfinite(s_text)
    idx = np.flatnonzero(cand)
    return pd.DataFrame({
        "id": st.img_meta.iloc[idx]["id"].to_numpy(),
        "s_text": s_text[idx],
        "s_img": s_img[idx],
        "n_text": _minmax(s_text[idx]),
        "n_img": _minmax(s_img[idx]),
    })


def retrieve(
    query: str,
    mode: str = "multimodal",
    k: int = 5,
    alpha: float = 0.5,
    include_test: bool = False,
    n_passages: int = 2,
    dedupe: bool = False,
    candidate_ids: set[str] | None = None,
) -> list[Hit]:
    """Top-k articles for `query`.

    mode        "text" (B1) forces alpha=1.0; "multimodal" (M) uses `alpha`.
    alpha       score = alpha*s_text + (1-alpha)*s_img, on normalized streams.
    include_test
                False (default) restricts results to the 873-article pool — the
                generation path. Only evaluate.py's recall@k passes True, where
                the gold article must be reachable or the metric is 0 by
                construction. generate.py asserts no test id comes back.
    n_passages  how many of the article's own passages to attach as evidence.
    dedupe      drop articles whose lead passage duplicates one already selected
                (89/1023 BBC articles share an exact first paragraph).
    candidate_ids
                Optional explicit article allowlist. When supplied it replaces
                the inherited pool/test mask and is the required interface for
                the group-safe research split.
    """
    if mode not in ("text", "multimodal"):
        raise ValueError(f"mode must be 'text' or 'multimodal', got {mode!r}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if mode == "text":
        alpha = 1.0

    st = _get_state()
    q_text, q_img = encode_query(query)

    # --- text stream: per-passage scores -> per-article max-pool
    passage_scores = _dense_scores(st.text_index, q_text, len(st.passages))
    s_text = np.full(st.n_articles, -np.inf, dtype="float32")
    np.maximum.at(s_text, st.passage_article_pos, passage_scores)

    # --- image stream: already per-article
    s_img = _dense_scores(st.img_index, q_img, st.n_articles)

    # --- filter BEFORE normalizing (see module docstring, point 2)
    if candidate_ids is not None:
        cand = st.img_meta["id"].isin(candidate_ids).to_numpy(copy=True)
    else:
        cand = np.ones(st.n_articles, dtype=bool) if include_test else ~st.is_test
    cand &= np.isfinite(s_text)
    idx = np.flatnonzero(cand)
    if idx.size == 0:
        return []

    fused = alpha * _minmax(s_text[idx]) + (1 - alpha) * _minmax(s_img[idx])

    order = idx[np.argsort(-fused)]
    rank_of = {a: r for r, a in enumerate(idx)}

    # --- assemble, attaching each article's own best passages
    hits, seen_leads = [], set()
    for pos in order:
        if len(hits) >= k:
            break
        row = st.img_meta.iloc[pos]

        p_rows = np.flatnonzero(st.passage_article_pos == pos)
        best = p_rows[np.argsort(-passage_scores[p_rows])][:n_passages]
        texts = [st.passages.iloc[int(r)]["text"] for r in best]

        if dedupe:
            lead = st.passages.iloc[int(p_rows[0])]["text"][:200].strip().lower()
            if lead in seen_leads:
                continue
            seen_leads.add(lead)

        hits.append(Hit(
            article_id=row["id"],
            headline=row["headline"],
            caption=row["caption"],
            image_path=row["image_path"],
            split=row["split"],
            score=float(fused[rank_of[pos]]),
            s_text=float(s_text[pos]),
            s_img=float(s_img[pos]),
            passages=texts,
        ))
    return hits


# ---------------------------------------------------------------- eyeball

EYEBALL = [
    "humza yousaf labour deal",
    "storm warning flooding across the UK",
    "post office horizon scandal inquiry",
    "israel gaza hostages negotiation",
    "artificial intelligence regulation",
]


def main():
    import time

    print("loading indexes...")
    _get_state()

    t = time.time()
    retrieve("warm up the encoders", k=1)
    print(f"first query {time.time() - t:.2f}s (model load), then:\n")

    for q in EYEBALL:
        t = time.time()
        txt = retrieve(q, mode="text", k=5)
        mm = retrieve(q, mode="multimodal", k=5, alpha=0.5)
        dt = (time.time() - t) * 1000 / 2

        print("=" * 100)
        print(f"QUERY: {q!r}    ({dt:.0f} ms/query)")
        print(f"{'':2} {'TEXT-ONLY (B1)':<47} {'MULTIMODAL (M, alpha=0.5)':<47}")
        print("-" * 100)
        for i in range(5):
            a = f"{txt[i].s_text:.3f} {txt[i].headline}"[:46] if i < len(txt) else ""
            b = f"{mm[i].score:.3f} {mm[i].headline}"[:46] if i < len(mm) else ""
            print(f"{i + 1:2} {a:<47} {b:<47}")

        moved = {h.article_id for h in mm} - {h.article_id for h in txt}
        print(f"   articles multimodal added that text-only missed: {len(moved)}/5")
        if mm:
            print(f"   top image: {mm[0].image_path}")
            print(f"   caption:   {mm[0].caption[:90]!r}")
    print("=" * 100)


if __name__ == "__main__":
    main()
