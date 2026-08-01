"""
Day 2, step 1: turn the 1023 Day-1 articles into searchable vectors.

Reads  data/processed/data.parquet  (all 1023 rows)
Writes data/index/
    passages.parquet   article_id, passage_idx, text, split   (row order == text_emb)
    text_emb.npy       (N, 384) float32, unit-norm    SBERT all-MiniLM-L6-v2
    img_meta.parquet   id, headline, caption, image_path, section, split
    img_emb.npy        (1023, 512) float32, unit-norm  CLIP ViT-B/32 image tower

Two decisions from docs/day2_guide.md are baked in here:
  D1 — embed ALL 1023 rows and carry a `split` column. Filtering to the 873-row
       pool is a post-hoc mask in retrieve.py, not an indexing choice. With test
       rows unindexed, recall@k is 0 by construction and cannot measure anything.
  D3 — the indexed image vector is the PURE image embedding. Captions ride along
       in img_meta as payload for the Day-3 prompt and never enter the vector;
       blending them would make the multimodal arm partly a second text retriever
       and "does the image help?" would stop being answerable.

Run from repo root:  python src/embed.py
"""
import os
import re
import time

import numpy as np
import pandas as pd

SRC = "data/processed/data.parquet"
TEST = "data/processed/test.parquet"
OUT_DIR = "data/index"

SBERT_MODEL = "all-MiniLM-L6-v2"

# CLIP ViT-B/32. The model name MUST be the -quickgelu variant: OpenAI trained
# these weights with QuickGELU, while open_clip's plain "ViT-B-32" config uses
# nn.GELU. Loading the plain name warns and then works — same shapes, same norms,
# every sanity check passes — but the embeddings are quietly degraded. There is no
# error to catch, so it is pinned here rather than left to the caller.
CLIP_MODEL = "ViT-B-32-quickgelu"
CLIP_PRETRAINED = "openai"

N_SENTS = 4
STRIDE = 3
MIN_CHUNK_WORDS = 8


# ---------------------------------------------------------------- cleaning

# Fixed video-player boilerplate: 26.1% of bodies. Strip the phrase ONLY — the
# sentence that follows it is a genuine video caption ("Watch: I'm really,
# really angry - postmaster gets emotional") and is real visual evidence worth
# keeping. Left in, this phrase makes 267 unrelated articles near-identical in
# embedding space and becomes retrievable "evidence" in its own right.
_BOILERPLATE = [
    re.compile(r"This video can ?not be played\.?\s*", re.I),
    re.compile(r"To play this video you need to enable JavaScript in your browser\.?\s*", re.I),
    # trailing newsroom footers (7.2%) — never article content
    re.compile(r"Follow BBC[^\n]*", re.I),
    re.compile(r"The BBC is not responsible for the content of external sites\.?\s*", re.I),
    re.compile(r"Send your story ideas to:?\s*\S+@\S+", re.I),
    re.compile(r"Sign up for our[^.\n]*newsletter[^.\n]*\.?\s*", re.I),
    # embedded social-post consent block (1.3% of bodies). Not in the day2_guide
    # table but the same failure mode: ~90 fixed words that would otherwise make
    # 13 unrelated articles look alike. The quoted post text sits *before* the
    # block and is real content, so these patterns stay tightly anchored.
    re.compile(r"This \w+ post cannot be displayed in your browser\.?\s*", re.I),
    re.compile(r"Please enable Javascript or try a different browser\.?\s*", re.I),
    re.compile(r"View original content on \w+\s*", re.I),
    re.compile(r"(Skip|End of) \w+ post( by \S+)?\s*", re.I),
    re.compile(r"This article contains content provided by[^.]*\.\s*", re.I),
    re.compile(r"We ask for your permission before anything is loaded[^.]*\.\s*", re.I),
    re.compile(r"You may want to read[^.]*before accepting\.\s*", re.I),
    re.compile(r"To view this content choose[^.]*\.\s*", re.I),
]


def clean_body(text: str) -> str:
    """Strip BBC boilerplate from an article body, preserving real content."""
    if not text:
        return ""
    for pat in _BOILERPLATE:
        text = pat.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------- chunking

# nltk is installed but its punkt data is not downloaded, and requiring every
# teammate to fetch it adds a setup step that fails offline. News prose is
# well-behaved enough for a regex splitter, which is also deterministic across
# machines — which matters because B1-vs-M numbers are compared across clones.
_SENT_SPLIT = re.compile(r'(?<=[.!?])["”’\')\]]*\s+')
_ABBREV = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Mt|Rev|Hon|Gen|Col|Sgt|Lt|Capt|Supt"
    r"|vs|etc|no|approx|est|fig|pp|inc|ltd|co|dept|govt|univ"
    r"|e\.g|i\.e|a\.m|p\.m|u\.s|u\.k)\.$",
    re.I,
)


def split_sentences(text: str) -> list[str]:
    """Regex sentence splitter with an abbreviation guard."""
    out: list[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        parts = _SENT_SPLIT.split(para)
        buf = ""
        for p in parts:
            p = p.strip()
            if not p:
                continue
            buf = f"{buf} {p}".strip() if buf else p
            # keep merging while the fragment ends on an abbreviation or a single
            # capitalised initial ("J." in "J. Smith"), which are false boundaries
            if _ABBREV.search(buf) or re.search(r"\b[A-Z]\.$", buf):
                continue
            out.append(buf)
            buf = ""
        if buf:
            out.append(buf)
    return out


def chunk(body: str, n_sents: int = N_SENTS, stride: int = STRIDE) -> list[str]:
    """Sliding window of sentences -> list of passage texts."""
    sents = split_sentences(body)
    if not sents:
        return []
    chunks, prev_end = [], -1
    for start in range(0, len(sents), stride):
        end = min(start + n_sents, len(sents))
        if end <= prev_end:  # tail window fully covered by the previous one
            break
        text = " ".join(sents[start:end]).strip()
        if len(text.split()) >= MIN_CHUNK_WORDS:
            chunks.append(text)
        prev_end = end
        if end == len(sents):
            break
    return chunks


# ---------------------------------------------------------------- encoders

_sbert = None
_clip = None  # (model, preprocess, tokenizer)


def _get_sbert():
    global _sbert
    if _sbert is None:
        from sentence_transformers import SentenceTransformer

        _sbert = SentenceTransformer(SBERT_MODEL)
    return _sbert


def _get_clip():
    global _clip
    if _clip is None:
        import open_clip
        import torch

        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED
        )
        model.eval()
        torch.set_grad_enabled(False)
        _clip = (model, preprocess, open_clip.get_tokenizer(CLIP_MODEL))
    return _clip


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float32")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return x / norms


def embed_text(texts: list[str], batch_size: int = 256) -> np.ndarray:
    """SBERT passage/query embeddings -> (N, 384) float32, unit-norm."""
    emb = _get_sbert().encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.ascontiguousarray(emb, dtype="float32")


def embed_images(paths: list[str], batch_size: int = 64) -> np.ndarray:
    """CLIP image-tower embeddings -> (N, 512) float32, unit-norm."""
    import torch
    from PIL import Image

    model, preprocess, _ = _get_clip()
    out = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i : i + batch_size]
        tensors = [preprocess(Image.open(p).convert("RGB")) for p in batch]
        feats = model.encode_image(torch.stack(tensors))
        out.append(feats.cpu().numpy())
        print(f"  images {min(i + batch_size, len(paths))}/{len(paths)}", end="\r")
    print()
    return _l2_normalize(np.vstack(out))


def embed_clip_text(texts: list[str], batch_size: int = 256) -> np.ndarray:
    """CLIP TEXT-tower embeddings -> (N, 512), same space as embed_images().

    Used for queries only, so a text query can search the image index. Never used
    on captions for indexing (D3).
    """
    import torch

    model, _, tokenizer = _get_clip()
    out = []
    for i in range(0, len(texts), batch_size):
        feats = model.encode_text(tokenizer(texts[i : i + batch_size]))
        out.append(feats.cpu().numpy())
    return _l2_normalize(np.vstack(out))


# ---------------------------------------------------------------- build

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    df = pd.read_parquet(SRC)
    test_ids = set(pd.read_parquet(TEST)["id"])
    df["split"] = np.where(df["id"].isin(test_ids), "test", "pool")
    print(f"loaded {len(df)} articles  (pool {(df.split == 'pool').sum()}, "
          f"test {(df.split == 'test').sum()})")

    # ---- passages
    t = time.time()
    rows = []
    for r in df.itertuples(index=False):
        for j, text in enumerate(chunk(clean_body(r.body))):
            rows.append({"article_id": r.id, "passage_idx": j, "text": text,
                         "split": r.split})
    passages = pd.DataFrame(rows)
    print(f"chunked -> {len(passages)} passages from {passages.article_id.nunique()} "
          f"articles  ({time.time() - t:.1f}s)")

    # ---- text embeddings
    t = time.time()
    text_emb = embed_text(passages["text"].tolist())
    print(f"SBERT {text_emb.shape}  ({time.time() - t:.1f}s, "
          f"{len(passages) / (time.time() - t):.0f}/s)")

    # ---- image embeddings (article-level, row order == img_meta)
    img_meta = df[["id", "headline", "caption", "image_path", "section", "split"]].reset_index(drop=True)
    t = time.time()
    img_emb = embed_images(img_meta["image_path"].tolist())
    print(f"CLIP  {img_emb.shape}  ({time.time() - t:.1f}s, "
          f"{len(img_meta) / (time.time() - t):.0f}/s)")

    # ---- write
    passages.to_parquet(f"{OUT_DIR}/passages.parquet", index=False)
    img_meta.to_parquet(f"{OUT_DIR}/img_meta.parquet", index=False)
    np.save(f"{OUT_DIR}/text_emb.npy", text_emb)
    np.save(f"{OUT_DIR}/img_emb.npy", img_emb)

    # ---- acceptance checks (day2_guide.md § Acceptance checks)
    print("\n--- checks ---")
    junk = passages["text"].str.contains(
        "enable Javascript|can ?not be played|cannot be displayed|Follow BBC"
        "|content provided by|View original content|newsletter and get BBC",
        regex=True, case=False).sum()
    print(f"{'PASS' if junk == 0 else 'FAIL'}  no boilerplate in passages ({junk} hits)")
    for name, emb, dim in [("text", text_emb, 384), ("img", img_emb, 512)]:
        dev = float(np.abs(np.linalg.norm(emb, axis=1) - 1).max())
        ok = emb.shape[1] == dim and emb.dtype == np.float32 and dev < 1e-5
        print(f"{'PASS' if ok else 'FAIL'}  {name}_emb {emb.shape} {emb.dtype} "
              f"max |‖v‖-1| = {dev:.2e}")
    aligned = len(passages) == len(text_emb) and len(img_meta) == len(img_emb)
    print(f"{'PASS' if aligned else 'FAIL'}  row counts aligned with embeddings")

    print(f"\nwrote -> {OUT_DIR}/  (total {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
