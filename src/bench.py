"""
Measure and record pipeline timings -> results/timings.csv

    python -m src.bench

Re-measures every step from scratch rather than scraping the console, so the CSV
is either current or obviously absent. Run it after anything that could plausibly
change cost: corpus size, chunk params, encoder swap, a new machine.

Why record this at all: the report claims this design scales (the proposal targets
~50k articles). items/sec turns that from an assertion into an extrapolation. The
per-query BREAKDOWN is the useful part — search is ~0.5 ms while encoding the query
is ~40 ms, so if the Day 4 demo feels slow the fix is query caching, not a fancier
index. Timings are machine-specific; the CSV records which machine produced them.
"""
import csv
import os
import platform
import resource
import time

import numpy as np
import pandas as pd

from . import index as faiss_index
from .embed import OUT_DIR as INDEX_DIR
from .embed import embed_clip_text, embed_images, embed_text
from .retrieve import _get_state, retrieve

OUT = "results/timings.csv"
QUERY = "post office horizon scandal inquiry"
REPEATS = 20


def peak_rss_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kilobytes
    return rss / 1e9 if platform.system() == "Darwin" else rss / 1e6


def timed(fn, repeats: int = 1):
    """-> (seconds per call, result of last call)."""
    t = time.perf_counter()
    for _ in range(repeats):
        out = fn()
    return (time.perf_counter() - t) / repeats, out


def main():
    os.makedirs("results", exist_ok=True)
    rows = []

    def rec(step, n, secs, note=""):
        rows.append({
            "step": step,
            "n_items": n,
            "seconds": round(secs, 4),
            "items_per_sec": round(n / secs, 1) if secs > 0 else "",
            "note": note,
        })
        print(f"  {step:26s} n={n:<6} {secs * 1000:9.2f} ms  "
              f"{n / secs:10.0f}/s" if secs > 0 else f"  {step}")

    passages = pd.read_parquet(f"{INDEX_DIR}/passages.parquet")
    img_meta = pd.read_parquet(f"{INDEX_DIR}/img_meta.parquet")

    print("encoders (cold-cache model load excluded from the encode rows):")
    _, _ = timed(lambda: embed_text(["warmup"]))          # load SBERT
    _, _ = timed(lambda: embed_clip_text(["warmup"]))     # load CLIP

    s, text_emb = timed(lambda: embed_text(passages["text"].tolist()))
    rec("sbert_encode_passages", len(passages), s, "all-MiniLM-L6-v2, 384-d")

    s, img_emb = timed(lambda: embed_images(img_meta["image_path"].tolist()))
    rec("clip_encode_images", len(img_meta), s, "ViT-B-32-quickgelu, 512-d")

    print("\nindex:")
    s, ti = timed(lambda: faiss_index.build(text_emb, "text"))
    rec("faiss_build_text", len(text_emb), s, "IndexFlatIP 384-d")
    s, ii = timed(lambda: faiss_index.build(img_emb, "img"))
    rec("faiss_build_img", len(img_emb), s, "IndexFlatIP 512-d")

    print("\nper-query breakdown:")
    qt = embed_text([QUERY])
    qi = embed_clip_text([QUERY])

    s, _ = timed(lambda: embed_text([QUERY]), REPEATS)
    rec("query_encode_sbert", 1, s, f"mean of {REPEATS}")
    s, _ = timed(lambda: embed_clip_text([QUERY]), REPEATS)
    rec("query_encode_clip", 1, s, f"mean of {REPEATS}")
    s, _ = timed(lambda: faiss_index.search(ti, qt, k=ti.ntotal), REPEATS)
    rec("faiss_search_text_full", 1, s, f"k=ntotal={ti.ntotal}")
    s, _ = timed(lambda: faiss_index.search(ii, qi, k=ii.ntotal), REPEATS)
    rec("faiss_search_img_full", 1, s, f"k=ntotal={ii.ntotal}")

    print("\nend-to-end retrieve():")
    _get_state()
    s, _ = timed(lambda: retrieve(QUERY, mode="text", k=5), REPEATS)
    rec("retrieve_text_b1", 1, s, "mode='text' (alpha=1.0)")
    s, _ = timed(lambda: retrieve(QUERY, mode="multimodal", k=5), REPEATS)
    rec("retrieve_multimodal_m", 1, s, "alpha=0.5")

    meta = {
        "machine": f"{platform.machine()} {platform.system()} {platform.release()}",
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "peak_rss_gb": round(peak_rss_gb(), 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
    }
    for k, v in meta.items():
        rows.append({"step": k, "n_items": "", "seconds": "", "items_per_sec": "", "note": v})

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "n_items", "seconds", "items_per_sec", "note"])
        w.writeheader()
        w.writerows(rows)

    print(f"\npeak RSS {meta['peak_rss_gb']} GB on {meta['machine']}")
    print(f"wrote -> {OUT}")


if __name__ == "__main__":
    main()
