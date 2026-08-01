"""
Day 1: Build a small multimodal news corpus from BBC News (RealTimeData/bbc_news_alltime).

Output: data/processed/data.parquet  with columns
    id, headline, body, caption, image_path, section
plus downloaded images under data/images/.

Usage:
    python3 src/day1_build_dataset.py --config 2024-01 --target 1500
    python3 src/day1_build_dataset.py --config 2024-02 --target 1500 --append   # add a second month
    python3 src/day1_build_dataset.py --config 2024-01 --target 1500 --no-images # text-only fallback

Run from the repo root. Requires: datasets, pandas, pyarrow, requests, pillow, tqdm.
"""
import argparse, hashlib, io, os
import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

RAW_DIR = "data/raw"
IMG_DIR = "data/images"
OUT = "data/processed/data.parquet"
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# candidate column names — the script auto-detects which exist
BODY_COLS    = ["content", "text", "body", "article"]
CAPTION_COLS = ["description", "summary", "caption", "abstract"]
TITLE_COLS   = ["title", "headline"]
SECTION_COLS = ["section", "category", "topic"]
IMG_URL_COLS = ["top_image", "image", "image_url", "images", "img_url"]


def first_present(cols, available):
    for c in cols:
        if c in available:
            return c
    return None


def pick_image_url(value):
    """value may be a single URL string or a list of URLs; return first http(s) URL."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        for v in value:
            if isinstance(v, str) and v.startswith("http"):
                return v
        return None
    if isinstance(value, str) and value.startswith("http"):
        return value
    return None


def download_image(url, dest_path, timeout=8):
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200 or len(r.content) < 2000:   # skip tiny/placeholder images
            return False
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        img.thumbnail((512, 512))                            # shrink; CLIP doesn't need big
        img.save(dest_path, "JPEG", quality=88)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="2024-01", help="BBC month, e.g. 2024-01")
    ap.add_argument("--target", type=int, default=1500, help="max clean rows to keep")
    ap.add_argument("--append", action="store_true", help="append to existing parquet")
    ap.add_argument("--no-images", action="store_true", help="text-only fallback, skip downloads")
    args = ap.parse_args()

    print(f"Loading RealTimeData/bbc_news_alltime [{args.config}] ...")
    ds = load_dataset("RealTimeData/bbc_news_alltime", args.config, split="train")
    cols = ds.column_names
    print("columns:", cols)

    body_c    = first_present(BODY_COLS, cols)
    cap_c     = first_present(CAPTION_COLS, cols)
    title_c   = first_present(TITLE_COLS, cols)
    section_c = first_present(SECTION_COLS, cols)
    img_c     = first_present(IMG_URL_COLS, cols)
    print(f"using -> body:{body_c} caption:{cap_c} title:{title_c} "
          f"section:{section_c} image:{img_c}")
    if body_c is None or title_c is None:
        raise SystemExit("Could not find body/title columns — inspect ds.column_names manually.")

    rows, kept = [], 0
    for rec in tqdm(ds, desc="processing"):
        if kept >= args.target:
            break
        body = (rec.get(body_c) or "").strip()
        if len(body.split()) < 100:              # drop stubs
            continue
        title = (rec.get(title_c) or "").strip()
        if not title:
            continue
        uid = hashlib.md5((title + body[:50]).encode()).hexdigest()[:16]

        image_path = None
        if not args.no_images and img_c:
            url = pick_image_url(rec.get(img_c))
            if url:
                dest = os.path.join(IMG_DIR, f"{uid}.jpg")
                if os.path.exists(dest) or download_image(url, dest):
                    image_path = dest
            if image_path is None:               # require an image unless --no-images
                continue

        rows.append({
            "id": uid,
            "headline": title,
            "body": body,
            "caption": (rec.get(cap_c) or "").strip() if cap_c else "",
            "image_path": image_path,
            "section": (rec.get(section_c) or "").strip() if section_c else "",
        })
        kept += 1

    new_df = pd.DataFrame(rows).drop_duplicates(subset="id")
    if args.append and os.path.exists(OUT):
        old = pd.read_parquet(OUT)
        new_df = pd.concat([old, new_df]).drop_duplicates(subset="id").reset_index(drop=True)

    new_df.to_parquet(OUT, index=False)
    print(f"\nDONE. wrote {len(new_df)} rows -> {OUT}")
    print(f"with images: {new_df['image_path'].notna().sum()}")
    if len(new_df) < 500:
        print("NOTE: <500 rows. Add another month with --append, or use --no-images fallback.")


if __name__ == "__main__":
    main()
