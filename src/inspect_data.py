"""
Inspect the Day 1 parquet files. Read-only — safe to run any time.

    python src/inspect_data.py                    # overview of all three files
    python src/inspect_data.py --file test        # one file in detail
    python src/inspect_data.py --row 5            # full contents of row 5
    python src/inspect_data.py --search steel     # rows matching a keyword
    python src/inspect_data.py --row 5 --open     # also open the image (macOS)

Run from the repo root.
"""
import argparse
import os
import subprocess
import textwrap

import pandas as pd

PROC = "data/processed"

FILES = {
    "data": (
        f"{PROC}/data.parquet",
        "EVERYTHING scraped on Day 1. The raw pool, before splitting. "
        "You generally don't use this file directly — it's the parent of the other two.",
    ),
    "index_pool": (
        f"{PROC}/index_pool.parquet",
        "The SEARCHABLE CORPUS. These articles get chunked, embedded, and loaded into "
        "FAISS. When the demo retrieves evidence, it retrieves from here.",
    ),
    "test": (
        f"{PROC}/test.parquet",
        "The HELD-OUT QUERIES. Never indexed. You ask questions derived from these "
        "articles and check whether the summary is faithful. Kept separate so the "
        "system can't cheat by having memorized the answer.",
    ),
}

COLUMNS = {
    "id":         "md5 hash of headline+body — unique key, also the image filename",
    "headline":   "article title from BBC",
    "body":       "full article text; this is what gets chunked into passages",
    "caption":    "BBC's one-line 'description'; carries VISUAL evidence into the text prompt",
    "image_path": "path to the downloaded jpg under data/images/",
    "section":    "BBC's own section label, e.g. 'UK Politics', 'Middle East'",
    "story_type": "coarse bucket derived from section by day1_split.py (added at split time)",
}


def load(name):
    path, _ = FILES[name]
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} — run src/day1_build_dataset.py and src/day1_split.py first")
    return pd.read_parquet(path)


def wrap(text, width=100, indent="    "):
    return textwrap.fill(str(text).replace("\n", " "), width=width,
                         initial_indent=indent, subsequent_indent=indent)


def overview():
    print("=" * 78)
    print("WHAT THESE FILES ARE")
    print("=" * 78)
    for name, (path, why) in FILES.items():
        if not os.path.exists(path):
            print(f"\n{name:12s} MISSING ({path})")
            continue
        df = pd.read_parquet(path)
        print(f"\n{name}.parquet   {len(df):>5,} rows   {os.path.getsize(path)/1e6:.1f} MB")
        print(wrap(why))
    print()

    print("=" * 78)
    print("COLUMNS")
    print("=" * 78)
    df = load("test")
    for col in df.columns:
        print(f"  {col:12s} {COLUMNS.get(col, '')}")
    print()

    print("=" * 78)
    print("HOW THEY RELATE")
    print("=" * 78)
    d, ip, te = load("data"), load("index_pool"), load("test")
    print(f"""
    data.parquet  ({len(d):,} rows)
         │  day1_split.py  — shuffle seed=42, add story_type
         ├──────────────► test.parquet        ({len(te):,} rows)  ASK questions about these
         └──────────────► index_pool.parquet  ({len(ip):,} rows)  SEARCH these

    id overlap between pool and test: {len(set(ip['id']) & set(te['id']))}   (must be 0 — that's the point)
    """)

    print("=" * 78)
    print("STORY TYPES (index_pool)")
    print("=" * 78)
    counts = ip["story_type"].value_counts()
    for k, v in counts.items():
        print(f"  {k:18s} {v:5,}  {'█' * int(40 * v / counts.max())}")
    print()

    print("=" * 78)
    print("HEALTH CHECKS")
    print("=" * 78)
    for name in ("index_pool", "test"):
        df = load(name)
        missing_img = (~df["image_path"].map(os.path.exists)).sum()
        words = df["body"].str.split().str.len()
        print(f"  {name:12s} nulls={df.isna().sum().sum():<4} dup_ids={df['id'].duplicated().sum():<4} "
              f"missing_images={missing_img:<4} body_words median={int(words.median())}")
    print()
    print("Next:  python src/inspect_data.py --row 0        # look at one article")
    print("       python src/inspect_data.py --search steel # find articles by keyword")


def show_file(name):
    df = load(name)
    path, why = FILES[name]
    print(f"{path}  —  {len(df):,} rows\n")
    print(wrap(why, indent=""), "\n")
    print(df.dtypes.to_string(), "\n")
    print("sample of 5:")
    for i, (_, r) in enumerate(df.sample(min(5, len(df)), random_state=42).iterrows()):
        print(f"\n  [{i}] {r['headline'][:88]}")
        print(f"      section={r['section'] or '(none)'}  img={os.path.basename(r['image_path'])}")


def show_row(name, i, open_image=False):
    df = load(name)
    if not 0 <= i < len(df):
        raise SystemExit(f"row {i} out of range for {name} (0..{len(df)-1})")
    r = df.iloc[i]
    print("=" * 78)
    print(f"{name}.parquet  row {i}")
    print("=" * 78)
    for col in ("id", "section", "story_type", "image_path"):
        if col in r:
            print(f"{col:12s} {r[col]}")
    print(f"\nHEADLINE\n{wrap(r['headline'])}")
    print(f"\nCAPTION  (this is what the multimodal prompt injects as [IMAGE k])\n{wrap(r['caption'])}")
    body = r["body"]
    print(f"\nBODY  ({len(body.split()):,} words — chunked into passages on Day 2)")
    print(wrap(body[:1200] + ("..." if len(body) > 1200 else "")))
    if open_image:
        if os.path.exists(r["image_path"]):
            subprocess.run(["open", r["image_path"]], check=False)
            print(f"\nopened {r['image_path']}")
        else:
            print(f"\nimage missing: {r['image_path']}")


def search(term, name):
    df = load(name)
    t = term.lower()
    hit = df[df["headline"].str.lower().str.contains(t, regex=False)
             | df["body"].str.lower().str.contains(t, regex=False)]
    print(f"{len(hit)} rows in {name} match '{term}'\n")
    for i, (idx, r) in enumerate(hit.head(20).iterrows()):
        pos = df.index.get_loc(idx)
        print(f"  row {pos:<5} {r['headline'][:84]}")
    if len(hit) > 20:
        print(f"  ... and {len(hit)-20} more")
    if len(hit):
        print(f"\nlook closer:  python src/inspect_data.py --file {name} --row <row>")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", choices=list(FILES), default="index_pool")
    ap.add_argument("--row", type=int, help="show one row in full")
    ap.add_argument("--search", metavar="TERM", help="find rows containing TERM")
    ap.add_argument("--open", action="store_true", help="open the row's image (macOS)")
    ap.add_argument("--list", action="store_true", help="summarise the chosen --file")
    args = ap.parse_args()

    if args.row is not None:
        show_row(args.file, args.row, args.open)
    elif args.search:
        search(args.search, args.file)
    elif args.list:
        show_file(args.file)
    else:
        overview()


if __name__ == "__main__":
    main()
