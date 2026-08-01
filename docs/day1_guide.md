# Day 1 — Detailed Guide: Environment + Data

**Goal by end of today:** a folder `data/processed/` containing `index_pool.parquet` and `test.parquet`, where every row has an article body, a caption, a **downloaded local image**, and a story-type label. If you have that, Day 1 is a success and the rest of the week is mostly code.

**Corpus choice:** `RealTimeData/bbc_news_alltime` on Hugging Face. Reasons: it bundles article `content`, a `description` (use as caption), a `section` (use as story-type), and image URLs; it's English; one month is a few thousand articles (perfect for our ~1–2k target); and BBC image URLs download far more reliably than NYT's.

---

## The 3 things to work on today

1. **Environment & repo** — get everything installed and importing cleanly.
2. **Land the dataset** — download BBC articles, fetch their images, write one clean `data.parquet`.
3. **Sanity-check & split** — verify pairs look right, label story-type, split into index pool + test.

Do them in order. Don't start #2 until #1 imports cleanly; don't start #3 until #2 produced a parquet.

---

## Thing 1 — Environment & repo (target: ~45 min)

**Step 1.** Make the repo and folders:
```bash
mkdir -p data/raw data/processed data/images src app results docs
python3 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
```

**Step 2.** `requirements.txt` is already in this repo. Review it.

**Step 3.** Install (this is the step most likely to be slow — torch is big):
```bash
pip install -r requirements.txt
```

**Step 4.** Verify every import works BEFORE writing real code:
```bash
python3 -c "
import datasets, sentence_transformers, faiss, open_clip, torch, PIL, requests, pandas
print('all imports OK, torch cuda:', torch.cuda.is_available())
"
```
If any import errors, fix it now — don't proceed. (CPU-only is fine; you don't need CUDA for 1–2k items.)

**Step 5.** `.gitignore` is already in this repo (ignores `.venv/`, `data/`, `.env`).

**Step 6.** Put your OpenAI key in `.env` (you won't use it today, but set it up now):
```
OPENAI_API_KEY=sk-...
```

Checkpoint: imports pass, folders exist. Move on.

---

## Thing 2 — Land the dataset (target: ~2–3 hrs, mostly image downloads running)

**Step 1 — Inspect the schema first (30 seconds, saves hours).** Column names can drift, so print them before trusting any field:
```bash
python3 -c "
from datasets import load_dataset
ds = load_dataset('RealTimeData/bbc_news_alltime', '2024-01', split='train')
print('rows:', len(ds)); print('cols:', ds.column_names)
print({k: str(v)[:80] for k,v in ds[0].items()})
"
```
You want: a body field (`content`), a caption field (`description`), a section field (`section`), and an image URL — likely `top_image` or `images`. Note which image field actually has URLs. The build script auto-detects, but confirming saves debugging.

**Step 2 — Run the build script** (`src/day1_build_dataset.py`, already in this repo). It loads one month of BBC news, picks whichever image-URL column exists, downloads each image with a timeout and skips failures, keeps only rows with body ≥ 100 words AND a downloaded image, and writes `data/processed/data.parquet` + images under `data/images/`:
```bash
python3 src/day1_build_dataset.py --config 2024-01 --target 1500
```

**Step 3 — Watch the yield.** If you get **≥ 800 clean pairs**, stop. If too few, add a second month:
```bash
python3 src/day1_build_dataset.py --config 2024-02 --target 1500 --append
```

**Step 4 — If images become a rabbit hole (timebox ~3 hrs):** fall back to text-only with `--no-images` — the parquet keeps bodies + `description` as the caption; add images tomorrow. A working text pipeline you extend beats a perfect dataset you never finish.

Checkpoint: `data/processed/data.parquet` exists with a few hundred+ rows.

---

## Thing 3 — Sanity-check & split (target: ~45 min)

**Step 1 — Eyeball 5 random pairs** (catches silent image/article mismatches):
```bash
python3 -c "
import pandas as pd
df = pd.read_parquet('data/processed/data.parquet')
print(len(df), 'rows'); print(df.columns.tolist())
for _,r in df.sample(5, random_state=1).iterrows():
    print('---'); print('HEAD:', r['headline'][:80])
    print('CAP :', str(r['caption'])[:80]); print('IMG :', r['image_path'])
    print('BODY:', r['body'][:120])
"
```
Open a couple `image_path` files by hand; confirm they match. If mismatched, fix the join before moving on.

**Step 2 — Story-type label** (for the hypothesis later). Run `src/day1_split.py` (already in this repo) — it buckets `section` into coarse story types and splits into `index_pool` / `test`:
```bash
python3 src/day1_split.py
```

**Step 3 — Commit the code (not the data):**
```bash
git add src/ docs/ requirements.txt .gitignore && git commit -m "Day 1: env + BBC dataset build + split"
```

Done. You now have `index_pool.parquet` + `test.parquet` with local images and story-type labels — exactly what Day 2 (embeddings + FAISS) needs.

---

## End-of-day success check

- [ ] All imports pass.
- [ ] `data/processed/index_pool.parquet` and `test.parquet` exist.
- [ ] Each row has: `headline, body, caption, image_path, section, story_type`.
- [ ] You opened a few images by hand and they match their articles.
- [ ] ≥ ~500 rows total (more is nice, not required).

If image download defeated you: same checklist but with `caption`-only (from `description`) and no `image_path` — add images Day 2 morning.
