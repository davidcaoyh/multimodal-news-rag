# Setup Guide (teammates start here)

Getting from a fresh clone to a working environment. **~10 minutes**, most of it
downloading PyTorch.

---

## TL;DR

```bash
git clone <repo-url> && cd project
bash scripts/setup.sh
source .venv/bin/activate
python src/inspect_data.py          # confirm the corpus looks right
```

If `setup.sh` ends with all-green `OK` lines, you are done. The corpus ships with
the repo — **do not re-download it** (see "Why the data is committed" below).

---

## Prerequisites

**Python 3.10 or newer.** This is the single most common failure.

| | |
|---|---|
| macOS system Python | 3.9 — **will not work** |
| Fix (macOS) | `brew install python@3.12` |
| Fix (Ubuntu) | `sudo apt install python3.12 python3.12-venv` |

Every core dependency requires ≥3.10: torch 2.13, faiss-cpu 1.14.3,
sentence-transformers 5.6.1, datasets 5.0.1, streamlit 1.60. On 3.9, pip silently
resolves years-old versions or fails outright on faiss.

`setup.sh` auto-detects 3.13/3.12/3.11/3.10 and tells you what to install if none
is found. You do **not** need a GPU — everything runs on CPU in seconds.

**Disk:** ~1.5 GB for `.venv`, ~700 MB for cached model weights.

---

## What `setup.sh` does

Idempotent — re-run it any time.

1. Finds a Python ≥3.10.
2. Creates `.venv/` (skips if present; `rm -rf .venv` to rebuild clean).
3. Installs from `requirements.lock.txt` — **pinned**, so everyone gets identical
   versions. Falls back to `requirements.txt` with a warning.
4. Runs `scripts/fix_openmp.sh` on macOS (see below).
5. Verifies imports, runs a real FAISS search and checks the neighbours are
   correct, then reports whether the corpus and API key are in place.

---

## The macOS OpenMP trap

**If you are on a Mac, read this — it will bite you otherwise.**

torch, faiss, and scikit-learn each bundle their own `libomp.dylib`. Loading two
OpenMP runtimes in one process aborts with `OMP: Error #15` or segfaults
(exit 134/139) the **first time FAISS runs a search** — not at import. So
`import faiss, torch` looks fine and the crash arrives later, in `index.py`.

`scripts/fix_openmp.sh` points faiss's and sklearn's copies at torch's single
runtime. `setup.sh` runs it for you.

> **Re-run `bash scripts/fix_openmp.sh` after ANY pip install that touches torch,
> faiss-cpu, or scikit-learn.** pip restores the bundled copies and the crash
> silently comes back.

Workarounds that do *not* work, so don't waste time on them: importing faiss
before torch, `KMP_DUPLICATE_LIB_OK=TRUE`, or `OMP_NUM_THREADS=1`.

---

## API key (not needed until Day 3)

Days 1–2 are fully local. When you reach generation:

```bash
cp .env.example .env       # then edit .env
```

Create a **restricted** project key at <https://platform.openai.com/api-keys> with
exactly one permission:

> **Model capabilities → Chat completions (`/v1/chat/completions`) = Request**

Everything else stays **None**. Note the dropdown says *Request*, not *Write* —
inference endpoints use None/Request. **Embeddings are not needed**: SBERT and
CLIP run locally. Set a **$5–10 spend cap** on the project; whole-week usage is
well under $1.

`.env` is gitignored. Never commit it.

---

## Why the data is committed

`data/processed/*.parquet` (5 MB) and `data/images/` (38 MB) are **in the repo on
purpose**, which is unusual and deliberate.

Rebuilding from Hugging Face is not deterministic — image downloads fail at
different rates on different machines and networks, so each person would end up
with a different number of pairs and therefore a different `index_pool`/`test`
split. B1-vs-M faithfulness numbers computed on different corpora aren't
comparable, which would defeat the point of the experiment.

So: **clone, don't rebuild.** You should see 873 index-pool rows, 150 test rows,
and 1023 images. `setup.sh` checks this.

Only rebuild if you are deliberately changing the corpus:

```bash
python src/day1_build_dataset.py --config 2024-01 --target 1500   # ~9 min
python src/day1_split.py
```

---

## Exploring the data

```bash
python src/inspect_data.py                    # what the files are + health checks
python src/inspect_data.py --row 0            # one article in full
python src/inspect_data.py --search steel     # find articles by keyword
python src/inspect_data.py --row 337 --open   # also open the image (macOS)
```

Start here if you're unsure what `index_pool` vs `test` means — the overview
explains the split and why it exists.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `need Python >= 3.10` | System Python is 3.9. `brew install python@3.12`, re-run. |
| Segfault / `OMP: Error #15` at a FAISS call | `bash scripts/fix_openmp.sh` |
| `ModuleNotFoundError` | venv not active. `source .venv/bin/activate` |
| `torch.cuda.is_available()` is `False` | Expected on Mac. MPS is there; CPU is fast enough. |
| `OPENAI_API_KEY` is `None` | `load_dotenv()` resolves relative to the **calling file's** directory, not cwd. From a scratch file outside the repo it finds nothing and fails silently. Pass an explicit path. |
| 401 from OpenAI | Key scope wrong — needs Chat completions = Request. |
| `insufficient_quota` | Billing/credits, not permissions. |
| `pandas<3.0.0` conflict warning | Only if you installed `parquet-tools` yourself. Harmless; it isn't a project dep. |

---

## Repo map

```
src/day1_build_dataset.py   DONE  fetch BBC + images -> data.parquet
src/day1_split.py           DONE  story_type labels + index_pool/test split
src/inspect_data.py         DONE  read-only data explorer
scripts/setup.sh            DONE  this guide, automated
scripts/fix_openmp.sh       DONE  macOS libomp fix
src/embed.py                TODO  Day 2 — SBERT + CLIP encoders
src/index.py                TODO  Day 2 — build/load/search FAISS
src/retrieve.py             TODO  Day 2 — text-only + fused retrieval
src/generate.py             TODO  Day 3 — prompt + LLM + abstention
app/streamlit_app.py        TODO  Day 4 — demo with text/multimodal toggle
src/evaluate.py             TODO  Day 5 — faithfulness + recall@k
```

Read `docs/plan_7day.md` for the plan and `CLAUDE.md` for architecture,
conventions, and current status.
