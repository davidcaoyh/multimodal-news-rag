#!/usr/bin/env bash
# One-shot environment setup. Idempotent — safe to re-run.
#
#   bash scripts/setup.sh
#
# Creates .venv, installs pinned deps, fixes the macOS OpenMP conflict,
# and verifies the corpus is present. Run from the repo root.
set -uo pipefail

RED=$'\033[31m'; GRN=$'\033[32m'; YLW=$'\033[33m'; OFF=$'\033[0m'
ok()   { echo "${GRN}  OK${OFF}   $*"; }
warn() { echo "${YLW}  WARN${OFF} $*"; }
die()  { echo "${RED}  FAIL${OFF} $*"; exit 1; }

[ -f requirements.txt ] || die "run this from the repo root (requirements.txt not found)"

# ---------------------------------------------------------------- 1. Python
echo "[1/5] Python interpreter"
PY=""
for cand in python3.13 python3.12 python3.11 python3.10; do
    if command -v "$cand" >/dev/null 2>&1; then PY=$(command -v "$cand"); break; fi
done
if [ -z "$PY" ] && command -v python3 >/dev/null 2>&1; then
    if python3 -c 'import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)'; then
        PY=$(command -v python3)
    fi
fi
if [ -z "$PY" ]; then
    echo "${RED}  FAIL${OFF} need Python >= 3.10 (torch, faiss-cpu, datasets, streamlit all require it)."
    echo "         macOS system Python is 3.9 and will NOT work. Install one:"
    echo "           macOS   brew install python@3.12"
    echo "           Ubuntu  sudo apt install python3.12 python3.12-venv"
    exit 1
fi
ok "$($PY --version) at $PY"

# ---------------------------------------------------------------- 2. venv
echo "[2/5] virtualenv"
if [ -d .venv ]; then
    ok ".venv already exists (delete it and re-run for a clean rebuild)"
else
    "$PY" -m venv .venv || die "venv creation failed"
    ok "created .venv"
fi
VPY=.venv/bin/python
[ -f "$VPY" ] || die ".venv looks broken — rm -rf .venv and re-run"

# ---------------------------------------------------------------- 3. deps
echo "[3/5] dependencies (torch is ~1 GB, first run takes a few minutes)"
"$VPY" -m pip install --quiet --upgrade pip setuptools wheel || die "pip upgrade failed"
if [ -f requirements.lock.txt ]; then
    "$VPY" -m pip install --quiet -r requirements.lock.txt || die "pip install failed"
    ok "installed from requirements.lock.txt (pinned — use this)"
else
    "$VPY" -m pip install --quiet -r requirements.txt || die "pip install failed"
    warn "installed from requirements.txt (unpinned); versions may drift"
fi

# ---------------------------------------------------------------- 4. OpenMP
echo "[4/5] macOS OpenMP conflict"
if [ "$(uname)" = "Darwin" ]; then
    bash scripts/fix_openmp.sh >/dev/null 2>&1 \
        && ok "torch/faiss/sklearn share one libomp" \
        || die "fix_openmp.sh failed — run it directly to see why"
else
    ok "not macOS, skipping"
fi

# ---------------------------------------------------------------- 5. verify
echo "[5/5] verification"
"$VPY" - <<'PY'
import sys
try:
    import datasets, sentence_transformers, faiss, open_clip, torch, PIL, pandas, streamlit, openai
except Exception as e:
    print(f"  import failed: {e}"); sys.exit(1)
import numpy as np
e = np.random.rand(2000, 384).astype("float32"); faiss.normalize_L2(e)
idx = faiss.IndexFlatIP(384); idx.add(e)
D, I = idx.search(e[:50], 5)
assert (I[:, 0] == np.arange(50)).all(), "FAISS returned wrong neighbours"
print(f"  python {sys.version.split()[0]} | torch {torch.__version__} | faiss {faiss.__version__}")
print(f"  MPS available: {torch.backends.mps.is_available()}  (CUDA False on mac is expected)")
print("  torch + faiss coexist, search correct")
PY
[ $? -eq 0 ] || die "verification failed"
ok "all imports and FAISS search work"

# corpus
if [ -f data/processed/index_pool.parquet ] && [ -f data/processed/test.parquet ]; then
    n=$("$VPY" -c "import pandas as pd;print(len(pd.read_parquet('data/processed/index_pool.parquet')),len(pd.read_parquet('data/processed/test.parquet')))" 2>/dev/null)
    imgs=$(ls -1 data/images 2>/dev/null | wc -l | tr -d ' ')
    ok "corpus present — index_pool/test = $n, images = $imgs"
else
    warn "corpus missing. It should have come with the clone. Rebuild (~9 min):"
    echo "         .venv/bin/python src/day1_build_dataset.py --config 2024-01 --target 1500"
    echo "         .venv/bin/python src/day1_split.py"
fi

# api key
if [ ! -f .env ]; then
    warn "no .env — copy the template and add your key:  cp .env.example .env"
else
    "$VPY" - <<'PY'
import os
from dotenv import load_dotenv
load_dotenv(".env")          # explicit path: find_dotenv() can't walk the stack from stdin
k = os.getenv("OPENAI_API_KEY", "")
if not k or k.startswith("sk-proj-..."):
    print("\033[33m  WARN\033[0m .env exists but OPENAI_API_KEY is unset/placeholder (only needed from Day 3)")
else:
    print(f"\033[32m  OK\033[0m   OPENAI_API_KEY loaded ({k[:7]}...{k[-4:]})")
PY
fi

echo
echo "Done. Activate with:  source .venv/bin/activate"
echo "Explore the data:     python src/inspect_data.py"
