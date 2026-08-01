#!/usr/bin/env bash
# macOS/arm64 fix: torch, faiss, and sklearn each bundle their own libomp.dylib.
# Loading two of them in one process aborts with "OMP: Error #15" or segfaults
# (exit 134/139) the first time FAISS runs a parallel region — e.g. index.search().
#
# Fix: point faiss's and sklearn's copies at torch's single runtime.
# Re-run this after ANY pip install/upgrade that touches torch, faiss-cpu, or
# scikit-learn — pip restores the bundled copies and the crash comes back.
#
#   bash scripts/fix_openmp.sh
set -euo pipefail

VENV="${1:-.venv}"
SITE=$(cd "$VENV" && ./bin/python -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
TORCH_OMP="$SITE/torch/lib/libomp.dylib"

[ -f "$TORCH_OMP" ] || { echo "no torch libomp at $TORCH_OMP — nothing to do"; exit 0; }

for pkg in faiss sklearn; do
    target="$SITE/$pkg/.dylibs/libomp.dylib"
    if [ -e "$target" ] && [ ! -L "$target" ]; then
        ln -sf "$TORCH_OMP" "$target"
        echo "linked $pkg/.dylibs/libomp.dylib -> torch/lib/libomp.dylib"
    elif [ -L "$target" ]; then
        echo "$pkg already linked"
    fi
done

echo "verifying..."
"$VENV/bin/python" - <<'PY'
import numpy as np, torch, faiss
e = np.random.rand(2000, 384).astype('float32'); faiss.normalize_L2(e)
idx = faiss.IndexFlatIP(384); idx.add(e)
D, I = idx.search(e[:50], 5)
assert (I[:, 0] == np.arange(50)).all() and np.allclose(D[:, 0], 1.0, atol=1e-4)
print("OK — torch + faiss coexist, search results correct")
PY
