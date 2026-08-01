"""
Day 2, step 2: FAISS indexes over the vectors embed.py wrote.

    python src/embed.py     # first
    python src/index.py     # then this

Writes data/index/text.faiss and data/index/img.faiss.

IndexFlatIP is exhaustive inner-product search. On unit-norm vectors inner
product IS cosine similarity — but only on unit-norm vectors. Un-normalized
input still returns k results, silently ranked wrong, with no error. That is why
verify_norms() runs on load rather than being left to the caller.

macOS note: run `bash scripts/fix_openmp.sh` before the first run, and again
after any pip install touching torch/faiss/sklearn. torch, faiss and sklearn each
bundle their own libomp.dylib and two OpenMP runtimes in one process abort with
OMP: Error #15 (exit 134/139) on the first FAISS *search* — not on import. The
smoke search in __main__ exists to fire that here, where the cause is obvious,
instead of inside retrieve.py where it looks like a retrieval bug.
"""
import os

import faiss
import numpy as np

INDEX_DIR = "data/index"
NAMES = {"text": "text_emb.npy", "img": "img_emb.npy"}


def verify_norms(emb: np.ndarray, name: str = "emb", tol: float = 1e-5) -> None:
    """Fail loudly if vectors are not unit-norm float32 — IndexFlatIP needs both."""
    if emb.dtype != np.float32:
        raise TypeError(f"{name}: FAISS needs float32, got {emb.dtype}")
    dev = float(np.abs(np.linalg.norm(emb, axis=1) - 1).max())
    if dev > tol:
        raise ValueError(
            f"{name}: vectors not unit-norm (max |‖v‖-1| = {dev:.2e} > {tol:.0e}). "
            "IndexFlatIP would return cosine-looking scores that are not cosine."
        )


def build(emb: np.ndarray, name: str = "emb") -> faiss.IndexFlatIP:
    verify_norms(emb, name)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(np.ascontiguousarray(emb))
    return index


def save(index: faiss.Index, name: str) -> str:
    os.makedirs(INDEX_DIR, exist_ok=True)
    path = f"{INDEX_DIR}/{name}.faiss"
    faiss.write_index(index, path)
    return path


def load(name: str) -> faiss.Index:
    path = f"{INDEX_DIR}/{name}.faiss"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing — run: python src/index.py")
    return faiss.read_index(path)


def search(index: faiss.Index, q: np.ndarray, k: int = 5):
    """q: (d,) or (n, d) unit-norm float32 -> (scores, ids), each (n, k)."""
    q = np.atleast_2d(np.ascontiguousarray(q, dtype="float32"))
    return index.search(q, min(k, index.ntotal))


def main():
    print("building indexes...")
    import time

    built = {}
    for name, npy in NAMES.items():
        emb = np.load(f"{INDEX_DIR}/{npy}")
        t = time.time()
        idx = build(emb, name)
        save(idx, name)
        built[name] = (idx, emb)
        print(f"  {name:5s} {emb.shape} -> {idx.ntotal} vectors  ({time.time() - t:.3f}s)")

    # --- smoke search: forces any OpenMP abort to surface HERE
    print("\nsmoke search (forces the OpenMP fault if libomp is unfixed)...")
    for name, (idx, emb) in built.items():
        t = time.time()
        D, I = search(idx, emb[:1], k=5)
        print(f"  {name:5s} ok, top-5 scores {np.round(D[0], 3)}  "
              f"({(time.time() - t) * 1000:.2f} ms)")

    # --- self-retrieval: an indexed vector must come back rank 1 at score ~1.0.
    # Catches dim, dtype and index-order bugs in one line.
    print("\nself-retrieval check...")
    rng = np.random.default_rng(42)
    for name, (idx, emb) in built.items():
        probe = rng.choice(len(emb), size=min(50, len(emb)), replace=False)
        D, I = search(idx, emb[probe], k=1)
        rank1 = bool((I[:, 0] == probe).all())
        score1 = bool(np.allclose(D[:, 0], 1.0, atol=1e-4))
        print(f"  {'PASS' if rank1 and score1 else 'FAIL'}  {name:5s} "
              f"{len(probe)} probes returned themselves at rank 1, "
              f"min score {D[:, 0].min():.5f}")

    print("\nindexes ready -> data/index/*.faiss")


if __name__ == "__main__":
    main()
