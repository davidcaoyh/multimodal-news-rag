"""Generate report/figures/*.pdf from committed experiment artifacts.

Every number plotted here is read from a tracked CSV/JSON under
results/experiments/ — nothing is hand-typed, so a figure cannot drift from
the run that produced it. Paired-difference statistics (bootstrap CI,
Wilcoxon signed-rank p) are recomputed here directly from the committed
per-item faithfulness scores, using the same bootstrap procedure as
`research_evaluation._bootstrap` (10,000 resamples, seed=42), so they
reproduce `docs/final_results.md` exactly.

    python -m src.figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "results" / "experiments"
OUT = ROOT / "report" / "figures"

# Validated categorical palette (dataviz skill, light-surface slots 1-4).
# Fixed system -> color mapping, held constant across every figure in the
# report so an entity never changes color between panels.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY, INK, MUTED, GRID = "#b3b1a6", "#16150f", "#52514e", "#e6e5df"
COLOR = {"B1": BLUE, "M": ORANGE, "M_vision": AQUA, "M_nocap": YELLOW}

plt.rcParams.update({
    "font.size": 10.5,
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "pdf.fonttype": 42,
})


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _bootstrap_ci(diff: np.ndarray, seed: int = 42, resamples: int = 10000):
    rng = np.random.default_rng(seed)
    means = rng.choice(diff, size=(resamples, len(diff)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _paired_stats(per_item: pd.DataFrame, left: str, right: str, allowed: set[str]):
    part = per_item[per_item.kind.isin(allowed)]
    wide = part.pivot(index="test_id", columns="config", values="faithfulness")
    valid = wide[[left, right]].dropna()
    diff = (valid[right] - valid[left]).to_numpy()
    lo, hi = _bootstrap_ci(diff)
    p = float(wilcoxon(diff).pvalue)
    return {"n": len(diff), "mean": float(diff.mean()), "lo": lo, "hi": hi, "p": p,
            "wins": int((diff > 0).sum()), "losses": int((diff < 0).sum())}


# ---------------------------------------------------------------- Retrieval

def fig_alpha_sweep():
    df = pd.read_csv(EXP / "E07_research_retrieval" / "summary.csv")
    alphas = {"alpha_0": 0.0, "alpha_0.25": 0.25, "alpha_0.5": 0.5,
              "alpha_0.75": 0.75, "alpha_0.9": 0.9, "alpha_1": 1.0}
    d = df[df.method.isin(alphas)].copy()
    d["alpha"] = d.method.map(alphas)
    d = d.sort_values("alpha")
    rrf = float(df.loc[df.method == "rrf", "recall_at_5"].iloc[0])
    sel = d[np.isclose(d.alpha, 0.75)].iloc[0]

    fig, ax = plt.subplots(figsize=(4.3, 3.5), dpi=200)
    ax.plot(d.alpha, d.recall_at_5, color=BLUE, linewidth=2, marker="o",
            markersize=5, zorder=3)
    ax.scatter([sel.alpha], [sel.recall_at_5], color=AQUA, s=95, zorder=4,
               edgecolors="white", linewidths=1.3)
    ax.annotate(f"selected $\\alpha$=0.75\nRecall@5={sel.recall_at_5:.3f}",
                xy=(0.75, sel.recall_at_5), xytext=(0.30, 0.965),
                fontsize=8.7, color=INK,
                arrowprops=dict(arrowstyle="-", color=GRAY, linewidth=0.9))
    ax.axhline(rrf, color=GRAY, linestyle=":", linewidth=1.1, zorder=1)
    ax.text(0.01, rrf - 0.028, f"RRF fusion  {rrf:.3f}", fontsize=7.8, color=MUTED)
    ax.set_xlabel(r"fusion weight $\alpha$ (1 = text only, 0 = image only)")
    ax.set_ylabel("Recall@5 (development, $n$=150)")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(0.58, 1.02)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(OUT / "alpha_sweep.pdf")
    plt.close(fig)


def fig_retrieval_ceiling():
    e08 = pd.read_csv(EXP / "E08_research_reranking" / "summary.csv").set_index("method")
    e07 = pd.read_csv(EXP / "E07_research_retrieval" / "summary.csv").set_index("method")
    methods = [
        ("Image only\n($\\alpha$=0)", float(e07.loc["alpha_0", "recall_at_5"]), False),
        ("Dense text\n($\\alpha$=1)", float(e08.loc["dense_text", "recall_at_5"]), False),
        ("Fused $\\alpha$=0.75\n(ours)", float(e08.loc["mm_no_rerank", "recall_at_5"]), True),
        ("Lexical\nTF-IDF", float(e08.loc["lexical", "recall_at_5"]), False),
    ]
    labels = [m[0] for m in methods]
    values = [m[1] for m in methods]
    colors = [AQUA if m[2] else GRAY for m in methods]

    fig, ax = plt.subplots(figsize=(4.6, 3.5), dpi=200)
    bars = ax.bar(labels, values, color=colors, width=0.6, zorder=3)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.3f}",
                ha="center", fontsize=9, color=INK)
    ax.axhline(values[3], color=GRAY, linestyle=":", linewidth=1.1, zorder=1)
    ax.text(-0.42, values[3] + 0.025, "lexical ceiling\n(no neural component)",
            fontsize=7.8, color=MUTED, ha="left", va="bottom")
    ax.set_ylabel("Recall@5 (development, $n$=150)")
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    _clean(ax)
    fig.tight_layout()
    fig.savefig(OUT / "retrieval_ceiling.pdf")
    plt.close(fig)


# ------------------------------------------------------------- Faithfulness

def fig_paired_differences():
    per_item = pd.read_csv(EXP / "E12b_full_final_sample" / "evaluation" / "per_item.csv")
    rows = [
        ("$M-B_1$", "B1", "M", "prespecified"),
        ("$M_{\\mathrm{vision}}-M$", "M", "M_vision", "prespecified"),
        ("$M_{\\mathrm{vision}}-B_1$", "B1", "M_vision", "post hoc"),
    ]
    stats = [(label, status, _paired_stats(per_item, left, right, {"usable"}))
             for label, left, right, status in rows]

    fig, ax = plt.subplots(figsize=(6.4, 2.9), dpi=200)
    ypos = list(range(len(stats)))[::-1]
    for y, (label, status, s) in zip(ypos, stats):
        color = BLUE if status == "prespecified" else ORANGE
        marker = "o" if status == "prespecified" else "D"
        ax.plot([s["lo"], s["hi"]], [y, y], color=color, linewidth=2.2, zorder=2)
        ax.scatter([s["mean"]], [y], color=color, marker=marker, s=75,
                   zorder=3, edgecolors="white", linewidths=1.0)
        sig = "" if s["hi"] > 0 > s["lo"] else "*"
        ax.text(s["hi"] + 0.006, y,
                 f"{s['mean']:+.3f}  ({status}, $n$={s['n']}, $p$={s['p']:.3f}){sig}",
                 va="center", fontsize=8.6, color=INK)
    ax.axvline(0, color=MUTED, linewidth=1.0, linestyle="--", zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=10)
    ax.set_xlabel("difference in macro claim faithfulness (usable cut)")
    ax.set_xlim(-0.05, 0.17)
    ax.set_ylim(-0.7, len(stats) - 0.3)
    ax.grid(axis="y", visible=False)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(OUT / "paired_differences.pdf")
    plt.close(fig)
    return stats


def fig_sample_size():
    e12 = pd.read_csv(EXP / "E12_final_comparison" / "evaluation" / "paired_differences.csv")
    e12b = pd.read_csv(EXP / "E12b_full_final_sample" / "evaluation" / "paired_differences.csv")
    e12 = e12[e12.cut == "usable"].set_index("comparison")
    e12b = e12b[e12b.cut == "usable"].set_index("comparison")

    comparisons = [("M-B1", "$M-B_1$"), ("M_vision-M", "$M_{\\mathrm{vision}}-M$")]
    fig, ax = plt.subplots(figsize=(5.8, 3.0), dpi=200)
    for i, (key, label) in enumerate(comparisons):
        y = len(comparisons) - 1 - i
        a, b = e12.loc[key], e12b.loc[key]
        xa, xb = a.mean_difference, b.mean_difference
        ax.plot([xa, xb], [y, y], color=GRAY, linewidth=1.6, zorder=1)
        ax.scatter([xa], [y], color="#9fc3ec", s=90, zorder=2,
                   edgecolors=BLUE, linewidths=1.3,
                   label="Pilot, $n$=12 (balanced 20-article sample)" if i == 0 else None)
        ax.scatter([xb], [y], color=BLUE, s=90, zorder=3,
                   edgecolors="white", linewidths=1.0,
                   label="Full run (150-article role)" if i == 0 else None)
        # each label sits beside its own point, pushed away from the other point
        a_ha = "right" if xa <= xb else "left"
        a_dx = -0.008 if a_ha == "right" else 0.008
        b_ha = "left" if xa <= xb else "right"
        b_dx = 0.008 if b_ha == "left" else -0.008
        ax.text(xa + a_dx, y - 0.28, f"pilot: {xa:+.3f}", fontsize=8, color=MUTED, ha=a_ha)
        ax.text(xb + b_dx, y + 0.28, f"full: {xb:+.3f} ($n$={int(b.n)})", fontsize=8,
                color=INK, ha=b_ha)
    ax.axvline(0, color=MUTED, linewidth=1.2, linestyle="--", zorder=1)
    ax.set_yticks([1, 0])
    ax.set_yticklabels([c[1] for c in comparisons], fontsize=10)
    ax.set_xlabel("mean paired difference in faithfulness")
    ax.set_xlim(-0.075, 0.075)
    ax.set_ylim(-0.65, 1.75)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2, fontsize=8,
              frameon=False, columnspacing=1.2, handletextpad=0.5)
    ax.grid(axis="y", visible=False)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(OUT / "sample_size.pdf")
    plt.close(fig)


# --------------------------------------------------- Mechanism & robustness

def fig_mechanism_stress():
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3), dpi=200)

    # Panel A: E10 development diagnostic, usable-cut faithfulness per config.
    ax = axes[0]
    e10 = pd.read_csv(EXP / "E10_development_claim_evaluation" / "metrics.csv")
    e10 = e10[e10.cut == "usable"]
    order = ["B1", "M_nocap", "M", "M_vision"]
    d = e10.set_index("config").loc[order]
    colors = [GRAY, GRAY, GRAY, AQUA]
    bars = ax.bar(order, d["mean"], color=colors, width=0.6, zorder=3)
    for bar, v, n in zip(bars, d["mean"], d["count"]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", fontsize=8.7, color=INK)
        ax.text(bar.get_x() + bar.get_width() / 2, 0.03, f"$n$={int(n)}",
                ha="center", fontsize=7.5, color="white", zorder=4)
    ax.set_ylabel("faithfulness (usable cut)")
    ax.set_ylim(0, 1.08)
    ax.set_title("(a) image-sensitive development sample", fontsize=10)
    _clean(ax)

    # Panel B: E11 wrong-image stress test, correct vs. wrong pixels, per case.
    ax = axes[1]
    e11 = pd.read_csv(EXP / "E11_wrong_image_stress" / "paired.csv")
    e11 = e11.sort_values("correct_faithfulness")
    y = np.arange(len(e11))
    for i, row in enumerate(e11.itertuples()):
        ax.plot([row.correct_faithfulness, row.wrong_faithfulness], [i, i],
                color=GRAY, linewidth=1.6, zorder=1)
    # "correct" drawn as a larger hollow ring first, "wrong" as a filled dot on
    # top, so the two cases where wrong-image causes no change still show both
    # (a ring around a dot) instead of one marker silently hiding the other.
    ax.scatter(e11.correct_faithfulness, y, s=130, facecolors="none",
               edgecolors=AQUA, linewidths=2.0, zorder=2, label="correct image")
    ax.scatter(e11.wrong_faithfulness, y, color=ORANGE, s=70, zorder=3,
               edgecolors="white", linewidths=1.0, label="wrong image")
    ax.set_yticks(y)
    ax.set_yticklabels([f"case {i+1}" for i in range(len(e11))], fontsize=9)
    ax.set_xlabel("faithfulness (clean-evidence claims)")
    ax.set_xlim(0.55, 1.05)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.set_title("(b) swapped-image stress test", fontsize=10)
    ax.grid(axis="y", visible=False)
    _clean(ax)

    fig.tight_layout()
    fig.savefig(OUT / "mechanism_stress.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_alpha_sweep()
    fig_retrieval_ceiling()
    stats = fig_paired_differences()
    fig_sample_size()
    fig_mechanism_stress()
    print(f"wrote figures to {OUT}")
    for label, status, s in stats:
        print(f"  {label:28s} {status:13s} n={s['n']:3d} mean={s['mean']:+.4f} "
              f"CI=[{s['lo']:+.4f},{s['hi']:+.4f}] p={s['p']:.4f} "
              f"win/loss={s['wins']}/{s['losses']}")


if __name__ == "__main__":
    main()
