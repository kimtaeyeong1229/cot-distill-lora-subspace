#!/usr/bin/env python3
"""Result figures for the manuscript (Figures 2-5), drawn from the experiment JSONs.

Plain matplotlib in the style of the reference papers: few colours, solid lines for students, a dashed line for
the teacher or a reference, log axes where values span orders of magnitude, nine-cell means as bars with the
cells as dots. Run from the repo root; writes PDFs into the manuscript repo's figures/ folder.
"""

import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = Path("experiments")
OUT = Path("/home/lab/workspace/NAACL_cot_distill_lora_subspace/figures")
L = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
X = [float(l) for l in L]
PAIRS = [("A", "A"), ("B", "B"), ("C", "D")]           # paper name, directory/key name
STU = [("Qwen", "Qwen2.5-0.5B"), ("Llama", "Llama-3.2-1B"), ("OLMo", "OLMo-2-1B")]
COL = {"Qwen": "#1f77b4", "Llama": "#ff7f0e", "OLMo": "#2ca02c"}
GREY = "#555555"
FULL, HALF = 6.9, 3.3                                   # ACL text width and column width in inches

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

load = lambda n: json.loads((EXP / n).read_text())
think = load("think_rate.json")
resample = load("phi_resample_s012.json")
consensus = load("consensus_pre.json")
base = load("baselines.json")
spread = load("seed_spread.json")
CELLS = [f"{p}/{s}" for p, _ in PAIRS for s, _ in STU]


def cell_key(p, s):
    return f"{p}/{s}"


# ---------------------------------------------------------------- Figure 2: R(lambda) against the teacher
def fig2():
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 1.9), sharey=True)
    for ax, (p, d) in zip(axes, PAIRS):
        rates = [think[d]["rates"][l] for l in L]
        drops = [rates[i] - rates[i + 1] for i in range(5)]
        i = max(range(5), key=lambda k: drops[k])
        ax.axvspan(X[i], X[i + 1], color="#f2e3b3", alpha=0.6, lw=0)
        ax.plot(X, rates, ls="--", color=GREY, marker="s", ms=3, lw=1.2, label="Teacher think rate")
        for s, full in STU:
            rows = resample[f"{d}/{full}"]["rows"]
            y = [rows[l]["retention"] for l in L]
            e = [rows[l].get("retention_sd", 0) for l in L]
            ax.errorbar(X, y, yerr=e, color=COL[s], marker="o", ms=3, lw=1.3, capsize=1.5, label=f"$R$, {full}")
        ax.set_title(f"Pair {p}")
        ax.set_xticks(X)
        ax.set_xticklabels(["0", ".2", ".4", ".6", ".8", "1"])
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylim(-0.03, 1.05)
    axes[0].set_ylabel("Think rate / $R(\\lambda)$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.12))
    fig.savefig(OUT / "fig2_switch.pdf")


# ---------------------------------------------------------------- Figure 3: coverage of the shared region
def fig3():
    fig, ax = plt.subplots(figsize=(HALF, 1.8))
    xs = range(len(CELLS))
    for name, key, mk, c in [("Pre-switch (LOO)", "in_loo", "o", "#1f77b4"), ("Post-switch", "out", "s", "#d62728"),
                             ("Chance", "chance", "x", GREY)]:
        ys = [consensus[f"{d}/{full}"][key] for p, d in PAIRS for s, full in STU]
        ax.plot(xs, ys, ls="none", marker=mk, ms=4, color=c, label=name)
    ax.set_yscale("log")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{p}/{s}" for p, _ in PAIRS for s, _ in STU], rotation=45, ha="right")
    for b in (2.5, 5.5):
        ax.axvline(b, color="#cccccc", lw=0.6)
    ax.set_ylabel("Coverage of the\nshared region")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    fig.savefig(OUT / "fig3_coverage.pdf")


# ---------------------------------------------------------------- Figure 4: removing the region
COND = [("base0", r"$\lambda=0$"), ("rm_pre", "Shared\nremoved"), ("budget", "Length-\nmatched\ncut"),
        ("rm_other", "Other\nremoved"), ("gs_energy", "Top\nremoved\n(matched)"), ("rm_scale", "Scaled\ndown")]


def bars(ax, vals, colors, log=False):
    for j, v in enumerate(vals):
        ax.bar(j, st.mean(v), color=colors[j], width=0.65, alpha=0.85, zorder=1)
        ax.scatter([j + (k - 4) * 0.045 for k in range(len(v))], v, s=6, color="black", alpha=0.6, zorder=2, lw=0)
    if log:
        ax.set_yscale("log")


def fig4():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL, 1.9))
    colors = ["#9e9e9e", "#e8b730", "#7aa6d6", "#7aa6d6", "#7aa6d6", "#7aa6d6"]
    mk = [[base[c][k]["markers"]["mean"] for c in CELLS] for k, _ in COND]
    ac = [[100 * base[c][k]["accuracy"]["mean"] for c in CELLS] for k, _ in COND]
    bars(a1, mk, colors, log=True)
    a1.set_ylabel("Reflective markers\nper 1,000 words")
    bars(a2, ac, colors)
    a2.set_ylabel("MATH-500 accuracy (%)")
    for ax in (a1, a2):
        ax.set_xticks(range(len(COND)))
        ax.set_xticklabels([n for _, n in COND], fontsize=6.5)
    fig.savefig(OUT / "fig4_remove.pdf")


# ---------------------------------------------------------------- Figure 5: graft and amplify
def fig5():
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 1.9))
    a1, a2, a3 = axes
    e = lambda c, k, f: spread[c]["edits"][k][f]["mean"]
    lam0 = [spread[c]["table1"]["0.0"]["mean"] for c in CELLS]
    # (a) markers at beta = 2, early window, relative to the lambda = 0 adapter's early-window markers
    head0 = {}
    for c in CELLS:
        md = json.loads((EXP / "main_density.json").read_text())[c]["0.0"]
        head0[c] = md["head_markers_per_1k"]
    conds = [("graft2", "Graft\nshared"), ("graft_other2", "Graft\nother"), ("graft_whole2", "Graft\nwhole")]
    vals = [[e(c, k, "head_markers_per_1k") for c in CELLS] for k, _ in conds]
    bars(a1, vals, ["#e8b730", "#7aa6d6", "#9e9e9e"], log=True)
    a1.axhline(st.mean(head0.values()), ls="--", color=GREY, lw=1)
    a1.text(-0.45, st.mean(head0.values()) * 1.25, r"$\lambda=0$ adapter", ha="left", fontsize=6.5, color=GREY)
    a1.set_xticks(range(3)); a1.set_xticklabels([n for _, n in conds])
    a1.set_ylabel("Markers per 1,000 words\n(early window)")
    a1.set_title(r"Graft into $\lambda=1$ ($\beta=2$)")
    # (b) completion at beta = 2
    vals = [[100 * e(c, k, "completed") for c in CELLS] for k, _ in conds]
    bars(a2, vals, ["#e8b730", "#7aa6d6", "#9e9e9e"])
    a2.set_xticks(range(3)); a2.set_xticklabels([n for _, n in conds])
    a2.set_ylabel("Completed responses (%)")
    a2.set_title(r"Graft into $\lambda=1$ ($\beta=2$)")
    # (c) amplify with the norm held fixed, relative to the unedited lambda = 0 adapter
    amp = [("ampn1", r"$\gamma=1$"), ("ampn4", r"$\gamma=4$"), ("ampV", "All in\nregion")]
    vals = [[100 * e(c, k, "markers_per_1k") / lam0[i] for i, c in enumerate(CELLS)] for k, _ in amp]
    bars(a3, vals, ["#e8b730"] * 3)
    a3.axhline(100, ls="--", color=GREY, lw=1)
    a3.set_xticks(range(3)); a3.set_xticklabels([n for _, n in amp])
    a3.set_ylabel(r"Markers (% of $\lambda=0$)")
    a3.set_title("Amplify (norm fixed)")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / "fig5_graft_amplify.pdf")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig2(); fig3(); fig4(); fig5()
    print("written to", OUT)
