#!/usr/bin/env python3
"""Result figures for the manuscript (Figures 2-5), drawn from the experiment JSONs.

Styled after Jack Morris's plots (e.g., "How much do language models memorize?"): cream plot area with a grey
grid and a full black frame, bold axis titles in sentence case with the unit on a lighter second line, an ordered
navy-grey-yellow palette, boxed legends with a bold title, and labelled dashed reference lines. Categorical
comparisons are dot plots: one dot per cell, coloured by student, with the nine-cell mean as a black bar.
Run from the repo root; writes PDFs into the manuscript repo's figures/ folder.
"""

import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

EXP = Path("experiments")
OUT = Path("/home/lab/workspace/NAACL_cot_distill_lora_subspace/figures")
L = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
X = [float(l) for l in L]
PAIRS = [("A", "A"), ("B", "B"), ("C", "D")]          # paper name, key in the result files
STU = [("Qwen", "Qwen2.5-0.5B"), ("Llama", "Llama-3.2-1B"), ("OLMo", "OLMo-2-1B")]
NAVY, SLATE, YELLOW = "#1f2c56", "#7b8798", "#f2cf3a"
COL = {"Qwen": NAVY, "Llama": SLATE, "OLMo": YELLOW}
CREAM, GRID, REF, RED = "#f8f2e7", "#cfc8bb", "#8c8c8c", "#b2223a"
FULL, HALF = 6.9, 3.3

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold",
    "axes.labelsize": 8, "axes.labelweight": "bold", "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
    "legend.fontsize": 6.5, "legend.title_fontsize": 6.8, "axes.facecolor": CREAM, "axes.edgecolor": "black",
    "axes.linewidth": 0.8, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "legend.frameon": True, "legend.framealpha": 1.0, "legend.edgecolor": "#444444", "legend.fancybox": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "mathtext.fontset": "custom",
    "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
})

load = lambda n: json.loads((EXP / n).read_text())
think = load("think_rate.json")
resample = load("phi_resample_s012.json")
consensus = load("consensus_pre.json")
base = load("baselines.json")
spread = load("seed_spread.json")
density = load("main_density.json")
CELLS = [(p, s) for p, _ in PAIRS for s, _ in STU]


def labels(ax, x=None, xunit=None, y=None, yunit=None):
    """Bold axis title in sentence case; the unit goes on a lighter second line, placed against the title."""
    if x:
        ax.set_xlabel(x, fontweight="bold", labelpad=3)
        if xunit:
            ax.annotate(xunit, xy=(0.5, 0), xycoords=ax.xaxis.label, xytext=(0, -1.5), textcoords="offset points",
                        ha="center", va="top", fontsize=7, color="#333333")
    if y:
        ax.set_ylabel(y, fontweight="bold", labelpad=11 if yunit else 4)
        if yunit:
            ax.annotate(yunit, xy=(1, 0.5), xycoords=ax.yaxis.label, xytext=(1.5, 0), textcoords="offset points",
                        ha="left", va="center", rotation=90, fontsize=7, color="#333333")


def student_handles(lines=False):
    return [Line2D([], [], color=COL[s], marker="o", ms=3.5, lw=1.2 if lines else 0, mew=0.4,
                   mec="black" if (s == "OLMo" or not lines) else COL[s], label=full) for s, full in STU]


def student_legend(ax, loc="lower left", extra=(), title="Student"):
    h = [Line2D([], [], color=COL[s], marker="o", ms=3.5, lw=1.2, mec="black" if s == "OLMo" else COL[s], mew=0.4,
                label=full) for s, full in STU] + list(extra)
    ax.legend(handles=h, title=title, loc=loc)


# ---------------------------------------------------------------- Figure 2: R(lambda) against the teacher
def fig2():
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.3), sharey=True)
    for ax, (p, d) in zip(axes, PAIRS):
        rates = [think[d]["rates"][l] for l in L]
        drops = [rates[i] - rates[i + 1] for i in range(5)]
        i = max(range(5), key=lambda k: drops[k])
        ax.axvspan(X[i], X[i + 1], color="#eadfc4", lw=0, zorder=0)
        ax.plot(X, rates, ls="--", color=REF, lw=1.2, marker="s", ms=2.6, zorder=2)
        for s, full in STU:
            rows = resample[f"{d}/{full}"]["rows"]
            y = [rows[l]["retention"] for l in L]
            e = [rows[l].get("retention_sd", 0) for l in L]
            ax.errorbar(X, y, yerr=e, color=COL[s], marker="o", ms=3.2, lw=1.3, capsize=1.5, zorder=3,
                        mec="black" if s == "OLMo" else COL[s], mew=0.4)
        ax.set_title(f"Pair {p}")
        ax.set_xticks(X)
        ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])
        ax.set_ylim(-0.04, 1.08)
        ax.set_xlim(-0.05, 1.05)
        labels(ax, x=r"Interpolation ratio $\lambda$", xunit=None)
    labels(axes[0], y=r"Direction overlap $R(\lambda)$", yunit=r"(1 = same directions as at $\lambda = 0$)")
    teacher = Line2D([], [], color=REF, ls="--", lw=1.2, marker="s", ms=2.6, label="Teacher think rate")
    fig.tight_layout(w_pad=0.8, rect=(0, 0.1, 1, 1))
    fig.legend(handles=student_handles(lines=True) + [teacher], loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.savefig(OUT / "fig2_switch.pdf")


# ---------------------------------------------------------------- Figure 3: coverage of the shared region
def fig3():
    fig, ax = plt.subplots(figsize=(HALF, 1.95))
    xs = list(range(9))
    for name, key, mk, c in [("Pre-switch (LOO)", "in_loo", "o", NAVY), ("Post-switch", "out", "s", SLATE),
                             ("Chance", "chance", "x", REF)]:
        ys = [consensus[f"{d}/{full}"][key] for p, d in PAIRS for s, full in STU]
        ax.plot(xs, ys, ls="none", marker=mk, ms=4.2, color=c, mec="black" if mk != "x" else c, mew=0.4,
                label=name)
    ax.set_yscale("log")
    ax.set_ylim(3e-3, 1.2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{p}/{s}" for p, s in CELLS], rotation=40, ha="right")
    for b in (2.5, 5.5):
        ax.axvline(b, color="#999999", lw=0.8, ls=":")
    labels(ax, y="Region coverage", yunit="(fraction of the shared region, log)")
    ax.legend(title="Adapters", loc="center right", bbox_to_anchor=(1.0, 0.42))
    fig.savefig(OUT / "fig3_coverage.pdf")


# ---------------------------------------------------------------- dot plots for the edits
def dotplot(ax, conds, values, log=False, ref=None, ref_label=None):
    """conds: list of names (top to bottom); values: dict name -> list of 9 values in CELLS order."""
    n = len(conds)
    for j, c in enumerate(conds):
        y = n - 1 - j
        v = values[c]
        for k, (p, s) in enumerate(CELLS):
            jitter = (STU.index(next(t for t in STU if t[0] == s)) - 1) * 0.14
            ax.plot(v[k], y + jitter, "o", ms=3.4, color=COL[s], mec="black", mew=0.35, zorder=3)
        m = st.mean(v)
        ax.plot([m, m], [y - 0.32, y + 0.32], color="black", lw=1.8, zorder=4)
    ax.set_yticks(range(n))
    ax.set_yticklabels(list(reversed(conds)))
    ax.set_ylim(-0.6, n - 0.4)
    if log:
        ax.set_xscale("log")
    if ref is not None:
        ax.axvline(ref, color=REF, ls="--", lw=1.1, zorder=2)
        if ref_label:
            ax.text(ref, -0.55, " " + ref_label, fontsize=6.2, color="#555555", ha="left", va="bottom")
    ax.grid(axis="y", visible=False)


MEAN_HANDLE = Line2D([], [], color="black", lw=1.8, label="Nine-cell mean")


def fig4():
    conds = [("base0", r"$\lambda = 0$ adapter"), ("rm_pre", "Shared removed"), ("budget", "Length-matched cut"),
             ("rm_other", "Other removed"), ("gs_energy", "Top removed (matched)"), ("rm_scale", "Scaled down")]
    names = [n for _, n in conds]
    cell = lambda p, s: f"{p}/{s}"
    mk = {n: [base[cell(p, s)][k]["markers"]["mean"] for p, s in CELLS] for k, n in conds}
    ac = {n: [100 * base[cell(p, s)][k]["accuracy"]["mean"] for p, s in CELLS] for k, n in conds}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL, 2.2), sharey=True)
    dotplot(a1, names, mk, log=True)
    labels(a1, x="Reflective markers", xunit="(per 1,000 words, log scale)")
    dotplot(a2, names, ac)
    labels(a2, x="MATH-500 accuracy", xunit="(%)")
    fig.tight_layout(w_pad=1.0, rect=(0, 0.1, 1, 1))
    fig.legend(handles=student_handles() + [MEAN_HANDLE], loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.savefig(OUT / "fig4_remove.pdf")


def fig5():
    cell = lambda p, s: f"{p}/{s}"
    e = lambda p, s, k, f: spread[cell(p, s)]["edits"][k][f]["mean"]
    grafts = [("graft2", "Graft shared"), ("graft_other2", "Graft other"), ("graft_whole2", "Graft whole")]
    head0 = st.mean(density[cell(p, s)]["0.0"]["head_markers_per_1k"] for p, s in CELLS)
    gm = {n: [e(p, s, k, "head_markers_per_1k") for p, s in CELLS] for k, n in grafts}
    gc = {n: [100 * e(p, s, k, "completed") for p, s in CELLS] for k, n in grafts}
    amps = [("ampn1", r"$\gamma = 1$"), ("ampn4", r"$\gamma = 4$"), ("ampV", "All in region")]
    am = {n: [100 * e(p, s, k, "markers_per_1k") / spread[cell(p, s)]["table1"]["0.0"]["mean"] for p, s in CELLS]
          for k, n in amps}
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 1.85), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    a1, a2, a3 = axes
    dotplot(a1, [n for _, n in grafts], gm, log=True, ref=head0)
    labels(a1, x="Reflective markers", xunit="(per 1,000 words, early window, log)")
    a1.set_title(r"Graft into $\lambda = 1$ ($\beta = 2$)")
    dotplot(a2, [n for _, n in grafts], gc)
    a2.set_yticklabels([])
    labels(a2, x="Completed responses", xunit="(%)")
    a2.set_title(r"Graft completion ($\beta = 2$)")
    dotplot(a3, [n for _, n in amps], am, ref=100)
    labels(a3, x="Reflective markers", xunit=r"(% of the $\lambda = 0$ adapter)")
    a3.set_title("Amplify (norm fixed)")
    h = student_handles() + [MEAN_HANDLE, Line2D([], [], color=REF, ls="--", lw=1.1, label=r"$\lambda = 0$ adapter")]
    fig.tight_layout(w_pad=0.6, rect=(0, 0.1, 1, 1))
    fig.legend(handles=h, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02), title=None)
    fig.savefig(OUT / "fig5_graft_amplify.pdf")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig2(); fig3(); fig4(); fig5()
    print("written to", OUT)
