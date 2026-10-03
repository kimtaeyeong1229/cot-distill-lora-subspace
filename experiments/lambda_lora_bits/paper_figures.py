#!/usr/bin/env python3
"""Result figures for the manuscript (Figures 2-5), drawn from the experiment JSONs.

Formats follow the closest prior work. The lambda sweep (Figure 2) is a line plot with the teacher's think rate in
its own row above the student curves, as in Wu et al. (2026, model interpolation, Figure 3). The edits (Figures 4
and 5) are grouped bar charts, one group per condition, as in Arditi et al. (2024, refusal direction, Figures 1
and 3) and Venhoff et al. (2026, Figure 4), with the edited condition highlighted. Bars are the mean over the three
teacher pairs for each student; black dots are the three pairs. Colours and type follow Jack Morris's plots: cream
plot area with white grid lines, no top or right spine, ordered navy-slate-yellow palette, bold sentence-case
axis titles with the unit on a lighter second line. Run from the repo root; writes PDFs into the manuscript repo.
"""

import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

EXP = Path("experiments")
OUT = Path("/home/lab/workspace/NAACL_cot_distill_lora_subspace/figures")
L = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
X = [float(l) for l in L]
PAIRS = [("A", "A"), ("B", "B"), ("C", "D")]          # paper name, key in the result files
STU = [("Qwen", "Qwen2.5-0.5B"), ("Llama", "Llama-3.2-1B"), ("OLMo", "OLMo-2-1B")]
NAVY, SLATE, YELLOW = "#1f2c56", "#7b8798", "#f2cf3a"
COL = {"Qwen": NAVY, "Llama": SLATE, "OLMo": YELLOW}
CREAM, REF, RED, HILITE, CHANCE = "#f8f2e7", "#8c8c8c", "#a3242f", "#f3e8bd", "#d4cfc6"
FULL, HALF = 6.9, 3.3

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold",
    "axes.labelsize": 8.5, "axes.labelweight": "bold", "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "legend.title_fontsize": 7.5, "axes.facecolor": CREAM, "axes.edgecolor": "#333333",
    "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "white", "grid.linewidth": 0.9, "axes.axisbelow": True, "xtick.direction": "out",
    "ytick.direction": "out", "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.color": "#333333",
    "ytick.color": "#333333", "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03, "mathtext.fontset": "custom", "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
})

load = lambda n: json.loads((EXP / n).read_text())
think = load("think_rate.json")
resample = load("phi_resample_s012.json")
consensus = load("consensus_pre.json")
base = load("baselines.json")
spread = load("seed_spread.json")
CELLS = [(p, s) for p, _ in PAIRS for s, _ in STU]
cell = lambda p, s: f"{p}/{s}"


def labels(ax, x=None, xunit=None, y=None, yunit=None):
    """Bold axis title in sentence case; the unit goes on a lighter second line, placed against the title."""
    if x:
        ax.set_xlabel(x, fontweight="bold", labelpad=3)
        if xunit:
            ax.annotate(xunit, xy=(0.5, 0), xycoords=ax.xaxis.label, xytext=(0, -1.5), textcoords="offset points",
                        ha="center", va="top", fontsize=7.5, color="#444444")
    if y:
        ax.set_ylabel(y, fontweight="bold", labelpad=12 if yunit else 4)
        if yunit:
            ax.annotate(yunit, xy=(1, 0.5), xycoords=ax.yaxis.label, xytext=(1.5, 0), textcoords="offset points",
                        ha="left", va="center", rotation=90, fontsize=7.5, color="#444444")


def student_handles(lines=False):
    return [Line2D([], [], color=COL[s], marker="o", ms=4, lw=1.4, mec="white", mew=0.6, label=full)
            if lines else Patch(facecolor=COL[s], edgecolor="#333333", lw=0.5, label=full) for s, full in STU]


PAIR_HANDLE = Line2D([], [], ls="none", marker="o", ms=2.6, mfc="white", mec="black", mew=0.6,
                     label="Teacher pairs A, B, C")


def bottom_legend(fig, handles, ncol):
    fig.legend(handles=handles, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, 0.0), handlelength=1.6,
               columnspacing=1.6)


# ---------------------------------------------------------------- Figure 2: R(lambda) against the teacher
def fig2():
    fig, axes = plt.subplots(2, 3, figsize=(FULL, 2.75), sharex=True, sharey="row",
                             gridspec_kw={"height_ratios": [1, 2.4], "hspace": 0.12})
    for j, (p, d) in enumerate(PAIRS):
        top, ax = axes[0, j], axes[1, j]
        rates = [100 * think[d]["rates"][l] for l in L]
        drops = [rates[i] - rates[i + 1] for i in range(5)]
        i = max(range(5), key=lambda k: drops[k])
        sw = (X[i] + X[i + 1]) / 2
        top.fill_between(X, rates, color=CHANCE, lw=0, zorder=1)
        top.plot(X, rates, color="#555555", lw=1.2, marker="s", ms=2.8, zorder=2)
        for a in (top, ax):
            a.axvline(sw, color=RED, ls="--", lw=1.1, zorder=4)
        top.text(sw + 0.03, 104, "teacher stops\nthinking", color=RED, fontsize=6.8, fontweight="bold",
                 ha="left", va="top", linespacing=1.05)
        top.set_title(f"Pair {p}", pad=4)
        top.set_ylim(0, 108)
        top.set_yticks([0, 50, 100])
        for s, full in STU:
            rows = resample[f"{d}/{full}"]["rows"]
            y = [rows[l]["retention"] for l in L]
            e = [rows[l].get("retention_sd", 0) for l in L]
            ax.errorbar(X, y, yerr=e, color=COL[s], marker="o", ms=4, lw=1.4, capsize=1.5, elinewidth=0.8,
                        mec="white", mew=0.6, zorder=3)
        ax.set_xticks(X)
        ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])
        ax.set_ylim(0, 1.06)
        ax.set_xlim(-0.04, 1.04)
        labels(ax, x=r"Interpolation ratio $\lambda$")
    labels(axes[0, 0], y="Teacher", yunit="think rate (%)")
    labels(axes[1, 0], y=r"Direction overlap $R(\lambda)$", yunit=r"(1 = same directions as at $\lambda = 0$)")
    fig.align_ylabels(axes[:, 0])
    fig.tight_layout(w_pad=0.6, h_pad=0.3)
    bottom_legend(fig, student_handles(lines=True), 3)
    fig.savefig(OUT / "fig2_switch.pdf")


# ---------------------------------------------------------------- Figure 3: coverage of the shared region
def fig3():
    fig, ax = plt.subplots(figsize=(HALF, 2.05))
    w = 0.26
    series = [("Pre-switch (held out)", "in_loo", NAVY), ("Post-switch", "out", SLATE), ("Chance", "chance", CHANCE)]
    for k, (name, key, c) in enumerate(series):
        ys = [consensus[f"{d}/{full}"][key] for p, d in PAIRS for s, full in STU]
        ax.bar([i + (k - 1) * w for i in range(9)], ys, width=w, color=c, edgecolor="#333333", lw=0.4, label=name)
    ax.set_yscale("log")
    ax.set_ylim(2e-3, 1)
    ax.set_xticks(range(9))
    ax.set_xticklabels([s for _ in PAIRS for s, _ in STU], fontsize=6.8)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", visible=False)
    for b in (2.5, 5.5):
        ax.axvline(b, color="white", lw=2.5)
    for g, (p, _) in enumerate(PAIRS):
        ax.text(3 * g + 1, -0.11, f"Pair {p}", transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=7.5, fontweight="bold")
    labels(ax, y="Region coverage", yunit="(fraction of the region, log)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=6.8, handlelength=1.0,
              columnspacing=1.0)
    fig.savefig(OUT / "fig3_coverage.pdf")


# ---------------------------------------------------------------- grouped bars for the edits
def bars(ax, conds, values, highlight=(), ref=None):
    """conds: condition names (left to right); values: name -> 9 values in CELLS order.

    One bar per student (mean over the three pairs) with the three pair values as dots.
    """
    w = 0.26
    for g, c in enumerate(conds):
        if c in highlight:
            ax.axvspan(g - 0.48, g + 0.48, color=HILITE, lw=0, zorder=0)
        for k, (s, _) in enumerate(STU):
            vals = [values[c][i] for i, (p, ss) in enumerate(CELLS) if ss == s]
            x = g + (k - 1) * w
            ax.bar(x, st.mean(vals), width=w, color=COL[s], edgecolor="#333333", lw=0.4, zorder=2)
            ax.plot([x] * 3, vals, ls="none", marker="o", ms=2.3, mfc="white", mec="black", mew=0.5, zorder=3)
    if ref is not None:
        ax.axhline(ref, color=REF, ls="--", lw=1.0, zorder=1)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds)
    ax.set_xlim(-0.55, len(conds) - 0.45)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", visible=False)


def fig4():
    conds = [("base0", "$\\lambda = 0$\nadapter"), ("rm_pre", "Shared\nremoved"), ("budget", "Length\nmatched"),
             ("rm_other", "Other\nremoved"), ("gs_energy", "Top\nremoved"), ("rm_scale", "Scaled\ndown")]
    names = [n for _, n in conds]
    mk = {n: [base[cell(p, s)][k]["markers"]["mean"] for p, s in CELLS] for k, n in conds}
    ac = {n: [100 * base[cell(p, s)][k]["accuracy"]["mean"] for p, s in CELLS] for k, n in conds}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL, 2.15))
    bars(a1, names, mk, highlight=[names[1]])
    labels(a1, y="Reflective markers", yunit="(per 1,000 words)")
    bars(a2, names, ac, highlight=[names[1]])
    labels(a2, y="MATH-500 accuracy", yunit="(%)")
    for a in (a1, a2):
        a.tick_params(axis="x", labelsize=7)
    fig.tight_layout(w_pad=1.5)
    bottom_legend(fig, student_handles() + [PAIR_HANDLE], 4)
    fig.savefig(OUT / "fig4_remove.pdf")


def fig5():
    e = lambda p, s, k, f: spread[cell(p, s)]["edits"][k][f]["mean"]
    b = lambda p, s, k: spread[cell(p, s)]["base"][k]["mean"]
    t2 = lambda p, s, k: spread[cell(p, s)]["table2"][k]["mean"]
    graft = [("graft2", "Graft\nshared"), ("graft_other2", "Graft\nother"), ("graft_whole2", "Graft\nwhole")]
    l1 = "$\\lambda = 1$\nadapter"
    gm = {n: [e(p, s, k, "head_markers_per_1k") for p, s in CELLS] for k, n in graft}
    gm[l1] = [b(p, s, "base1_head_markers_per_1k") for p, s in CELLS]
    gc = {n: [100 * e(p, s, k, "completed") for p, s in CELLS] for k, n in graft}
    gc[l1] = [100 * (1 - t2(p, s, "trunc1")) for p, s in CELLS]
    head0 = st.mean(b(p, s, "base0_head_markers_per_1k") for p, s in CELLS)
    done0 = st.mean(100 * (1 - t2(p, s, "trunc0")) for p, s in CELLS)
    amps = [("ampn1", "$\\gamma = 1$"), ("ampn4", "$\\gamma = 4$"), ("ampV", "All in\nregion")]
    am = {n: [100 * e(p, s, k, "markers_per_1k") / spread[cell(p, s)]["table1"]["0.0"]["mean"] for p, s in CELLS]
          for k, n in amps}
    gnames = [l1] + [n for _, n in graft]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(FULL, 2.15), gridspec_kw={"width_ratios": [1.25, 1.25, 1]})
    bars(a1, gnames, gm, highlight=[gnames[1]], ref=head0)
    labels(a1, y="Reflective markers", yunit="(per 1,000 words, early window)")
    a1.set_title("Graft: thinking starts")
    bars(a2, gnames, gc, highlight=[gnames[1]], ref=done0)
    labels(a2, y="Completed responses", yunit="(%)")
    a2.set_title("Graft: never finishes")
    bars(a3, [n for _, n in amps], am, ref=100)
    labels(a3, y="Reflective markers", yunit=r"(% of the $\lambda = 0$ adapter)")
    a3.set_title("Amplify: no more thinking")
    for a in (a1, a2, a3):
        a.tick_params(axis="x", labelsize=7)
    ref = Line2D([], [], color=REF, ls="--", lw=1.0, label=r"$\lambda = 0$ adapter")
    fig.tight_layout(w_pad=1.2)
    bottom_legend(fig, student_handles() + [PAIR_HANDLE, ref], 5)
    fig.savefig(OUT / "fig5_graft_amplify.pdf")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig2(); fig3(); fig4(); fig5()
    print("written to", OUT)
