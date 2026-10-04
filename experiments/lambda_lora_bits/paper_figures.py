#!/usr/bin/env python3
"""Result figures for the manuscript (Figures 2 and 3), drawn from the experiment JSONs.

Style copies the plots in Morris et al. ("How much do language models memorize?", Figures 3-10): cream plot area
with a visible grey grid, black left and bottom spines only, thin lines with small dot markers, small tick labels,
a bold axis title with the unit in a lighter line below it, a white boxed legend with a bold title inside the
plot, and a thick dashed black reference line labelled in bold. Panels carry no titles beyond a short label.
Figure 2 (where): per teacher pair, the teacher think rate, R(lambda) and the shared-region coverage share one
lambda axis (stacked rows as in Wu et al. 2026, Figure 3).
Figure 3 (what): the edits as grouped bars per cell, the unedited adapter next to its control edit and its
shared-region edit, as in Arditi et al. (2024, Figures 1 and 3), which put models on the x axis and conditions as
bars. Bars are seed means; spreads stay in the appendix tables.
Figures are drawn at the printed width (ACL text width, 6.3 in), so font sizes are the printed sizes.
Run from the repo root; writes PDFs into the manuscript repo's figures/ folder.
"""

import json
import os
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

EXP = Path("experiments")
OUT = Path(os.environ.get("FIG_OUT", "/home/lab/workspace/NAACL_cot_distill_lora_subspace/figures"))
L = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
X = [float(l) for l in L]
PAIRS = [("A", "A"), ("B", "B"), ("C", "D")]          # paper name, key in the result files
STU = [("Qwen", "Qwen2.5-0.5B"), ("Llama", "Llama-3.2-1B"), ("OLMo", "OLMo-2-1B")]
NAVY, SLATE, YELLOW, RED = "#2f4a6d", "#3f7a5a", "#e7a33e", "#c8372d"   # denim, scarf green, mustard, Carp red
COL = {"Qwen": NAVY, "Llama": SLATE, "OLMo": YELLOW}
CREAM, GRID = "#f6efe4", "#d8cdbd"
# Figure 1 colours (Contextual Document Embeddings palette), reused so the edits read the same in Figures 1 and 3
F1_LAVENDER, F1_PINK, F1_SHARED, F1_SHARED_LIGHT = "#d9cef9", "#fba9cb", "#9c86f0", "#b7a6f4"
F1_SWITCH = "#f25c93"                                  # the dashed "teacher stops thinking" line in Figure 1
# Figure 3 roles, in the ordered navy-gray-yellow palette of Morris et al. (memorization paper), same in every panel
F3_SHARED, F3_SHARED_SOFT = "#1d2b4f", "#3f5490"      # edit of the shared region (gamma = 4, gamma = 1)
F3_BASE, F3_CONTROL = "#7d7d7d", "#f2e04e"            # unedited adapter, same edit outside the region
WIDTH = 6.3

plt.rcParams.update({
    "font.family": "Arial", "font.size": 7, "axes.labelsize": 7.5, "axes.labelweight": "bold",
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.2, "legend.title_fontsize": 6.2,
    "axes.facecolor": CREAM, "axes.edgecolor": "black", "axes.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "legend.frameon": True, "legend.framealpha": 1.0, "legend.facecolor": "white",
    "legend.edgecolor": "#9a9a9a", "legend.fancybox": False, "legend.borderpad": 0.5,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "mathtext.fontset": "custom",
    "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
})

load = lambda n: json.loads((EXP / n).read_text())
think = load("think_rate.json")
resample = load("phi_resample_s012.json")
coverage = load("coverage_by_lambda.json")
base = load("baselines.json")
spread = load("seed_spread.json")
CELLS = [(p, s) for p, _ in PAIRS for s, _ in STU]
cell = lambda p, s: f"{p}/{s}"


def labels(ax, x=None, xunit=None, y=None, yunit=None):
    """Bold axis title; the unit goes on a lighter line below it (left of it for the y axis)."""
    if x:
        ax.set_xlabel(x, labelpad=2)
        if xunit:
            ax.annotate(xunit, xy=(0.5, 0), xycoords=ax.xaxis.label, xytext=(0, -1), textcoords="offset points",
                        ha="center", va="top", fontsize=6.2, color="#555555")
    if y:
        ax.set_ylabel(y, labelpad=10 if yunit else 3)
        if yunit:
            ax.annotate(yunit, xy=(1, 0.5), xycoords=ax.yaxis.label, xytext=(1, 0), textcoords="offset points",
                        ha="left", va="center", rotation=90, fontsize=6.2, color="#555555")


def panel_legend(ax, letter, title):
    """Panel label and title on top, then a one-row boxed legend just above the plot area."""
    ax.text(0, 1.2, f"({letter}) {title}", transform=ax.transAxes, fontsize=7.2, fontweight="bold", va="bottom")
    lg = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, handlelength=1.4, columnspacing=1.0,
                   fontsize=5.9, borderpad=0.35)
    lg.get_frame().set_linewidth(0.6)


def legend(ax, handles, title, loc, ncol=1):
    lg = ax.legend(handles=handles, title=title, loc=loc, handlelength=1.8, ncol=ncol, columnspacing=1.2)
    lg.get_title().set_fontweight("bold")
    lg.get_frame().set_linewidth(0.6)


def switch_line(ax, x):
    ax.axvline(x, color=F1_SWITCH, ls="--", lw=1.4, zorder=4)


# ---------------------------------------------------------------- Figure 2: where the adapter switches
def fig2():
    fig, axes = plt.subplots(3, 3, figsize=(WIDTH, 3.5), sharex=True, sharey="row",
                             gridspec_kw={"height_ratios": [0.75, 1.3, 1.3], "hspace": 0.12, "wspace": 0.08})
    for j, (p, d) in enumerate(PAIRS):
        top, mid, bot = axes[:, j]
        rates = [100 * think[d]["rates"][l] for l in L]
        i = max(range(5), key=lambda k: rates[k] - rates[k + 1])
        sw = (X[i] + X[i + 1]) / 2
        top.plot(X, rates, color="#333333", lw=1.1, marker="o", ms=2.2)
        top.set_ylim(-5, 105); top.set_yticks([0, 50, 100])
        top.text(0.03, 0.08, f"Pair {p}", transform=top.transAxes, fontsize=7, fontweight="bold")
        for s, full in STU:
            rows = resample[f"{d}/{full}"]["rows"]
            mid.plot(X, [rows[l]["retention"] for l in L], color=COL[s], lw=1.1, marker="o", ms=2.2)
            cv = coverage[f"{d}/{full}"]["coverage"]
            bot.plot(X, [cv[l] for l in L], color=COL[s], lw=1.1, marker="o", ms=2.2)
        chance = st.mean(coverage[f"{d}/{full}"]["chance"] for _, full in STU)
        bot.axhline(chance, color="#555555", ls=":", lw=1.0)
        if j == 0:
            SW_A = sw
        if j == 1:
            CHANCE_B = chance
        for a in (top, mid, bot):
            switch_line(a, sw)
        mid.set_ylim(0, 1.05); bot.set_ylim(0, 0.65)
        bot.set_xticks(X); bot.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"]); bot.set_xlim(-0.03, 1.03)
        labels(bot, x=r"Interpolation ratio $\lambda$")
    labels(axes[0, 0], y="Think rate", yunit="(teacher, %)")
    labels(axes[1, 0], y=r"Overlap ratio $R(\lambda)$", yunit=r"(1 = as at $\lambda = 0$)")
    labels(axes[2, 0], y="Region coverage", yunit="(share of the shared region)")
    fig.align_ylabels(axes[:, 0])
    handles = [Line2D([], [], color=COL[s], lw=1.1, marker="o", ms=2.2, label=full) for s, full in STU]
    legend(axes[2, 0], handles, "Student", "upper right")
    # reference lines are labelled once, on the line itself (Morris style)
    a_top = axes[0, 0]
    a_top.text(SW_A + 0.03, 0.95, "Teacher stops thinking", transform=a_top.get_xaxis_transform(), fontsize=6.2,
               fontweight="bold", color=F1_SWITCH, ha="left", va="top")
    b_bot = axes[2, 1]
    b_bot.text(1.0, CHANCE_B + 0.012, "Random adapter", fontsize=6.0, color="#555555", ha="right", va="bottom")
    fig.savefig(OUT / "fig2_switch.pdf")


# ---------------------------------------------------------------- Figure 3: what the shared region does
def fig3():
    def bars(ax, series, ylabel, yunit):
        """series: list of (label, colour, values over the 9 cells); one group of bars per cell."""
        n, w = len(series), 0.8 / len(series)
        for k, (name, c, vals) in enumerate(series):
            ax.bar([i + (k - (n - 1) / 2) * w for i in range(9)], vals, width=w, color=c, edgecolor="black",
                   lw=0.4, label=name, zorder=2)
        ax.set_xticks(range(9))
        ax.set_xticklabels([s for _ in PAIRS for s, _ in STU], fontsize=5.8)
        ax.tick_params(axis="x", length=0, pad=2)
        top = max(max(v) for _, _, v in series)
        ax.set_ylim(0, top * 1.06)
        ax.set_xlim(-0.5, 8.5)
        ax.grid(axis="x", visible=False)
        for b in (2.5, 5.5):
            ax.axvline(b, color=GRID, lw=0.8)
        for g, (p, _) in enumerate(PAIRS):
            ax.text(3 * g + 1, -0.13, f"Pair {p}", transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=6.5, fontweight="bold")
        labels(ax, y=ylabel, yunit=yunit)

    b0 = lambda k, f: [base[cell(p, s)][k][f]["mean"] for p, s in CELLS]
    ed = lambda k, f: [spread[cell(p, s)]["edits"][k][f]["mean"] for p, s in CELLS]
    sb = lambda k: [spread[cell(p, s)]["base"][k]["mean"] for p, s in CELLS]
    t2 = lambda k: [spread[cell(p, s)]["table2"][k]["mean"] for p, s in CELLS]
    pct = lambda v: [100 * x for x in v]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 3.6), gridspec_kw={"hspace": 0.85, "wspace": 0.25})
    (a, b), (c, d) = axes
    bars(a, [(r"$\lambda = 0$ adapter", F3_BASE, b0("base0", "markers")),
             ("Other removed", F3_CONTROL, b0("rm_other", "markers")),
             ("Shared removed", F3_SHARED, b0("rm_pre", "markers"))], "Reflective markers", "(per 1,000 words)")
    panel_legend(a, "a", r"Remove from the $\lambda = 0$ adapter")
    bars(b, [(r"$\lambda = 1$ adapter", F3_BASE, sb("base1_head_markers_per_1k")),
             ("Graft other", F3_CONTROL, ed("graft_other2", "head_markers_per_1k")),
             ("Graft shared", F3_SHARED, ed("graft2", "head_markers_per_1k"))],
         "Reflective markers", "(per 1,000 words, early)")
    panel_legend(b, "b", r"Graft into the $\lambda = 1$ adapter ($\beta = 2$)")
    bars(c, [(r"$\lambda = 1$ adapter", F3_BASE, pct([1 - t for t in t2("trunc1")])),
             ("Graft other", F3_CONTROL, pct(ed("graft_other2", "completed"))),
             ("Graft shared", F3_SHARED, pct(ed("graft2", "completed")))], "Completed responses", "(%)")
    c.set_yticks([0, 25, 50, 75, 100])
    panel_legend(c, "c", r"Graft into the $\lambda = 1$ adapter ($\beta = 2$)")
    bars(d, [(r"$\lambda = 0$ adapter", F3_BASE, [spread[cell(p, s)]["table1"]["0.0"]["mean"] for p, s in CELLS]),
             (r"Amplified, $\gamma = 1$", F3_SHARED_SOFT, ed("ampn1", "markers_per_1k")),
             (r"Amplified, $\gamma = 4$", F3_SHARED, ed("ampn4", "markers_per_1k"))],
         "Reflective markers", "(per 1,000 words)")
    panel_legend(d, "d", r"Amplify within the $\lambda = 0$ adapter (norm fixed)")
    fig.savefig(OUT / "fig3_states.pdf")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig2(); fig3()
    print("written to", OUT)
