#!/usr/bin/env python3
"""Result figures for the manuscript (Figures 2 and 3), drawn from the experiment JSONs. One figure per claim.

Figure 2 (where): for each teacher pair, three rows share one lambda axis and one dashed line at the teacher's
largest drop in think rate: the teacher think rate, the students' direction overlap R(lambda), and the students'
coverage of the shared region (experiments/coverage_by_lambda.json). The stacked think-rate row follows Wu et al.
(2026, model interpolation, Figure 3).
Figure 3 (what): each edit as a move between the two behaviours, the lambda = 0 adapter (thinks, rarely finishes)
and the lambda = 1 adapter (short, finishes), on reflective markers x completed responses. Points are nine-cell
means; per-cell values and spreads are in the appendix tables, not in the figures.
Colours and type follow Jack Morris's plots: cream plot area with white grid lines, no top or right spine, ordered
navy-slate-yellow palette, bold sentence-case axis titles with the unit on a lighter second line.
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
CREAM, REF, RED, CHANCE = "#f8f2e7", "#8c8c8c", "#a3242f", "#d4cfc6"
FULL = 6.9

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
coverage = load("coverage_by_lambda.json")
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


def student_handles():
    return [Line2D([], [], color=COL[s], marker="o", ms=4, lw=1.4, mec="white", mew=0.6, label=full)
            for s, full in STU]


def bottom_legend(fig, handles, ncol):
    fig.legend(handles=handles, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, 0.0), handlelength=1.6,
               columnspacing=1.6)


LAM0, LAM1, SHARED_C, CTRL = NAVY, RED, YELLOW, "#9aa3b0"


# ---------------------------------------------------------------- Figure 2: where the adapter switches


def fig2():
    """Claim 1: teacher stops thinking -> adapters change direction -> shared region empties, at one lambda."""
    fig, axes = plt.subplots(3, 3, figsize=(FULL, 3.6), sharex=True, sharey="row",
                             gridspec_kw={"height_ratios": [0.8, 1.5, 1.5], "hspace": 0.14})
    for j, (p, d) in enumerate(PAIRS):
        top, mid, bot = axes[:, j]
        rates = [100 * think[d]["rates"][l] for l in L]
        i = max(range(5), key=lambda k: rates[k] - rates[k + 1])
        sw = (X[i] + X[i + 1]) / 2
        top.fill_between(X, rates, color=CHANCE, lw=0)
        top.plot(X, rates, color="#555555", lw=1.3, marker="s", ms=2.8)
        top.set_ylim(0, 110); top.set_yticks([0, 50, 100])
        top.set_title(f"Pair {p}", pad=4)
        top.text(sw + 0.03, 106, "teacher stops thinking", color=RED, fontsize=6.8, fontweight="bold", va="top")
        for s, full in STU:
            rows = resample[f"{d}/{full}"]["rows"]
            mid.plot(X, [rows[l]["retention"] for l in L], color=COL[s], lw=1.5, marker="o", ms=3.8, mec="white",
                     mew=0.6)
            cv = coverage[f"{d}/{full}"]["coverage"]
            bot.plot(X, [cv[l] for l in L], color=COL[s], lw=1.5, marker="o", ms=3.8, mec="white", mew=0.6)
        chance = st.mean(coverage[f"{d}/{full}"]["chance"] for _, full in STU)
        bot.axhline(chance, color=REF, ls="--", lw=1.0)
        if j == 1:
            bot.text(-0.02, chance + 0.015, "chance", color="#555555", fontsize=6.8, ha="left", va="bottom")
        for a in (top, mid, bot):
            a.axvline(sw, color=RED, ls="--", lw=1.1)
        mid.set_ylim(0, 1.06); bot.set_ylim(0, 0.65)
        bot.set_xticks(X); bot.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"]); bot.set_xlim(-0.04, 1.04)
        labels(bot, x=r"Interpolation ratio $\lambda$")
    labels(axes[0, 0], y="Teacher", yunit="think rate (%)")
    labels(axes[1, 0], y="Direction overlap", yunit=r"$R(\lambda)$")
    labels(axes[2, 0], y="Shared region", yunit="coverage")
    fig.align_ylabels(axes[:, 0])
    fig.tight_layout(w_pad=0.6, h_pad=0.25)
    bottom_legend(fig, student_handles(), 3)
    fig.savefig(OUT / "fig2_switch.pdf")


# ---------------------------------------------------------------- Figure 3: what the shared region does
def fig3():
    """Claims 2-3: edits move an adapter between the two behaviours; the region acts as a switch, not a dial."""
    m0 = lambda k, f: st.mean(base[cell(p, s)][k][f]["mean"] for p, s in CELLS)
    me = lambda k, f: st.mean(spread[cell(p, s)]["edits"][k][f]["mean"] for p, s in CELLS)
    pt = {"l0": (m0("base0", "markers"), 100 * (1 - m0("base0", "truncated"))),
          "l1": (m0("base1", "markers"), 100 * (1 - m0("base1", "truncated")))}
    for k in ("rm_pre", "rm_other", "gs_energy", "rm_scale"):
        pt[k] = (m0(k, "markers"), 100 * (1 - m0(k, "truncated")))
    for k in ("graft2", "graft_other2", "graft_whole2", "ampn1", "ampn4", "ampV"):
        pt[k] = (me(k, "markers_per_1k"), 100 * me(k, "completed"))
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.5), sharey=True)
    # (key, label, shared-region edit?, label offset in points, ha, source adapter)
    panels = [
        ("Remove from the \u03bb = 0 adapter",
         [("rm_pre", "Shared removed", True, (-8, 0), "right", "l0"),
          ("gs_energy", "Top removed", False, (7, 0), "left", "l0"),
          ("rm_scale", "Scaled down", False, (7, 0), "left", "l0"),
          ("rm_other", "Other removed", False, (-7, 3), "right", "l0")],
         [("l0", (0, -12)), ("l1", (0, 11))]),
        ("Graft into \u03bb = 1 (\u03b2 = 2), amplify \u03bb = 0",
         [("graft2", "Graft shared", True, (-8, 0), "right", "l1"),
          ("graft_whole2", "Graft whole", False, (0, -11), "center", "l1"),
          ("graft_other2", "Graft other", False, (7, 0), "left", "l1"),
          ("ampn1", None, True, None, None, "l0"), ("ampn4", None, True, None, None, "l0"),
          ("ampV", "Amplify (\u03b3 = 1, 4, all)", True, (-8, -2), "right", "l0")],
         [("l0", (0, 11)), ("l1", (0, 11))]),
    ]
    names = {"l0": ("\u03bb = 0 adapter", LAM0), "l1": ("\u03bb = 1 adapter", LAM1)}
    for ax, (title, edits, anchors) in zip(axes, panels):
        ax.set_xscale("log"); ax.set_xlim(0.12, 80); ax.set_ylim(-12, 108)
        for k, name, shared, off, ha, src in edits:
            (x0, y0), (x1, y1) = pt[src], pt[k]
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color="black" if shared else CTRL, lw=1.5 if shared else 1.0,
                                        shrinkA=6, shrinkB=5, mutation_scale=9), zorder=2)
            ax.plot(x1, y1, "o", ms=7 if shared else 5.5, color=SHARED_C if shared else "white",
                    mec="black" if shared else CTRL, mew=1.0 if shared else 1.2, zorder=4)
            if name:
                ax.annotate(name, (x1, y1), xytext=off, textcoords="offset points", ha=ha, va="center",
                            fontsize=7.5, fontweight="bold" if shared else "normal",
                            color="black" if shared else "#555555")
        for k, off in anchors:
            name, c = names[k]
            ax.plot(*pt[k], "s", ms=8, color=c, mec="black", mew=0.8, zorder=5)
            ax.annotate(name, pt[k], xytext=off, textcoords="offset points", ha="center", va="center",
                        fontsize=7.5, fontweight="bold", color=c)
        ax.set_title(title)
        labels(ax, x="Reflective markers", xunit="(per 1,000 words, log)")
    labels(axes[0], y="Completed responses", yunit="(%)")
    fig.tight_layout(w_pad=1.0)
    fig.savefig(OUT / "fig3_states.pdf")

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig2(); fig3()
    print("written to", OUT)
