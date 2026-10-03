#!/usr/bin/env python3
"""Result figures for the manuscript (Figures 2 and 3), drawn from the experiment JSONs. One figure per claim.

Figure 2 (where): for each teacher pair, three rows share one lambda axis and one dashed line at the teacher's
largest drop in think rate: the teacher think rate, the students' direction overlap R(lambda), and the students'
coverage of the shared region (experiments/coverage_by_lambda.json). The stacked think-rate row follows Wu et al.
(2026, model interpolation, Figure 3).
Figure 3 (what): one panel per edit (remove, graft markers, graft completion, amplify); bars are nine-cell means
with the value printed, as in Venhoff et al. (2026, Figure 4); per-cell values and spreads stay in the appendix tables.
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
from matplotlib.patches import Patch

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
    """Claims 2-3: one panel per edit, bars are nine-cell means with their values printed (as in Venhoff et al.
    2026, Figure 4); the shared-region edit is yellow, controls grey, unedited adapters navy (lambda = 0) / red."""
    m0 = lambda k, f: st.mean(base[cell(p, s)][k][f]["mean"] for p, s in CELLS)
    me = lambda k, f: st.mean(spread[cell(p, s)]["edits"][k][f]["mean"] for p, s in CELLS)
    sb = lambda k: st.mean(spread[cell(p, s)]["base"][k]["mean"] for p, s in CELLS)
    t2 = lambda k: st.mean(spread[cell(p, s)]["table2"][k]["mean"] for p, s in CELLS)
    amp = lambda k: st.mean(100 * spread[cell(p, s)]["edits"][k]["markers_per_1k"]["mean"]
                            / spread[cell(p, s)]["table1"]["0.0"]["mean"] for p, s in CELLS)
    l0, l1 = "", ""
    panels = [
        ("Remove: thinking off", None, "Reflective markers", "(per 1,000 words)", None, "{:.1f}",
         [(l0, m0("base0", "markers"), LAM0), ("Shared", m0("rm_pre", "markers"), SHARED_C),
          ("Other", m0("rm_other", "markers"), CTRL), ("Top", m0("gs_energy", "markers"), CTRL),
          ("Scaled", m0("rm_scale", "markers"), CTRL)]),
        ("Graft: thinking on", None, "Reflective markers", "(per 1,000 words, early window)",
         sb("base0_head_markers_per_1k"), "{:.1f}",
         [(l1, sb("base1_head_markers_per_1k"), LAM1), ("Shared", me("graft2", "head_markers_per_1k"), SHARED_C),
          ("Other", me("graft_other2", "head_markers_per_1k"), CTRL),
          ("Whole", me("graft_whole2", "head_markers_per_1k"), CTRL)]),
        ("Graft: never finishes", None, "Completed responses", "(%)", 100 * (1 - t2("trunc0")), "{:.0f}",
         [(l1, 100 * (1 - t2("trunc1")), LAM1), ("Shared", 100 * me("graft2", "completed"), SHARED_C),
          ("Other", 100 * me("graft_other2", "completed"), CTRL),
          ("Whole", 100 * me("graft_whole2", "completed"), CTRL)]),
        ("Amplify: no more thinking", None, "Reflective markers", "(% of the \u03bb = 0 adapter)", None, "{:.0f}",
         [(l0, 100.0, LAM0), ("\u03b3 = 1", amp("ampn1"), SHARED_C), ("\u03b3 = 4", amp("ampn4"), SHARED_C),
          ("All", amp("ampV"), SHARED_C)]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 2.2), gridspec_kw={"width_ratios": [5, 4, 4, 4]})
    for ax, (title, xl, y, yunit, ref, fmt, bars_) in zip(axes, panels):
        xs = range(len(bars_))
        vals = [v for _, v, _ in bars_]
        ax.bar(xs, vals, width=0.62, color=[c for _, _, c in bars_], edgecolor="#333333", lw=0.5, zorder=2)
        top = max(vals + ([ref] if ref else [])) * 1.18
        for x, v in zip(xs, vals):
            ax.text(x, v + top * 0.015, fmt.format(v), ha="center", va="bottom", fontsize=7, fontweight="bold")
        if ref is not None:
            ax.axhline(ref, color=REF, ls="--", lw=1.0, zorder=1)
        ax.set_ylim(0, top)
        ax.set_xticks(list(xs)); ax.set_xticklabels([n for n, _, _ in bars_], fontsize=7)
        ax.set_xlim(-0.6, len(bars_) - 0.4)
        ax.tick_params(axis="x", length=0); ax.grid(axis="x", visible=False)
        ax.set_title(title, fontsize=8)
        ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.grid(False)
        labels(ax, x=y, xunit=yunit)
    axes[3].axhline(100, color=REF, ls="--", lw=1.0, zorder=1)
    fig.tight_layout(w_pad=1.2)
    handles = [Patch(facecolor=LAM0, edgecolor="#333333", lw=0.5, label="\u03bb = 0 adapter"),
               Patch(facecolor=LAM1, edgecolor="#333333", lw=0.5, label="\u03bb = 1 adapter"),
               Patch(facecolor=SHARED_C, edgecolor="#333333", lw=0.5, label="Shared-region edit"),
               Patch(facecolor=CTRL, edgecolor="#333333", lw=0.5, label="Control edit"),
               Line2D([], [], color=REF, ls="--", lw=1.0, label="\u03bb = 0 level")]
    bottom_legend(fig, handles, 5)
    fig.savefig(OUT / "fig3_states.pdf")

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig2(); fig3()
    print("written to", OUT)
