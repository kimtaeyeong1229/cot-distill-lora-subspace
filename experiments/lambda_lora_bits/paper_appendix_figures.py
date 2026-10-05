#!/usr/bin/env python3
"""Appendix figures for the manuscript (drafts for the appendix compaction), drawn from the same JSONs as the tables.

Style, palette and helpers are copied from paper_figures.py (Morris et al. style: cream plot area, grey grid, black
left and bottom spines, bold axis title with a lighter unit line, white boxed legends, denim / green / mustard for the
three students, pink dashed line for the teacher switch). Bars and lines are seed means; no seed or cell scatter.

  appfig_lambda   replaces Tables 2, 3 and the B(lambda) block of Table 16:
                  (a) reflective markers of students and teacher, (b) R and R_sym over teacher think-rate bars,
                  (c) data-only baseline B(lambda), per pair, over lambda.
  appfig_ood      replaces Tables 13 and 14: GSM8K and GPQA-Diamond markers, GSM8K accuracy and truncation, per cell.
  appfig_off      replaces Table 15 (per-cell ways to turn thinking off): (a) nine-cell means, one labelled point per
                  method (a comparison display; flagged for the user's decision), (b) per-cell accuracy of the options
                  without further training, (c) per-cell accuracy of the options that need a short-solution adapter.

Run from the repo root on the host (Arial is installed there):
  FIG_OUT=<dir> python3 experiments/lambda_lora_bits/paper_appendix_figures.py
Writes PDFs plus a JPG preview (longest side 1600 px) into FIG_OUT; the manuscript uses the PDFs.
"""

import json
import os
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

EXP = Path("experiments")
OUT = Path(os.environ.get("FIG_OUT", "appfig_out"))
L = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
X = [float(l) for l in L]
PAIRS = [("A", "A"), ("B", "B"), ("C", "D")]          # paper name, key in the result files
STU = [("Qwen", "Qwen2.5-0.5B"), ("Llama", "Llama-3.2-1B"), ("OLMo", "OLMo-2-1B")]
UNTR = {"Qwen": "Qwen2.5-0.5B-Instruct", "Llama": "Llama-3.2-1B-Instruct", "OLMo": "OLMo-2-1B-Instruct"}
NAVY, SLATE, YELLOW = "#2f4a6d", "#3f7a5a", "#e7a33e"   # denim, scarf green, mustard
COL = {"Qwen": NAVY, "Llama": SLATE, "OLMo": YELLOW}
CREAM, GRID = "#f6efe4", "#d8cdbd"
F1_LAVENDER, F1_SWITCH = "#d9cef9", "#f25c93"
F3_SHARED, F3_SHARED_SOFT = "#1d2b4f", "#3f5490"
F3_BASE, F3_CONTROL = "#7d7d7d", "#f2e04e"
LIGHT = "#c9c9c9"                                      # second unedited adapter (lambda = 1) next to F3_BASE
THINK_BAR = "#d9ccb8"                                  # teacher think-rate bars behind the lines
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
tdens = load("teacher_density.json")
resample = load("phi_resample_s012.json")
rsym = load("symmetric_R.json")
spread = load("seed_spread.json")
base = load("baselines.json")
untrained = load("base_students.json")
ood = load("ood.json")
review = load("review_stats.json")
CELLS = [(p, s) for p, _ in PAIRS for s, _ in STU]
cell = lambda p, s: f"{p}/{s}"
full = dict(STU)
key = dict(PAIRS)


# ---------------------------------------------------------------- helpers copied from paper_figures.py
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


def panel_legend(ax, letter, title, ncol=3):
    """Panel label and title on top, then a one-row boxed legend just above the plot area."""
    ax.text(0, 1.2, f"({letter}) {title}", transform=ax.transAxes, fontsize=7.2, fontweight="bold", va="bottom")
    lg = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=ncol, handlelength=1.4, columnspacing=1.0,
                   fontsize=5.9, borderpad=0.35)
    lg.get_frame().set_linewidth(0.6)


def legend(ax, handles, title, loc, ncol=1, **kw):
    lg = ax.legend(handles=handles, title=title, loc=loc, handlelength=1.8, ncol=ncol, columnspacing=1.2, **kw)
    if title:
        lg.get_title().set_fontweight("bold")
    lg.get_frame().set_linewidth(0.6)
    return lg


def switch_line(ax, x):
    ax.axvline(x, color=F1_SWITCH, ls="--", lw=1.4, zorder=4)


def bars(ax, series, ylabel, yunit, top=None, fs=5.8):
    """series: list of (label, colour, values over the 9 cells); one group of bars per cell (as Figure 3)."""
    n, w = len(series), 0.8 / len(series)
    for k, (name, c, vals) in enumerate(series):
        ax.bar([i + (k - (n - 1) / 2) * w for i in range(9)], vals, width=w, color=c, edgecolor="black",
               lw=0.4, label=name, zorder=2)
    ax.set_xticks(range(9))
    ax.set_xticklabels([s for _ in PAIRS for s, _ in STU], fontsize=fs)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_ylim(0, top if top else max(max(v) for _, _, v in series) * 1.06)
    ax.set_xlim(-0.5, 8.5)
    ax.grid(axis="x", visible=False)
    for b in (2.5, 5.5):
        ax.axvline(b, color=GRID, lw=0.8)
    for g, (p, _) in enumerate(PAIRS):
        ax.text(3 * g + 1, -0.13, f"Pair {p}", transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=6.5, fontweight="bold")
    labels(ax, y=ylabel, yunit=yunit)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    png = OUT / f"{name}.png"
    fig.savefig(png, dpi=300)
    plt.close(fig)
    from PIL import Image                                # preview: JPG, longest side at most 1600 px
    im = Image.open(png).convert("RGB")
    im.thumbnail((1600, 1600), Image.LANCZOS)
    im.save(OUT / f"{name}.jpg", quality=92)
    png.unlink()


def switch_x(d):
    rates = [think[d]["rates"][l] for l in L]
    i = max(range(5), key=lambda k: rates[k] - rates[k + 1])
    return (X[i] + X[i + 1]) / 2


# ---------------------------------------------------------------- Figure: the lambda sweep (Tables 2, 3, 16 left)
def fig_lambda():
    fig, axes = plt.subplots(3, 3, figsize=(WIDTH, 3.6), sharex=True, sharey="row",
                             gridspec_kw={"height_ratios": [1.15, 1.25, 1.0], "hspace": 0.14, "wspace": 0.2})
    fig.subplots_adjust(top=0.93)
    twins = []
    for j, (p, d) in enumerate(PAIRS):
        a, b, c = axes[:, j]
        sw = switch_x(d)
        # (a) reflective markers: students (seed means) and the teacher's solutions
        for s, f in STU:
            t1 = spread[cell(p, s)]["table1"]
            a.plot(X, [t1[l]["mean"] for l in L], color=COL[s], lw=1.1, marker="o", ms=2.2, zorder=3)
        tm = tdens[p]["teacher"]
        a.plot(X, [tm[l] for l in L], color="#222222", lw=1.1, ls=(0, (1.2, 1.0)), marker="s", ms=2.2, zorder=3)
        a.text(0.97, 0.93, f"Pair {p}", transform=a.transAxes, fontsize=7, fontweight="bold", ha="right", va="top")
        # (b) think-rate bars (read on the right axis) under R (solid) and R_sym (dashed)
        rates = [think[d]["rates"][l] for l in L]
        b.bar(X, rates, width=0.11, color=THINK_BAR, edgecolor="none", zorder=1)
        for s, f in STU:
            rows = resample[f"{d}/{f}"]["rows"]
            b.plot(X, [rows[l]["retention"] for l in L], color=COL[s], lw=1.1, marker="o", ms=2.2, zorder=3)
            b.plot(X, [rsym[f"{d}/{f}"]["R_sym"][l] for l in L], color=COL[s], lw=1.0, ls="--", zorder=3)
        tw = b.twinx()
        tw.set_ylim(0, 105 / 100 * 100)
        tw.set_yticks([0, 50, 100])
        tw.spines["right"].set_visible(j == 2)
        tw.spines["right"].set_color("#8a7d6b")
        tw.tick_params(axis="y", colors="#6b604f", labelsize=6.2, right=(j == 2), labelright=(j == 2))
        tw.grid(False)
        twins.append(tw)
        # (c) data-only baseline B(lambda), the denominator of R
        for s, f in STU:
            rows = resample[f"{d}/{f}"]["rows"]
            c.plot(X, [rows[l]["baseline"] for l in L], color=COL[s], lw=1.1, marker="o", ms=2.2, zorder=3)
        ch = st.mean(resample[f"{d}/{f}"]["chance"] for _, f in STU)
        c.axhline(ch, color="#555555", ls=":", lw=1.0)
        for ax in (a, b, c):
            switch_line(ax, sw)
            ax.tick_params(labelleft=True)
        b.set_ylim(0, 1.05); c.set_ylim(0, 0.75)
        c.set_xticks(X); c.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"]); c.set_xlim(-0.08, 1.08)
        labels(c, x=r"Interpolation ratio $\lambda$")
        if j == 1:
            c.text(1.06, ch + 0.015, "Random adapter", fontsize=6.0, color="#555555", ha="right", va="bottom")
    twins[2].set_ylabel("Think rate", fontsize=7, fontweight="bold", color="#6b604f", labelpad=2)
    twins[2].annotate("(teacher, %)", xy=(1, 0.5), xycoords=twins[2].yaxis.label, xytext=(1, 0),
                      textcoords="offset points", ha="left", va="center", rotation=90, fontsize=6.2, color="#8a7d6b")
    labels(axes[0, 0], y="Reflective markers", yunit="(per 1,000 words)")
    labels(axes[1, 0], y=r"Overlap ratio", yunit=r"(1 = as at $\lambda = 0$)")
    labels(axes[2, 0], y=r"Data-only $B(\lambda)$", yunit="(overlap of data halves)")
    fig.align_ylabels(axes[:, 0])
    a_top = axes[0, 0]
    a_top.text(switch_x("A") + 0.04, 0.62, "Teacher stops thinking", transform=a_top.get_xaxis_transform(),
               fontsize=6.2, fontweight="bold", color=F1_SWITCH, ha="left", va="top")
    handles = [Line2D([], [], color=COL[s], lw=1.1, marker="o", ms=2.2, label=f) for s, f in STU]
    handles += [Line2D([], [], color="#222222", lw=1.1, ls=(0, (1.2, 1.0)), marker="s", ms=2.2,
                       label="Teacher solutions"),
                Line2D([], [], color="#555555", lw=1.1, label=r"$R(\lambda)$"),
                Line2D([], [], color="#555555", lw=1.0, ls="--", label=r"$R_{\mathrm{sym}}(\lambda)$"),
                Patch(facecolor=THINK_BAR, edgecolor="none", label="Teacher think rate")]
    lg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.945), ncol=7, handlelength=1.8,
                    columnspacing=1.1, fontsize=6.0, borderpad=0.4)
    lg.get_frame().set_linewidth(0.6)
    save(fig, "appfig_lambda")


# ---------------------------------------------------------------- Figure: other benchmarks (Tables 13, 14)
def fig_ood():
    g = lambda bench, cond, f: [ood[bench][cell(p, s)][cond][f]["mean"] for p, s in CELLS]
    pct = lambda v: [100 * x for x in v]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 2.95), gridspec_kw={"hspace": 0.66, "wspace": 0.3})
    fig.subplots_adjust(top=0.86)
    (a, b), (c, d) = axes
    title = lambda ax, t: ax.text(0, 1.04, t, transform=ax.transAxes, fontsize=7.2, fontweight="bold", va="bottom")
    for ax, bench, letter, name in ((a, "gsm8k", "a", "GSM8K"), (b, "gpqa_diamond", "b", "GPQA-Diamond")):
        bars(ax, [(r"$\lambda = 0$ adapter", F3_BASE, g(bench, "base0", "markers")),
                  ("Other removed", F3_CONTROL, g(bench, "rm_other", "markers")),
                  ("Shared removed", F3_SHARED, g(bench, "rm_pre", "markers"))],
             "Reflective markers", "(per 1,000 words)")
        title(ax, f"({letter}) {name}, markers")
    four = lambda f: [(r"$\lambda = 0$ adapter", F3_BASE, pct(g("gsm8k", "base0", f))),
                      ("Other removed", F3_CONTROL, pct(g("gsm8k", "rm_other", f))),
                      ("Shared removed", F3_SHARED, pct(g("gsm8k", "rm_pre", f))),
                      (r"$\lambda = 1$ adapter", LIGHT, pct(g("gsm8k", "base1", f)))]
    bars(c, four("accuracy"), "Accuracy", "(%)", top=75)
    title(c, "(c) GSM8K, accuracy")
    bars(d, four("truncated"), "Truncated responses", "(%)", top=100)
    d.set_yticks([0, 25, 50, 75, 100])
    title(d, "(d) GSM8K, truncation")
    h, l = c.get_legend_handles_labels()
    lg = fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.93), ncol=4, handlelength=1.4,
                    columnspacing=1.2, fontsize=6.2, borderpad=0.4)
    lg.get_frame().set_linewidth(0.6)
    save(fig, "appfig_ood")


# ---------------------------------------------------------------- Figure: ways to turn thinking off (Table 15)
METHODS = [  # key in baselines.json, label, colour, needs a short-solution adapter
    ("base0", r"$\lambda = 0$ adapter", F3_BASE, False),
    ("rmseed", r"Two-seed $\lambda = 0$ region removed", F3_SHARED, False),
    ("gs_tau", "Gold-Switch-style top removed", F3_SHARED_SOFT, False),
    ("budget", "Length-matched cut", F1_LAVENDER, False),
    ("nothink", "Think-skip prompt (adapted)", "#ffffff", False),
    ("rmseed_other", "Other removed (two-seed region)", F3_CONTROL, False),
]


def fig_off():
    m = lambda k, f: [base[cell(p, s)][k][f]["mean"] for p, s in CELLS]
    unt = lambda f: [untrained[UNTR[s]][f]["mean"] for _, s in CELLS]
    pct = lambda v: [100 * x for x in v]
    fig = plt.figure(figsize=(WIDTH, 4.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.15], height_ratios=[1.15, 1.0], hspace=0.75, wspace=0.3)
    a, c, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    # (a) nine-cell means, one labelled point per method (comparison display; flagged for the user)
    pts = [("base0", r"$\lambda = 0$"), ("rm_pre", "Shared removed"), ("rmseed", "Two-seed removed"),
           ("gs_tau", "Gold-Switch-style"), ("budget", "Length-matched cut"), ("nothink", "Think-skip"),
           ("rmseed_other", "Other removed"), ("merge0.5", r"Merged with $\lambda = 1$"), ("base1", r"$\lambda = 1$")]
    # label position in data coordinates (x, y, ha) and whether a short leader line joins it to its point;
    # labels of the three points near (1, 25) sit apart with leaders, the rest sit right next to their point
    place = {"base0": (23.3, 13.9, "left", False), "rm_pre": (6.4, 23.2, "left", "angle"),
             "rmseed": (3.1, 21.7, "left", False), "gs_tau": (6.6, 19.1, "left", False),
             "budget": (13.6, 16.0, "right", False), "nothink": (14.9, 13.9, "right", False),
             "rmseed_other": (19.6, 16.1, "left", False), "merge0.5": (6.4, 26.0, "left", True),
             "base1": (6.4, 24.6, "left", True)}
    second = {"merge0.5", "base1"}
    for k, lab in pts:
        mk, ac = st.mean(m(k, "markers")), 100 * st.mean(m(k, "accuracy"))
        filled = k not in second
        a.plot(mk, ac, marker="o" if filled else "D", ms=4.2, color=F3_SHARED if filled else "white",
               mec=F3_SHARED, mew=1.0, ls="none", zorder=3)
        tx, ty, ha, lead = place[k]
        arrow = dict(arrowstyle="-", lw=0.5, color="#555555", shrinkA=1.0, shrinkB=3.0) if lead else None
        if lead == "angle":                              # run under the lambda = 1 point, then up to this one
            arrow["connectionstyle"] = "angle,angleA=0,angleB=90,rad=0"
        a.annotate(lab, (mk, ac), xytext=(tx, ty), textcoords="data", fontsize=5.8, ha=ha, va="center", linespacing=1.0,
                   arrowprops=arrow)
        print(f"  mean {k:13s} markers {mk:5.1f}  tokens {st.mean(m(k, 'tokens')):6.0f}  acc {ac:4.1f}")
    u_acc = 100 * st.mean(unt("accuracy"))
    print(f"  untrained acc {u_acc:.1f}  markers(first sample) {st.mean(unt('markers_first_sample')):.2f}"
          f"  markers(all) {st.mean(unt('reflective_per_1k_words')):.2f}")
    a.axhline(u_acc, color="black", ls="--", lw=1.2, zorder=2)
    a.text(26.5, u_acc + 0.4, "Untrained student", fontsize=6.0, fontweight="bold", ha="right", va="bottom")
    a.set_xlim(-1.5, 27); a.set_ylim(12, 28.5)
    labels(a, x="Reflective markers left", xunit="(per 1,000 words)", y="MATH-500 accuracy", yunit="(%)")
    a.text(0, 1.2, "(a) Nine-cell means", transform=a.transAxes, fontsize=7.2, fontweight="bold", va="bottom")
    legend(a, [Line2D([], [], marker="o", ms=4, color=F3_SHARED, ls="none", label="No further training"),
               Line2D([], [], marker="D", ms=4, color="white", mec=F3_SHARED, ls="none",
                      label="Short-solution adapter")], None, "lower left", fontsize=5.6,
           bbox_to_anchor=(0, 1.0), ncol=2, borderpad=0.35)
    # (b) per-cell accuracy of the options without further training, untrained student as a dashed tick
    bars(b, [(lab, col, pct(m(k, "accuracy"))) for k, lab, col, _ in METHODS], "MATH-500 accuracy", "(%)", top=40)
    ua = pct(unt("accuracy"))
    for i in range(9):
        b.plot([i - 0.44, i + 0.44], [ua[i]] * 2, color="black", ls="--", lw=1.0, zorder=3)
    b.text(0, 1.3, "(c) Without further training", transform=b.transAxes, fontsize=7.2, fontweight="bold",
           va="bottom")
    h, l = b.get_legend_handles_labels()
    h.append(Line2D([], [], color="black", ls="--", lw=1.0)); l.append("Untrained student")
    lg = b.legend(h, l, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, handlelength=1.3, columnspacing=0.8,
                  fontsize=5.6, borderpad=0.35)
    lg.get_frame().set_linewidth(0.6)
    # (c) per-cell accuracy of the options that need a short-solution adapter
    bars(c, [(r"Merged with $\lambda = 1$", F3_SHARED_SOFT, pct(m("merge0.5", "accuracy"))),
             (r"$\lambda = 1$ adapter", LIGHT, pct(m("base1", "accuracy")))], "MATH-500 accuracy", "(%)", top=40,
         fs=5.4)
    for i in range(9):
        c.plot([i - 0.44, i + 0.44], [ua[i]] * 2, color="black", ls="--", lw=1.0, zorder=3)
    c.text(0, 1.2, "(b) With a short-solution adapter", transform=c.transAxes, fontsize=7.2, fontweight="bold",
           va="bottom")
    h, l = c.get_legend_handles_labels()
    h.append(Line2D([], [], color="black", ls="--", lw=1.0)); l.append("Untrained student")
    lg = c.legend(h, l, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, handlelength=1.3, columnspacing=0.8,
                  fontsize=5.6, borderpad=0.35)
    lg.get_frame().set_linewidth(0.6)
    save(fig, "appfig_off")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_lambda(); fig_ood(); fig_off()
    print("written to", OUT)
