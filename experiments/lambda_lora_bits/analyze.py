#!/usr/bin/env python3
"""Aggregate the lambda sweep into a CSV, figures and a markdown report."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METRICS = [
    ("val_acc", "held-out acc"),
    ("test_acc", "MATH-500 acc"),
    ("train_tokens", "train tokens"),
    ("H_train_bits", "dataset bits H"),
    ("CL_bits", "CL(Δθ) bits"),
    ("c_star", "step c*"),
    ("CL_bits_per_param", "CL bits/param"),
    ("W_train_bits", "W(train) bits"),
    ("W_held_bits", "W(held) bits"),
    ("dW_bits", "ΔW bits"),
    ("dW_tok_bits", "ΔW token-scaled"),
    ("W_per_token", "W bits/token"),
    ("dW_per_token", "ΔW bits/token"),
    ("W_over_H", "W/H"),
]


def load(path):
    return json.loads(Path(path).read_text())


def parse_name(stem):
    lam, seed = stem.split("_seed")
    return lam.replace("lambda", ""), int(seed)


def mean_std(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not values:
        return None, None
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def fmt(value, std=None):
    if value is None:
        return "—"
    magnitude = abs(value)
    text = f"{value:.3g}" if magnitude < 1 else (f"{value:,.0f}" if magnitude >= 1000 else f"{value:.2f}")
    if std:
        text += f" ± {std:.2g}" if abs(std) < 1 else f" ± {std:,.0f}"
    return text


def collect(sweep: Path):
    runs = defaultdict(dict)
    for path in sorted((sweep / "bits").glob("lambda*_seed*.json")):
        bits = load(path)
        runs[parse_name(path.stem)].update(
            train_tokens=bits["train"]["tokens"],
            H_train_bits=bits["train"]["H_base_bits"],
            CL_bits=bits["CL"]["CL_update_bits"],
            c_star=bits["CL"]["c_star"],
            CL_bits_per_param=bits["CL"]["CL_update_bits_per_param"],
            W_train_bits=bits["train"]["W_bits"],
            W_held_bits=bits["held"]["W_bits"],
            dW_bits=bits["delta_W_bits"],
            dW_tok_bits=bits["delta_W_token_scaled_bits"],
            W_per_token=bits["train"]["W_bits_per_token"],
            dW_per_token=bits["delta_W_bits_per_train_token"],
            W_over_H=bits["train"]["W_over_H"],
            cross={k.replace("lambda", ""): v["W_bits_per_token"] for k, v in bits["cross"].items()},
        )
    base = {}
    for split in ("val", "test"):
        folder = sweep / f"eval_{split}"
        for path in sorted(folder.glob("*.json")):
            summary = load(path)["summary"]
            if path.stem == "base":
                base[split] = summary["accuracy"]
            else:
                runs[parse_name(path.stem)][f"{split}_acc"] = summary["accuracy"]
    return runs, base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep_dir", default="experiments/lambda_lora_bits/sweep")
    parser.add_argument("--out_dir", default=None, help="Where to write the report; defaults to sweep_dir.")
    args = parser.parse_args()
    sweep = Path(args.sweep_dir)
    out = Path(args.out_dir) if args.out_dir else sweep
    out.mkdir(parents=True, exist_ok=True)

    runs, base = collect(sweep)
    split_stats = load(sweep / "splits" / "split_stats.json")
    choice = load(sweep / "best_lambda.json") if (sweep / "best_lambda.json").exists() else {}
    lambdas = sorted({lam for lam, _ in runs}, key=float)

    with (out / "results.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lambda", "seed"] + [key for key, _ in METRICS])
        for (lam, seed), metrics in sorted(runs.items(), key=lambda item: (float(item[0][0]), item[0][1])):
            writer.writerow([lam, seed] + [metrics.get(key) for key, _ in METRICS])

    by_lambda = {lam: {key: mean_std([m.get(key) for (l, _), m in runs.items() if l == lam]) for key, _ in METRICS} for lam in lambdas}
    seeds = {lam: sorted(s for l, s in runs if l == lam) for lam in lambdas}

    lines = ["# λ sweep: LoRA bits of CoT distillation", ""]
    lines += [
        "Teacher pair: DeepSeek-R1-Distill-Qwen-1.5B (λ=0, Long CoT) ↔ Qwen2.5-Math-1.5B-Instruct (λ=1, Short CoT); "
        "student: Qwen2.5-0.5B-Instruct + LoRA (r=16, all linear layers).",
        f"Shared problems: {split_stats['shared']} (train {split_stats['n_train']}, held {split_stats['n_held']}). "
        f"Base student accuracy: held {fmt(base.get('val'))}, MATH-500 {fmt(base.get('test'))}.",
    ]
    if choice:
        lines.append(f"Best interior λ on held-out accuracy (seed 0): **{choice['best_interior_lambda']}** "
                     f"(best overall: {choice['best_overall_lambda']}).")
    lines += ["", "## Teacher CoT pools", "", "| λ | teacher acc | truncated | avg teacher tokens | avg student tokens (kept) | train tokens |", "|---|---|---|---|---|---|"]
    for lam, s in split_stats["per_lambda"].items():
        lines.append(f"| {lam} | {fmt(s['teacher_accuracy'])} | {fmt(s['truncated_rate'])} | {fmt(s['avg_teacher_tokens_all'])} | "
                     f"{fmt(s['avg_student_tokens_kept'])} | {fmt(s.get('train_tokens'))} |")

    lines += ["", "## Per-λ results (mean ± std over seeds)", "", "| λ | seeds | " + " | ".join(label for _, label in METRICS) + " |",
              "|" + "---|" * (len(METRICS) + 2)]
    for lam in lambdas:
        cells = [fmt(*by_lambda[lam][key]) for key, _ in METRICS]
        lines.append(f"| {lam} | {len(seeds[lam])} | " + " | ".join(cells) + " |")

    cross_runs = {lam: runs[(lam, 0)]["cross"] for lam in lambdas if "cross" in runs.get((lam, 0), {})}
    if cross_runs:
        cols = sorted(next(iter(cross_runs.values())), key=float)
        lines += ["", "## Cross-λ W (bits/token, seed 0): adapter trained on row λ, measured on column λ's held CoT", "",
                  "| train λ \\ eval λ | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
        for lam, row in cross_runs.items():
            lines.append(f"| {lam} | " + " | ".join(fmt(row[c]) for c in cols) + " |")

    x = [float(lam) for lam in lambdas]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    panels = [("CL_bits", "CL(Δθ) bits"), ("W_train_bits", "W(train) bits"), ("dW_bits", "ΔW bits (count-scaled)"),
              ("W_per_token", "W bits / train token"), ("dW_per_token", "ΔW bits / train token"), ("val_acc", "held-out accuracy")]
    for ax, (key, title) in zip(axes.flat, panels):
        means = [by_lambda[lam][key][0] for lam in lambdas]
        stds = [by_lambda[lam][key][1] or 0.0 for lam in lambdas]
        pts = [(xi, m, s) for xi, m, s in zip(x, means, stds) if m is not None]
        if pts:
            ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts], marker="o", capsize=3)
        if choice:
            best = choice["best_interior_lambda"]
            ax.axvline(float(best), color="tab:orange", linestyle=":", label=f"best interior λ={best}")
        if key == "val_acc" and "val" in base:
            ax.axhline(base["val"], color="gray", linestyle="--", label="base student")
        if choice or key == "val_acc":
            ax.legend(fontsize=8)
        if key == "CL_bits":
            # CL barely moves across arms; an offset y-axis would make a 0.1% change look large.
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.set_ylim(bottom=0)
            title += " (y from 0)"
        ax.set_title(title)
        ax.set_xlabel("λ (weight on instruct teacher)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_bits_vs_lambda.png", dpi=150)

    if cross_runs:
        cols = sorted(next(iter(cross_runs.values())), key=float)
        matrix = [[cross_runs[lam][c] for c in cols] for lam in cross_runs]
        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(matrix, cmap="viridis")
        ax.set_xticks(range(len(cols)), cols)
        ax.set_yticks(range(len(cross_runs)), list(cross_runs))
        ax.set_xlabel("eval λ (held CoT)")
        ax.set_ylabel("train λ (adapter)")
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(image, label="W bits/token")
        fig.tight_layout()
        fig.savefig(out / "fig_cross_W.png", dpi=150)

    lines += ["", "![bits vs lambda](fig_bits_vs_lambda.png)", "", "![cross W](fig_cross_W.png)", ""]
    (out / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
