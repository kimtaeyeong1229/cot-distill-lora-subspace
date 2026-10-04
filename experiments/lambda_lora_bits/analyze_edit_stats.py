#!/usr/bin/env python3
"""Seed-level spread, bootstrap intervals and medians for the main edit effects (Section 5).

Every edit was run once per adapter seed (0-2) in each of the nine cells. The tables report per-cell
means +- SD and nine-cell means; this script adds, from the same per-seed values,

  per cell     the cell value as the tables give it, and the lowest and highest of the three seed values
  nine cells   the mean of the cell values, a 95% bootstrap interval for it (10,000 resamples that
               redraw the three seeds of every cell with replacement, so only seed variation enters;
               cells are fixed), the median of the cell values, and the number of cells whose value
               is above zero (for differences)

For the two "% of lambda0" quantities the cell value is the ratio of the seed means (edited mean /
lambda0 mean, as the text reads Tables 6 and 10), the per-seed range uses each seed's own ratio, and a
bootstrap draw resamples seeds as (edited, lambda0) pairs before taking the ratio of means.

Quantities, all on MATH-500 (first 200 problems x 2 samples, markers on each problem's first response):
  rm_retained      markers after removing the shared region, % of the same seed's lambda0 adapter
  rm_acc           accuracy (%) after removing the shared region
  rm_minus_top     accuracy (pp) of shared removed minus top removed (energy-matched), same seed
  rm_minus_cut     accuracy (pp) of shared removed minus the length-matched cut, same seed
  graft2_head      early-response markers (first 300 words) of the beta = 2 shared-region graft
  graft2_done      completion rate (%) of the beta = 2 shared-region graft
  amp4_retained    markers after amplifying at gamma = 4 with the norm fixed, % of the same seed's lambda0

Inputs: experiments/seed_spread.json (edits, lambda0 bases) and experiments/baselines.json (top removed,
length-matched cut). Run from the repo root; writes experiments/edit_stats.json.
"""

import json
import random
import statistics as st
from pathlib import Path

X = Path("experiments")
CELLS = ["A/Qwen", "A/Llama", "A/OLMo", "B/Qwen", "B/Llama", "B/OLMo", "C/Qwen", "C/Llama", "C/OLMo"]
REPS = 10_000


RATIOS = {"rm_retained", "amp4_retained"}


def per_seed(spread, base):
    """Per-seed values; ratio quantities are kept as (edited, lambda0) pairs."""
    b0 = spread["base"]["base0_markers_per_1k"]["seeds"]
    e = spread["edits"]
    acc = lambda k: [100 * v for v in base[k]["accuracy"]["seeds"]]
    rm = acc("rm_pre")
    return {
        "rm_retained": list(zip(e["rm_pre"]["markers_per_1k"]["seeds"], b0)),
        "rm_acc": rm,
        "rm_minus_top": [a - b for a, b in zip(rm, acc("gs_energy"))],
        "rm_minus_cut": [a - b for a, b in zip(rm, acc("budget"))],
        "graft2_head": e["graft2"]["head_markers_per_1k"]["seeds"],
        "graft2_done": [100 * v for v in e["graft2"]["completed"]["seeds"]],
        "amp4_retained": list(zip(e["ampn4"]["markers_per_1k"]["seeds"], b0)),
    }


def value(k, seeds):
    """Cell value from a list of per-seed entries (ratio of means for the ratio quantities)."""
    if k in RATIOS:
        return 100 * st.mean(a for a, _ in seeds) / st.mean(b for _, b in seeds)
    return st.mean(seeds)


def seed_values(k, seeds):
    return [100 * a / b for a, b in seeds] if k in RATIOS else list(seeds)


def main():
    spread = json.loads((X / "seed_spread.json").read_text())
    base = json.loads((X / "baselines.json").read_text())
    cells = {c: per_seed(spread[c], base[c]) for c in CELLS}
    rng = random.Random(0)
    out = {"cells": {}, "nine_cell": {}}
    for c, q in cells.items():
        out["cells"][c] = {k: {"value": value(k, v), "min": min(seed_values(k, v)), "max": max(seed_values(k, v)),
                               "seeds": seed_values(k, v)} for k, v in q.items()}
    for k in cells[CELLS[0]]:
        means = [value(k, cells[c][k]) for c in CELLS]
        boots = sorted(st.mean(value(k, rng.choices(cells[c][k], k=3)) for c in CELLS) for _ in range(REPS))
        out["nine_cell"][k] = {"mean": st.mean(means), "ci95": [boots[int(0.025 * REPS)], boots[int(0.975 * REPS) - 1]],
                               "median": st.median(means), "min_cell": min(means), "max_cell": max(means),
                               "cells_above_zero": sum(m > 0 for m in means),
                               "mean_without_B_C_Qwen": st.mean(m for c, m in zip(CELLS, means)
                                                                 if c not in ("B/Qwen", "C/Qwen"))}
        n = out["nine_cell"][k]
        print(f"{k:14s} mean {n['mean']:7.2f} CI [{n['ci95'][0]:.2f}, {n['ci95'][1]:.2f}] median {n['median']:7.2f} "
              f"range {n['min_cell']:.2f}-{n['max_cell']:.2f} >0 in {n['cells_above_zero']}/9 "
              f"(without B/Qwen, C/Qwen {n['mean_without_B_C_Qwen']:.2f})")
    (X / "edit_stats.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
