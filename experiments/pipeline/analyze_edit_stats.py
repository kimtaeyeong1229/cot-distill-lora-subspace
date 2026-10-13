#!/usr/bin/env python3
import json
import random
import statistics as st
from pathlib import Path
from config import CELL_NAMES as CELLS, RESULTS as X

REPS = 10000
RATIOS = {"rm_retained", "amp4_retained"}


def per_seed(spread, base):
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
    if k in RATIOS:
        return 100 * st.mean((a for a, _ in seeds)) / st.mean((b for _, b in seeds))
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
        out["cells"][c] = {
            k: {
                "value": value(k, v),
                "min": min(seed_values(k, v)),
                "max": max(seed_values(k, v)),
                "seeds": seed_values(k, v),
            }
            for k, v in q.items()
        }
    for k in cells[CELLS[0]]:
        means = [value(k, cells[c][k]) for c in CELLS]
        boots = sorted(
            (st.mean((value(k, rng.choices(cells[c][k], k=3)) for c in CELLS)) for _ in range(REPS))
        )
        out["nine_cell"][k] = {
            "mean": st.mean(means),
            "ci95": [boots[int(0.025 * REPS)], boots[int(0.975 * REPS) - 1]],
            "median": st.median(means),
            "min_cell": min(means),
            "max_cell": max(means),
            "cells_above_zero": sum((m > 0 for m in means)),
            "mean_without_B_C_Qwen": st.mean(
                (m for c, m in zip(CELLS, means) if c not in ("B/Qwen", "C/Qwen"))
            ),
        }
        n = out["nine_cell"][k]
        print(
            f"{k:14s} mean {n['mean']:7.2f} CI [{n['ci95'][0]:.2f}, {n['ci95'][1]:.2f}] median {n['median']:7.2f} range {n['min_cell']:.2f}-{n['max_cell']:.2f} >0 in {n['cells_above_zero']}/9 (without B/Qwen, C/Qwen {n['mean_without_B_C_Qwen']:.2f})"
        )
    (X / "edit_stats.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
