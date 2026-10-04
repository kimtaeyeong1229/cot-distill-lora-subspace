#!/usr/bin/env python3
"""Problem-level paired bootstrap for the removal comparisons of Section 5.1.

Shared-region removal is compared with the length-matched cut (budget), with energy-matched top removal
(gs_energy) and with removing comparable energy elsewhere (rm_decisive), all edits of the same lambda = 0
adapter on the same 200 MATH-500 problems. Every stored response (two per problem and seed) is regraded
with the evaluation's grader. Per problem, accuracy is averaged over the two samples and three seeds, and
the difference to removal is resampled over problems (10,000 draws): per cell, and over the nine cells
with problems resampled within each cell.

Run from the repo root; writes experiments/paired_bootstrap.json.
"""

import gzip
import json
import random
import statistics as st
import sys
from pathlib import Path

import pandas as pd

E = Path("experiments/lambda_lora_bits")
sys.path.insert(0, str(E))
from common import is_correct  # noqa: E402

CELLS = {"A/Qwen": "sweep", "A/Llama": "sweep_llama", "A/OLMo": "sweep_olmo2",
         "B/Qwen": "pairB/qwen0.5b", "B/Llama": "pairB/llama1b", "B/OLMo": "pairB/olmo1b",
         "C/Qwen": "pairD/qwen0.5b", "C/Llama": "pairD/llama1b", "C/OLMo": "pairD/olmo1b"}
CONDS = {"rm_pre": "lambda0.0_rm_pre", "cut": "lambda0.0_budget", "top": "lambda0.0_gs_energy",
         "other": "lambda0.0_rm_decisive"}
SEEDS, SAMPLES, PROBLEMS, DRAWS = [0, 1, 2], 2, 200, 10000


def per_problem(tag, name, answers):
    """Accuracy per problem, mean over samples and seeds."""
    acc = [[] for _ in range(PROBLEMS)]
    for s in SEEDS:
        with gzip.open(E / tag / f"eval_main_s{s}" / f"{name}.samples.jsonl.gz", "rt") as f:
            for i, line in enumerate(f):
                r = json.loads(line)
                p = i // SAMPLES
                acc[p].append(float(is_correct(r["problem"], r["text"], answers[p])))
    return [st.mean(a) for a in acc]


def ci(diffs, rng):
    n = len(diffs)
    boots = sorted(st.mean(diffs[rng.randrange(n)] for _ in range(n)) for _ in range(DRAWS))
    return [boots[int(0.025 * DRAWS)], boots[int(0.975 * DRAWS)]]


def main():
    rows = pd.read_parquet(E / "data" / "math500.parquet").to_dict("records")[:PROBLEMS]
    answers = [str(r["answer"]) for r in rows]
    rng = random.Random(0)
    acc = {}
    for cell, tag in CELLS.items():
        acc[cell] = {c: per_problem(tag, name, answers) for c, name in CONDS.items()}
        print(cell, " ".join(f"{c} {100 * st.mean(v):.1f}" for c, v in acc[cell].items()), flush=True)
    out = {"cells": {}, "nine_cell": {}}
    for c in ("cut", "top", "other"):
        for cell in CELLS:
            d = [a - b for a, b in zip(acc[cell]["rm_pre"], acc[cell][c])]
            lo, hi = ci(d, rng)
            out["cells"].setdefault(cell, {})[c] = {"diff_pp": 100 * st.mean(d), "ci95_pp": [100 * lo, 100 * hi],
                                                    "rm_acc": 100 * st.mean(acc[cell]["rm_pre"]),
                                                    "other_acc": 100 * st.mean(acc[cell][c])}
        diffs = {cell: [a - b for a, b in zip(acc[cell]["rm_pre"], acc[cell][c])] for cell in CELLS}
        boots = []
        for _ in range(DRAWS):
            boots.append(st.mean(st.mean(d[rng.randrange(PROBLEMS)] for _ in range(PROBLEMS)) for d in diffs.values()))
        boots.sort()
        mean = st.mean(st.mean(d) for d in diffs.values())
        out["nine_cell"][c] = {"diff_pp": 100 * mean,
                               "ci95_pp": [100 * boots[int(0.025 * DRAWS)], 100 * boots[int(0.975 * DRAWS)]],
                               "cells_ci_above_zero": sum(out["cells"][k][c]["ci95_pp"][0] > 0 for k in CELLS),
                               "cells_mean_above_zero": sum(out["cells"][k][c]["diff_pp"] > 0 for k in CELLS)}
        print(c, out["nine_cell"][c], flush=True)
    Path("experiments/paired_bootstrap.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
