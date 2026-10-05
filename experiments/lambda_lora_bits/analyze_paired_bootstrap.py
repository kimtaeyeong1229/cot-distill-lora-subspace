#!/usr/bin/env python3
"""Problem-level paired bootstrap for the removal comparisons of Section 5.1.

Shared-region removal is compared with the length-matched cut (budget), with energy-matched top removal
(gs_energy) and with removing comparable energy elsewhere (rm_decisive), all edits of the same lambda = 0
adapter on the same 200 MATH-500 problems. Every stored response (two per problem and seed) is regraded
with the evaluation's grader. Per problem, accuracy is averaged over the two samples and three seeds, and
the difference to removal is resampled over problems (10,000 draws): per cell, and over the nine cells
with one draw of problem indices shared by all cells, since every cell answers the same 200 problems. The
per-cell intervals condition on the three trained adapters; a second per-cell interval also resamples
the three seeds (seed indices drawn with replacement, then problems), counting both sources of noise.

Writes experiments/paired_bootstrap.json.
"""

import gzip
import json
import random
import statistics as st

import pandas as pd

from common import is_correct
from config import CELL_DIR as CELLS, EVAL_PROBLEMS as PROBLEMS, EVAL_SAMPLES as SAMPLES, EXP as E, RESULTS, SEEDS

CONDS = {"rm_pre": "lambda0.0_rm_pre", "cut": "lambda0.0_budget", "top": "lambda0.0_gs_energy",
         "other": "lambda0.0_rm_decisive"}
DRAWS = 10000


def per_problem(tag, name, answers):
    """Accuracy per seed and problem, mean over the two samples."""
    acc = {s: [[] for _ in range(PROBLEMS)] for s in SEEDS}
    for s in SEEDS:
        with gzip.open(E / tag / f"eval_main_s{s}" / f"{name}.samples.jsonl.gz", "rt") as f:
            for i, line in enumerate(f):
                r = json.loads(line)
                p = i // SAMPLES
                acc[s][p].append(float(is_correct(r["problem"], r["text"], answers[p])))
    return {s: [st.mean(a) for a in v] for s, v in acc.items()}


def seed_mean(per_seed):
    return [st.mean(per_seed[s][p] for s in SEEDS) for p in range(PROBLEMS)]


def ci(diffs, rng):
    n = len(diffs)
    boots = sorted(st.mean(diffs[rng.randrange(n)] for _ in range(n)) for _ in range(DRAWS))
    return [boots[int(0.025 * DRAWS)], boots[int(0.975 * DRAWS)]]


def main():
    rows = pd.read_parquet(E / "data" / "math500.parquet").to_dict("records")[:PROBLEMS]
    answers = [str(r["answer"]) for r in rows]
    rng = random.Random(0)
    cache = RESULTS / "paired_bootstrap_grades.json"
    if cache.exists():
        raw = {cell: {c: {int(s): v for s, v in d.items()} for c, d in cd.items()}
               for cell, cd in json.loads(cache.read_text()).items()}
    else:
        raw = {cell: {c: per_problem(tag, name, answers) for c, name in CONDS.items()} for cell, tag in CELLS.items()}
        cache.write_text(json.dumps(raw))
    acc = {cell: {c: seed_mean(v) for c, v in cd.items()} for cell, cd in raw.items()}
    for cell in CELLS:
        print(cell, " ".join(f"{c} {100 * st.mean(v):.1f}" for c, v in acc[cell].items()), flush=True)
    out = {"cells": {}, "nine_cell": {}}
    for c in ("cut", "top", "other"):
        for cell in CELLS:
            d = [a - b for a, b in zip(acc[cell]["rm_pre"], acc[cell][c])]
            lo, hi = ci(d, rng)
            boots = []
            for _ in range(DRAWS):
                ss = [rng.choice(SEEDS) for _ in SEEDS]
                ps = [rng.randrange(PROBLEMS) for _ in range(PROBLEMS)]
                boots.append(st.mean(st.mean(raw[cell]["rm_pre"][s][p] - raw[cell][c][s][p] for s in ss) for p in ps))
            boots.sort()
            out["cells"].setdefault(cell, {})[c] = {"diff_pp": 100 * st.mean(d), "ci95_pp": [100 * lo, 100 * hi],
                                                    "ci95_pp_seeds_resampled": [100 * boots[int(0.025 * DRAWS)],
                                                                                100 * boots[int(0.975 * DRAWS)]],
                                                    "rm_acc": 100 * st.mean(acc[cell]["rm_pre"]),
                                                    "other_acc": 100 * st.mean(acc[cell][c])}
        diffs = {cell: [a - b for a, b in zip(acc[cell]["rm_pre"], acc[cell][c])] for cell in CELLS}
        boots = []
        for _ in range(DRAWS):
            ps = [rng.randrange(PROBLEMS) for _ in range(PROBLEMS)]       # one draw shared by all cells
            boots.append(st.mean(st.mean(d[p] for p in ps) for d in diffs.values()))
        boots.sort()
        mean = st.mean(st.mean(d) for d in diffs.values())
        out["nine_cell"][c] = {"diff_pp": 100 * mean,
                               "ci95_pp": [100 * boots[int(0.025 * DRAWS)], 100 * boots[int(0.975 * DRAWS)]],
                               "cells_ci_above_zero": sum(out["cells"][k][c]["ci95_pp"][0] > 0 for k in CELLS),
                               "cells_mean_above_zero": sum(out["cells"][k][c]["diff_pp"] > 0 for k in CELLS),
                               "cells_ci_above_zero_seeds_resampled":
                                   sum(out["cells"][k][c]["ci95_pp_seeds_resampled"][0] > 0 for k in CELLS)}
        print(c, out["nine_cell"][c], flush=True)
    (RESULTS / "paired_bootstrap.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
