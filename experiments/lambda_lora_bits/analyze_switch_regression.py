#!/usr/bin/env python3
"""Pooled regression of region loss on length drop and a switch indicator (Appendix B).

Over the held and switch intervals of all nine cells, -ln(kept share against the same-lambda half-data
reference) is regressed on the log-length drop and a switch dummy. Cells are resampled for a bootstrap
interval (several bootstrap seeds, to show how stable the lower end is). In pairs B and C every held
interval shortens solutions by about 0.2 and every switch by about 0.6-0.7, so length and class are
confounded there; only pair A separates them, and the pooled switch coefficient is not identified well.

Writes experiments/switch_regression.json.
"""

import json
import math
import random

import numpy as np

from config import RESULTS as X


def fit(rows):
    A = np.array([[1.0, r["log_len"], r["switch"]] for r in rows])
    y = np.array([r["loss"] for r in rows])
    return np.linalg.lstsq(A, y, rcond=None)[0]


def main():
    d = json.loads((X / "length_confound.json").read_text())
    rows = []
    for p, v in d["pairs"].items():
        for r in v["intervals"]:
            if r["class"] in ("held", "switch"):
                for s, k in r["kept_vs_half_reference"].items():
                    rows.append({"cell": f"{p}/{s}", "loss": -math.log(k), "log_len": r["log_len_drop"],
                                 "switch": float(r["class"] == "switch")})
    b = fit(rows)
    cells = sorted({r["cell"] for r in rows})
    boot = {}
    for seed in range(4):
        rng = random.Random(seed)
        coefs = []
        for _ in range(5000):
            pick = [rng.choice(cells) for _ in cells]
            coefs.append(fit([r for c in pick for r in rows if r["cell"] == c])[2])
        boot[seed] = [float(np.percentile(coefs, 2.5)), float(np.percentile(coefs, 97.5))]
    out = {"n": len(rows), "intercept": float(b[0]), "per_log_len": float(b[1]), "switch": float(b[2]),
           "switch_ci95_by_bootstrap_seed": boot}
    print(json.dumps(out, indent=1))
    (X / "switch_regression.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
