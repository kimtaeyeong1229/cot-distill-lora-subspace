#!/usr/bin/env python3
"""Floor-corrected coverage of the two-seed lambda = 0 region across the held and switch intervals (read only).

Input: experiments/length_confound.json, two_seed_region.<cell>.coverage (coverage of the region built from two
lambda = 0 seeds, eigenvalue cut 0.75, by the held-out third seed at every lambda, mean over the three choices
of held-out seed) and .chance, and pairs.<P>.intervals[] (class, log_len_drop). No new adapters are read.

Two floors f are subtracted from the coverage:
  chance   the coverage a random adapter of the same rank would have (two_seed_region.<cell>.chance)
  lambda1  the coverage at lambda = 1 (the short-answer adapter, which never thinks)
Kept shares:
  interval   (cov(b) - f) / (cov(a) - f) for the interval a -> b (the share of the interval start that the end keeps)
  cumulative (cov(b) - f) / (cov(0) - f) (the share of the lambda = 0 level)
Per unit of log training-length drop: -ln(interval kept) / log_len_drop. For pair A, the switch/held ratio of that
loss per student. This uses neither the half-data reference nor the leave-one-out region of the shared region.

Run from the repo root; writes experiments/two_seed_floor.json.
"""

import json
import math
from pathlib import Path

X = Path("experiments")
LAM = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
PAIR = {"A": "A", "B": "B", "C": "C"}


def main():
    lc = json.loads((X / "length_confound.json").read_text())
    ts = lc["two_seed_region"]
    out = {"definitions": __doc__.split("Run from")[0].strip(), "floors": {}}
    for fname in ("chance", "lambda1"):
        res = {}
        for p, v in lc["pairs"].items():
            for s in ("Qwen", "Llama", "OLMo"):
                cell = f"{p}/{s}"
                cov = ts[cell]["coverage"]
                f = ts[cell]["chance"] if fname == "chance" else cov["1.0"]
                rows = {}
                for r in v["intervals"]:
                    a, b = r["interval"].split("->")
                    num, den = cov[b] - f, cov[a] - f
                    kept = num / den if den > 0 else None
                    per = (-math.log(kept) / r["log_len_drop"]) if kept and kept > 0 and r["log_len_drop"] > 0 else None
                    rows[r["interval"]] = {"class": r["class"], "log_len_drop": r["log_len_drop"], "kept_interval": kept,
                                           "kept_cumulative": (num / (cov["0.0"] - f)), "loss_per_log_len": per}
                res[cell] = rows
        ratio = {s: res[f"A/{s}"]["0.2->0.4"]["loss_per_log_len"] / res[f"A/{s}"]["0.0->0.2"]["loss_per_log_len"]
                 for s in ("Qwen", "Llama", "OLMo")}
        out["floors"][fname] = {"cells": res, "pairA_switch_over_held": ratio}
        print(fname)
        for cell, rows in res.items():
            print(" ", cell, " | ".join(f"{iv} {r['class'][0]} kept {r['kept_interval']:.2f} per {r['loss_per_log_len']:.2f}"
                                        if r['kept_interval'] is not None and r['loss_per_log_len'] is not None else f"{iv} -"
                                        for iv, r in rows.items() if r['class'] != 'after'))
        print("  pair A switch/held", {k: round(x, 2) for k, x in ratio.items()})
    (X / "two_seed_floor.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
