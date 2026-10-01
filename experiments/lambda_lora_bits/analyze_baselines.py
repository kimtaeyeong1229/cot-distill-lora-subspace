#!/usr/bin/env python3
"""The think-off edit against the baselines a reviewer would ask for, under the shared student setting.

Per cell and seed (first 200 MATH-500 problems x 2 samples): accuracy and mean tokens over both samples,
truncation, and the reflective-marker density of each problem's first sample, for

  base0 / base1        the unedited lambda0 / lambda1 adapters (eval_math500[_7680], as Table 1)
  rm_pre               the consensus region removed from the lambda0 adapter (the edit)
  rm_other / rm_scale  the energy-matched removals of Table 6
  gs_tau / gs_energy   Gold-Switch: the top of the lambda0 update's own spectrum removed, at the paper's
                       tau = 0.6 and at rm_pre's energy
  nothink / budget     NoThinking prefill, and cutting at rm_pre's mean length with a forced answer
  merge0.5 / merge0.7  task-arithmetic merge of the lambda0 and lambda1 adapters

Tables report mean +- SD over seeds 0-2. Run from the repo root; writes experiments/baselines.json.
"""

import gzip
import json
import statistics as st
from pathlib import Path

from analyze_main_density import CELLS, PROBLEMS, REFLECTIVE, SAMPLES

E = Path("experiments/lambda_lora_bits")
SEEDS = [0, 1, 2]
CONDS = {"rm_pre": "lambda0.0_rm_pre", "rm_other": "lambda0.0_rm_decisive", "rm_scale": "lambda0.0_rm_scale",
         "gs_tau": "lambda0.0_gs_tau", "gs_energy": "lambda0.0_gs_energy", "nothink": "lambda0.0_nothink",
         "budget": "lambda0.0_budget", "merge0.5": "merge0.5", "merge0.7": "merge0.7",
         "rmseed": "lambda0.0_rmseed", "rmseed_other": "lambda0.0_rmseed_other"}


def density(texts):
    return 1000 * sum(len(REFLECTIVE.findall(t)) for t in texts) / max(sum(len(t.split()) for t in texts), 1)


def base(tag, bench, lam, seed):
    recs = json.loads((E / tag / bench / f"lambda{lam}_seed{seed}.json").read_text())["records"][:PROBLEMS]
    toks = [t for r in recs for t in r["tokens"][:SAMPLES]]
    trunc = [t for r in recs for t in r["truncated"][:SAMPLES]]
    corr = [bool(c) for r in recs for c in r["correct"][:SAMPLES]]
    return {"accuracy": sum(corr) / len(corr), "tokens": sum(toks) / len(toks),
            "truncated": sum(trunc) / len(trunc), "markers": density([r["response_0"] for r in recs])}


def edited(tag, name, seed):
    d = E / tag / f"eval_main_s{seed}"
    if not (d / f"{name}.json").exists():
        return None
    s = json.loads((d / f"{name}.json").read_text())
    with gzip.open(d / f"{name}.samples.jsonl.gz", "rt") as f:
        first = [json.loads(l)["text"] for i, l in enumerate(f) if i % SAMPLES == 0]
    return {"accuracy": s["accuracy"], "tokens": s["avg_tokens"], "truncated": s["truncated_rate"],
            "markers": density(first)}


def main():
    out = {}
    for cell, (tag, bench) in CELLS.items():
        runs = {"base0": [base(tag, bench, "0.0", s) for s in SEEDS],
                "base1": [base(tag, bench, "1.0", s) for s in SEEDS]}
        for c, name in CONDS.items():
            runs[c] = [edited(tag, name, s) for s in SEEDS]
        row = {}
        for c, vals in runs.items():
            vals = [v for v in vals if v]
            if not vals:
                continue
            row[c] = {k: {"mean": st.mean(v[k] for v in vals),
                          "sd": st.stdev(v[k] for v in vals) if len(vals) > 1 else float("nan"),
                          "seeds": [v[k] for v in vals]} for k in vals[0]}
        out[cell] = row
        print(f"== {cell}")
        for c, r in row.items():
            print(f"  {c:9s} acc {r['accuracy']['mean']:.3f}±{r['accuracy']['sd']:.3f}  tok {r['tokens']['mean']:6.0f}"
                  f"  trunc {r['truncated']['mean']:.2f}  markers {r['markers']['mean']:5.1f}  (n={len(r['accuracy']['seeds'])})")
    Path("experiments/baselines.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
