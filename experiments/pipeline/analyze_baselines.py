#!/usr/bin/env python3
import gzip
import json
import statistics as st
from pathlib import Path
from analyze_main_density import CELLS
from config import EVAL_PROBLEMS as PROBLEMS, EVAL_SAMPLES as SAMPLES, EXP as E, REFLECTIVE, RESULTS, SEEDS

CONDS = {
    "rm_pre": "lambda0.0_rm_pre",
    "rm_other": "lambda0.0_rm_decisive",
    "rm_scale": "lambda0.0_rm_scale",
    "gs_tau": "lambda0.0_gs_tau",
    "gs_energy": "lambda0.0_gs_energy",
    "nothink": "lambda0.0_nothink",
    "budget": "lambda0.0_budget",
    "merge0.5": "merge0.5",
    "merge0.7": "merge0.7",
    "rmseed": "lambda0.0_rmseed",
    "rmseed_other": "lambda0.0_rmseed_other",
    "rm_spanrand": "lambda0.0_rm_spanrand",
    "rm_post": "lambda0.0_rm_post",
}


def density(texts):
    return (
        1000 * sum((len(REFLECTIVE.findall(t)) for t in texts)) / max(sum((len(t.split()) for t in texts)), 1)
    )


def base(tag, bench, lam, seed):
    recs = json.loads((E / tag / bench / f"lambda{lam}_seed{seed}.json").read_text())["records"][:PROBLEMS]
    toks = [t for r in recs for t in r["tokens"][:SAMPLES]]
    trunc = [t for r in recs for t in r["truncated"][:SAMPLES]]
    corr = [bool(c) for r in recs for c in r["correct"][:SAMPLES]]
    return {
        "accuracy": sum(corr) / len(corr),
        "tokens": sum(toks) / len(toks),
        "truncated": sum(trunc) / len(trunc),
        "markers": density([r["response_0"] for r in recs]),
    }


def edited(tag, name, seed):
    d = E / tag / f"eval_main_s{seed}"
    if not (d / f"{name}.json").exists():
        return None
    s = json.loads((d / f"{name}.json").read_text())
    with gzip.open(d / f"{name}.samples.jsonl.gz", "rt") as f:
        first = [json.loads(l)["text"] for i, l in enumerate(f) if i % SAMPLES == 0]
    return {
        "accuracy": s["accuracy"],
        "tokens": s["avg_tokens"],
        "truncated": s["truncated_rate"],
        "markers": density(first),
    }


def main():
    out = {}
    for cell, (tag, bench) in CELLS.items():
        runs = {
            "base0": [base(tag, bench, "0.0", s) for s in SEEDS],
            "base1": [base(tag, bench, "1.0", s) for s in SEEDS],
        }
        for c, name in CONDS.items():
            runs[c] = [edited(tag, name, s) for s in SEEDS]
        row = {}
        for c, vals in runs.items():
            vals = [v for v in vals if v]
            if not vals:
                continue
            row[c] = {
                k: {
                    "mean": st.mean((v[k] for v in vals)),
                    "sd": st.stdev((v[k] for v in vals)) if len(vals) > 1 else float("nan"),
                    "seeds": [v[k] for v in vals],
                }
                for k in vals[0]
            }
        out[cell] = row
        print(f"== {cell}")
        for c, r in row.items():
            print(
                f"  {c:9s} acc {r['accuracy']['mean']:.3f}±{r['accuracy']['sd']:.3f}  tok {r['tokens']['mean']:6.0f}  trunc {r['truncated']['mean']:.2f}  markers {r['markers']['mean']:5.1f}  (n={len(r['accuracy']['seeds'])})"
            )
    (RESULTS / "baselines.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
