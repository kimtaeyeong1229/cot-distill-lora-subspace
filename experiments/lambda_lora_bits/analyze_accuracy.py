#!/usr/bin/env python3
"""Accuracy across the lambda sweep, with the truncation artifact separated out.

A student trained on long teacher CoTs writes long answers and hits the generation limit, so a
truncated sample is scored wrong for running out of room rather than for reasoning badly. Raw accuracy
therefore rises across the teacher's transition partly because the post-transition student writes short
answers. Re-scoring on the non-truncated samples only isolates how much of the rise is reasoning:

  raw          every sample, truncated ones counted wrong (what the benchmark reports)
  non-trunc    samples that finished, so the comparison is between answers that were actually completed
  gap          raw minus non-trunc: the part of the number that is the generation limit, not the student

Each (lambda, seed) cell is one adapter and each adapter was sampled n times per problem; samples are
pooled across seeds, and the pre/post effect is the mean over lambdas below vs at-or-above the cliff.
"""

import argparse
import json
import statistics as st
from pathlib import Path

EXP = Path(__file__).resolve().parent
LAMBDAS = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
SEEDS = [0, 1, 2]
CELLS = [
    ("A", "Qwen2.5-0.5B", "sweep", "0.4"), ("A", "Llama-3.2-1B", "sweep_llama", "0.4"),
    ("A", "OLMo-2-1B", "sweep_olmo2", "0.4"),
    ("B", "Qwen2.5-0.5B", "pairB/qwen0.5b", "0.6"), ("B", "Llama-3.2-1B", "pairB/llama1b", "0.6"),
    ("B", "OLMo-2-1B", "pairB/olmo1b", "0.6"),
    ("D", "Qwen2.5-0.5B", "pairD/qwen0.5b", "0.6"), ("D", "Llama-3.2-1B", "pairD/llama1b", "0.8"),
    ("D", "OLMo-2-1B", "pairD/olmo1b", "0.8"),
]


def read_cell(directory):
    """Per-lambda pooled statistics over the seeds that are present."""
    out = {}
    for lam in LAMBDAS:
        raw, keep, trunc, toks = [], [], [], []
        for seed in SEEDS:
            path = directory / f"lambda{lam}_seed{seed}.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            for rec in data["records"]:
                for ok, cut, n in zip(rec["correct"], rec["truncated"], rec["tokens"]):
                    raw.append(bool(ok))
                    trunc.append(bool(cut))
                    toks.append(n)
                    if not cut:
                        keep.append(bool(ok))
        if raw:
            out[lam] = dict(raw=sum(raw) / len(raw), n=len(raw),
                            non_trunc=(sum(keep) / len(keep)) if keep else float("nan"),
                            kept=len(keep), trunc_rate=sum(trunc) / len(trunc),
                            avg_tokens=sum(toks) / len(toks))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", choices=["eval_val", "eval_math500"], default="eval_math500")
    ap.add_argument("--output", default="")
    a = ap.parse_args()

    results, effects = {}, []
    for pair, student, tag, cliff in CELLS:
        rows = read_cell(EXP / tag / a.bench)
        if len(rows) < 6:
            print(f"\n=== 쌍 {pair} / {student}: λ {len(rows)}/6 — 건너뜀")
            continue
        i = LAMBDAS.index(cliff)
        pre = [rows[l] for l in LAMBDAS[:i]]
        post = [rows[l] for l in LAMBDAS[i:]]
        d_raw = st.mean(x["raw"] for x in post) - st.mean(x["raw"] for x in pre)
        d_keep = st.mean(x["non_trunc"] for x in post) - st.mean(x["non_trunc"] for x in pre)

        print(f"\n=== 쌍 {pair} / {student}  (절벽 λ {cliff}, {a.bench})")
        print(f"{'lam':5s} {'raw':>7s} {'비잘림':>8s} {'잘림률':>7s} {'평균토큰':>9s} {'표본':>7s}")
        for lam in LAMBDAS:
            r = rows[lam]
            print(f"{lam:5s} {r['raw']:7.1%} {r['non_trunc']:8.1%} {r['trunc_rate']:7.1%} "
                  f"{r['avg_tokens']:9.0f} {r['n']:7d}")
        print(f"  절벽 전후 차: raw {d_raw:+.1%} / 비잘림 {d_keep:+.1%} "
              f"→ 잘림이 설명하는 몫 {d_raw - d_keep:+.1%}")
        results[f"{pair}/{student}"] = dict(rows=rows, cliff=cliff, delta_raw=d_raw, delta_non_trunc=d_keep)
        effects.append((d_raw, d_keep))

    if effects:
        mr, mk = st.mean(x for x, _ in effects), st.mean(y for _, y in effects)
        print(f"\n{len(effects)}칸 평균 — raw {mr:+.1%}, 비잘림 {mk:+.1%}, "
              f"잘림 기여분 {mr - mk:+.1%} ({(mr - mk) / mr:.0%} of raw)" if mr else "")
    out = a.output or str(EXP.parent / f"accuracy_{a.bench}.json")
    Path(out).write_text(json.dumps(results, indent=2))
    print(out)


if __name__ == "__main__":
    main()
