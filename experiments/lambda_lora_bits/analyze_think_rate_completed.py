#!/usr/bin/env python3
"""Teacher think rate on completed solutions and on the training subset (review R2-W6, read only).

think_rate.json (measure_think_rate.py) counts </think> in every one of the 4,000 teacher generations per pair
and lambda (<pair>/cot/lambda<l>.jsonl, field cot_response). A generation that hits the 8,192-token limit
before closing </think> counts as "not thinking" there, so part of that rate is truncation. Here the same
count is made on two subsets of the same files:

  completed   generations with finish_reason == "stop" (did not hit the generation limit)
  train       generations of the training problems (<pair>/splits/split_ids.json "train"; the same problems at
              every lambda of a pair, kept only if every teacher solved them completely and correctly)

Both are compared with the all-generation rate (recomputed here and checked against think_rate.json), and the
Table 4 permutation test (analyze_alignment.py, adapter profiles from phi_resample_s012.json, as in
alignment_s012.json) is rerun with the teacher profile taken from each subset. Pair C is stored as pairD and
keyed "D", as in think_rate.json.

Run from the repo root; writes experiments/think_rate_completed.json.
"""

import json
import statistics as st
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

X = Path("experiments")
E = X / "lambda_lora_bits"
sys.path.insert(0, str(E))
from analyze_alignment import LAMBDAS, pearson, permutation_p  # noqa: E402

PAIRS = {"A": "sweep", "B": "pairB", "D": "pairD"}


def rates_for(tag):
    train = set(json.loads((E / tag / "splits" / "split_ids.json").read_text())["train"])
    per = {}
    for lam in LAMBDAS:
        rows = [json.loads(line) for line in open(E / tag / "cot" / f"lambda{lam}.jsonl")]
        think = ["</think>" in r["cot_response"] for r in rows]
        done = [r["finish_reason"] == "stop" for r in rows]
        tr = [r["unique_id"] in train for r in rows]
        n_done, n_tr = sum(done), sum(tr)
        per[lam] = {"n_all": len(rows), "all": sum(think) / len(rows),
                    "n_completed": n_done, "completed_share": n_done / len(rows),
                    "completed": sum(t for t, d in zip(think, done) if d) / max(n_done, 1),
                    "n_train": n_tr, "train": sum(t for t, x in zip(think, tr) if x) / max(n_tr, 1),
                    "truncated_without_tag": sum((not d) and (not t) for t, d in zip(think, done)) / len(rows)}
    return per


def alignment(rates_by_pair, res):
    """Table 4: per-cell exact permutation r/p, pair-level mean r and p, and the pooled p over three pairs."""
    perms = [list(q) for q in permutations(range(len(LAMBDAS) - 1))]
    cells, summary, null, pooled = {}, {}, [], 0.0
    for pair in PAIRS:
        rates = rates_by_pair[pair]
        teacher = [rates[LAMBDAS[i - 1]] - rates[LAMBDAS[i]] for i in range(1, len(LAMBDAS))]
        profiles = []
        for key, cell in sorted(res.items()):
            if key.split("/")[0] != pair:
                continue
            ad = [cell["rows"][LAMBDAS[i - 1]]["retention"] - cell["rows"][LAMBDAS[i]]["retention"]
                  for i in range(1, len(LAMBDAS))]
            r, p = permutation_p(teacher, ad)
            cells[key] = {"r": r, "p": p}
            profiles.append(ad)
        obs = st.mean(pearson(teacher, ad) for ad in profiles)
        dist = [st.mean(pearson([teacher[j] for j in q], ad) for ad in profiles) for q in perms]
        summary[pair] = {"teacher_profile": teacher, "mean_r": obs,
                         "p": sum(x >= obs - 1e-12 for x in dist) / len(dist), "cells": len(profiles)}
        null.append(np.array(dist))
        pooled += obs
    A, B, C = null
    total = A[:, None, None] + B[None, :, None] + C[None, None, :]
    summary["pooled"] = {"mean_r": pooled / 3, "p": float((total >= pooled - 1e-9).mean()), "orderings": int(total.size)}
    return {"cells": cells, "summary": summary}


def main():
    ref = json.loads((X / "think_rate.json").read_text())
    res = json.loads((X / "phi_resample_s012.json").read_text())
    out = {"per_pair": {}, "alignment": {}}
    for pair, tag in PAIRS.items():
        per = rates_for(tag)
        for lam in LAMBDAS:
            assert abs(per[lam]["all"] - ref[pair]["rates"][lam]) < 1e-9, (pair, lam)
        steps = {f"{a}->{b}": {s: per[b][s] - per[a][s] for s in ("all", "completed", "train")}
                 for a, b in zip(LAMBDAS, LAMBDAS[1:])}
        out["per_pair"][pair] = {"per_lambda": per, "steps": steps}
        print(f"{pair}: " + " | ".join(f"λ{l} all {100 * per[l]['all']:5.1f} done {100 * per[l]['completed']:5.1f}"
                                      f" ({100 * per[l]['completed_share']:3.0f}% done) train {100 * per[l]['train']:5.1f}"
                                      for l in LAMBDAS), flush=True)
    for subset in ("all", "completed", "train"):
        rates = {p: {l: out["per_pair"][p]["per_lambda"][l][subset] for l in LAMBDAS} for p in PAIRS}
        out["alignment"][subset] = alignment(rates, res)
        s = out["alignment"][subset]["summary"]
        print(f"Table 4 with {subset:9s}: " + " ".join(f"{p} r={s[p]['mean_r']:.3f} p={s[p]['p']:.3f}" for p in PAIRS)
              + f" | pooled r={s['pooled']['mean_r']:.3f} p={s['pooled']['p']:.4f}", flush=True)
    ref_align = json.loads((X / "alignment_s012.json").read_text())["_summary"]
    out["alignment_check_vs_alignment_s012"] = {
        p: abs(out["alignment"]["all"]["summary"][p]["mean_r"] - ref_align[p]["mean_r"]) < 1e-9 for p in ref_align}
    out["definitions"] = {"completed": "finish_reason == 'stop' (did not hit the 8,192-token teacher limit)",
                          "train": "splits/split_ids.json train ids", "tag": "</think> in cot_response",
                          "pair_C_key": "D"}
    (X / "think_rate_completed.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
