#!/usr/bin/env python3
"""Does the training-solution length explain the adapter switch as well as the teacher's think rate does?

Training solutions shorten as the teacher stops thinking, so R(lambda) and coverage could follow length
rather than the way of thinking. No length-matched training run exists, so this is only a re-analysis of
existing numbers:

  tokens    mean solution tokens per training problem at each lambda (split_stats.json: train_tokens /
            n_train, counted with the Qwen2.5-0.5B tokenizer used to filter every pair's splits)
  think     teacher think rate (experiments/think_rate.json)
  R         overlap ratio per cell (experiments/phi_resample_s012.json), and its pair mean
  coverage  shared-region coverage per cell (experiments/coverage_by_lambda.json)

Reported:
  1. the per-lambda table of the four quantities per pair;
  2. for each pair, the correlation over the five lambda intervals between the drop in a student's R and
     the drop in log tokens, averaged over the three students (the statistic of the permutation test,
     analyze_alignment.py), next to the same mean correlation with the drop in think rate;
  3. over all cells and lambda > 0 (45 points), the correlation of R with log relative length
     (tokens / tokens at lambda 0) and with think rate, and each partial correlation controlling for the
     other. The same for coverage.

Run from the repo root; writes experiments/length_confound.json.
"""

import json
import math
import statistics as st
from pathlib import Path

X = Path("experiments")
E = X / "lambda_lora_bits"
LAM = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
PAIRS = {"A": ("sweep", "A"), "B": ("pairB", "B"), "C": ("pairD", "D")}   # paper name -> (split dir, json key)
STUDENTS = ["Qwen2.5-0.5B", "Llama-3.2-1B", "OLMo-2-1B"]


def corr(x, y):
    mx, my = st.mean(x), st.mean(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return sxy / math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))


def partial(x, y, z):
    rxy, rxz, ryz = corr(x, y), corr(x, z), corr(y, z)
    return (rxy - rxz * ryz) / math.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))


def main():
    think = json.loads((X / "think_rate.json").read_text())
    res = json.loads((X / "phi_resample_s012.json").read_text())
    cov = json.loads((X / "coverage_by_lambda.json").read_text())
    out = {"per_pair": {}, "interval_corr": {}, "pooled": {}}
    pts = []
    for p, (split, key) in PAIRS.items():
        stats = json.loads((E / split / "splits" / "split_stats.json").read_text())
        n = stats["n_train"]
        tokens = {l: stats["per_lambda"][l]["train_tokens"] / n for l in LAM}
        rates = {l: 100 * think[key]["rates"][l] for l in LAM}
        R = {s: {l: res[f"{key}/{s}"]["rows"][l]["retention"] for l in LAM} for s in STUDENTS}
        C = {s: {l: cov[f"{key}/{s}"]["coverage"][l] for l in LAM} for s in STUDENTS}
        rows = {l: {"train_tokens": tokens[l], "think_rate": rates[l],
                    "R_mean": st.mean(R[s][l] for s in STUDENTS),
                    "coverage_mean": st.mean(C[s][l] for s in STUDENTS)} for l in LAM}
        out["per_pair"][p] = {"n_train": n, "rows": rows}
        d_len = [math.log(tokens[LAM[i]]) - math.log(tokens[LAM[i + 1]]) for i in range(5)]
        d_think = [rates[LAM[i]] - rates[LAM[i + 1]] for i in range(5)]
        d_R = [rows[LAM[i]]["R_mean"] - rows[LAM[i + 1]]["R_mean"] for i in range(5)]
        d_Rs = {s: [R[s][LAM[i]] - R[s][LAM[i + 1]] for i in range(5)] for s in STUDENTS}
        out["interval_corr"][p] = {"R_vs_log_tokens": st.mean(corr(d_Rs[s], d_len) for s in STUDENTS),
                                   "R_vs_think": st.mean(corr(d_Rs[s], d_think) for s in STUDENTS),
                                   "per_student_R_vs_log_tokens": {s: corr(d_Rs[s], d_len) for s in STUDENTS},
                                   "per_student_R_vs_think": {s: corr(d_Rs[s], d_think) for s in STUDENTS},
                                   "largest_log_token_drop": f"{LAM[d_len.index(max(d_len))]}->{LAM[d_len.index(max(d_len)) + 1]}",
                                   "largest_think_drop": f"{LAM[d_think.index(max(d_think))]}->{LAM[d_think.index(max(d_think)) + 1]}",
                                   "largest_R_drop": f"{LAM[d_R.index(max(d_R))]}->{LAM[d_R.index(max(d_R)) + 1]}",
                                   "log_token_drops": d_len, "R_drops": d_R}
        for s in STUDENTS:
            for l in LAM[1:]:
                pts.append((math.log(tokens[l] / tokens["0.0"]), rates[l], R[s][l], C[s][l]))
        print(f"pair {p}: tokens " + " ".join(f"{tokens[l]:.0f}" for l in LAM)
              + f" | interval r(R, log tokens) {out['interval_corr'][p]['R_vs_log_tokens']:.2f}"
              f" r(R, think) {out['interval_corr'][p]['R_vs_think']:.2f}")
    L, T, R, C = (list(v) for v in zip(*pts))
    for name, y in (("R", R), ("coverage", C)):
        out["pooled"][name] = {"r_log_rel_tokens": corr(y, L), "r_think": corr(y, T),
                               "partial_log_rel_tokens_given_think": partial(y, L, T),
                               "partial_think_given_log_rel_tokens": partial(y, T, L), "points": len(y)}
        print(name, {k: round(v, 3) for k, v in out["pooled"][name].items()})
    out["pooled"]["r_log_rel_tokens_vs_think"] = corr(L, T)
    print("collinearity r(log rel tokens, think) =", round(corr(L, T), 3))
    (X / "length_confound.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
