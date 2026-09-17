#!/usr/bin/env python3
"""Absolute-quantity view: student bits = teacher floor H + KL, with train/held gap (sample-specific fit)."""
import json, math, os
from pathlib import Path
D = Path("experiments/lambda_lora_bits/pilot_validity")
LAMS = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
STUD = ["qwen0.5b", "qwen1.5b", "qwen3b", "llama1b", "olmo1b"]
rows = []
print(f"{'student':9s} {'λ':4s} {'floor':>6s} {'before':>7s} {'after_tr':>8s} {'after_he':>8s} {'여유KL':>7s} {'전이':>6s} {'격차':>6s} {'strictMem':>9s} {'P1':>3s} {'P2':>3s}")
for s in STUD:
    for lam in LAMS:
        fp, sp = D / f"floor_lambda{lam}.json", D / f"student_{s}_lambda{lam}.json"
        if not (fp.exists() and sp.exists()):
            continue
        F = {k: {x["unique_id"]: x["floor_bits"] for x in v} for k, v in json.load(open(fp))["sets"].items()}
        S = json.load(open(sp))["sets"]
        def mean(lab, key, side=None):
            xs = [x for x in S[lab] if x["unique_id"] in F[lab]]
            if side:
                return sum(x[side][key] for x in xs) / len(xs)
            return sum(F[lab][x["unique_id"]] for x in xs) / len(xs)
        fl_tr, fl_he = mean("train", None), mean("held", None)
        b_tr, a_tr = mean("train", "bits", "base"), mean("train", "bits", "tuned")
        b_he, a_he = mean("held", "bits", "base"), mean("held", "bits", "tuned")
        kl, tr_gain, gap = b_he - fl_he, b_he - a_he, a_he - a_tr
        strict = max(0.0, fl_tr - a_tr)
        p1 = "OK" if a_he >= fl_he else "X"
        p2 = "OK" if tr_gain <= kl + 1e-9 else "X"
        print(f"{s:9s} {lam:4s} {fl_he:6.1f} {b_he:7.1f} {a_tr:8.1f} {a_he:8.1f} {kl:7.1f} {tr_gain:6.1f} {gap:6.1f} {strict:9.1f} {p1:>3s} {p2:>3s}")
        rows.append((s, float(lam), kl, tr_gain, gap, b_he, a_he, a_tr, fl_he))
def corr(x, y):
    n = len(x); mx, my = sum(x)/n, sum(y)/n
    sx = math.sqrt(sum((a-mx)**2 for a in x)); sy = math.sqrt(sum((b-my)**2 for b in y))
    return sum((a-mx)*(b-my) for a, b in zip(x, y))/(sx*sy) if sx and sy else float("nan")
kl = [r[2] for r in rows]; gap = [r[4] for r in rows]; tg = [r[3] for r in rows]
print(f"\ncells={len(rows)}  corr(여유KL, 격차)={corr(kl, gap):+.2f}  corr(여유KL, 전이)={corr(tg, kl):+.2f}")
print("relative: corr(log KL, 격차/여유) =", round(corr([math.log(max(k,1e-9)) for k in kl], [g/k if k else 0 for g, k in zip(gap, kl)]), 2))
for s in STUD:
    sub = [r for r in rows if r[0] == s]
    if len(sub) > 2:
        print(f"  within {s:9s}: corr(KL,격차)={corr([r[2] for r in sub],[r[4] for r in sub]):+.2f}  KL range {min(r[2] for r in sub):.0f}-{max(r[2] for r in sub):.0f}")
