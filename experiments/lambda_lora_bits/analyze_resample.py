#!/usr/bin/env python3
"""Subspace overlap judged against a data-resample baseline instead of a seed baseline.

The seed baseline changed two things at once -- the data and the LoRA initialisation -- and the
initialisation turned out to carry most of the pre-threshold agreement, so every cell fell below that
baseline at lambda 0.2 and the crossing lambda stopped telling the pairs apart. Here the seed is fixed
at 0 and only the data moves: each lambda is trained twice, once on each disjoint half of its training
set (run_halves.sh).

  treatment  phi(half at lambda0, same half at lambda)   same initialisation, teacher changed
  baseline   phi(halfA at lambda, halfB at lambda)       same initialisation, data changed

Both sides share the initialisation and the data budget, so the difference is attributable to the
teacher alone. The crossing lambda is where a cell's treatment first drops below its own baseline,
and that is the quantity compared against the lambda at which the teacher stops emitting </think>.
"""

import argparse
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

EXP = Path(__file__).resolve().parent
LAMBDAS = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
HALVES = ["halfA", "halfB"]
CELLS = [
    ("A", "Qwen2.5-0.5B", "sweep"), ("A", "Llama-3.2-1B", "sweep_llama"), ("A", "OLMo-2-1B", "sweep_olmo2"),
    ("B", "Qwen2.5-0.5B", "pairB/qwen0.5b"), ("B", "Llama-3.2-1B", "pairB/llama1b"), ("B", "OLMo-2-1B", "pairB/olmo1b"),
    ("D", "Qwen2.5-0.5B", "pairD/qwen0.5b"), ("D", "Llama-3.2-1B", "pairD/llama1b"), ("D", "OLMo-2-1B", "pairD/olmo1b"),
]
# lambda interval in which each teacher pair stops emitting </think>, read off the teacher generations
# lambda at which each pair's </think> rate reaches zero, measured by measure_think_rate.py
TEACHER_ZERO = {"A": "0.4", "B": "0.8", "D": "0.8"}


def load_basis(path):
    """Top-r left singular vectors and Frobenius norm of dW = (alpha/r) B A, per layer/module."""
    config = json.loads((path.parent / "adapter_config.json").read_text())
    scale = config["lora_alpha"] / config["r"]
    parts = defaultdict(dict)
    with safe_open(path, "pt") as handle:
        for key in handle.keys():
            m = re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+_proj)", key)
            if m:
                parts[(int(m.group(1)), m.group(2))]["A" if "lora_A" in key else "B"] = handle.get_tensor(key).float()
    out = {}
    for key, v in parts.items():
        if len(v) != 2:
            continue
        B, A = v["B"] * scale, v["A"]
        Qb, Rb = torch.linalg.qr(B)
        Qa, Ra = torch.linalg.qr(A.T)
        U, S, _ = torch.linalg.svd(Rb @ Ra.T)
        out[key] = (Qb @ U, float((S ** 2).sum().sqrt()), B.shape[0])
    return out


def overlap(x, y):
    keys = set(x) & set(y)
    weight = sum(x[k][1] for k in keys)
    if not weight:
        return float("nan")
    return sum(float((x[k][0].T @ y[k][0]).pow(2).sum() / x[k][0].shape[1]) * x[k][1] for k in keys) / weight


def chance(x):
    weight = sum(v[1] for v in x.values())
    return sum(v[0].shape[1] / v[2] * v[1] for v in x.values()) / weight


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=str(EXP.parent / "phi_resample.json"))
    a = ap.parse_args()

    results = {}
    for pair, student, tag in CELLS:
        runs = EXP / tag / "runs"
        basis = {}
        for lam in LAMBDAS:
            for half in HALVES:
                p = runs / f"lambda{lam}_{half}" / "adapter" / "adapter_model.safetensors"
                if p.exists():
                    basis[(lam, half)] = load_basis(p)
        if len(basis) < 12:
            print(f"\n=== 쌍 {pair} / {student}: {len(basis)}/12 — 건너뜀")
            continue

        zero = TEACHER_ZERO[pair]
        print(f"\n=== 쌍 {pair} / {student}  (teacher 0% 도달 λ{zero})")
        print(f"{'lam':5s} {'처리 phi':>16s} {'재표집 기준선':>14s} {'유지율':>9s}")
        rows, crossing = {}, None
        for lam in LAMBDAS:
            # both quantities cross the halves, so neither inherits a shared training set and
            # lambda0 is not compared against itself
            treat = [overlap(basis[("0.0", "halfA")], basis[(lam, "halfB")]),
                     overlap(basis[("0.0", "halfB")], basis[(lam, "halfA")])]
            base = overlap(basis[(lam, "halfA")], basis[(lam, "halfB")])
            tm = st.mean(treat)
            ts = st.stdev(treat) if len(treat) > 1 else 0.0
            # retention: how much of the lambda0 direction is left beyond what resampling alone destroys
            ret = tm / base if base else float("nan")
            print(f"{lam:5s} {tm:10.3f} +-{ts:4.3f} {base:14.3f} {ret:9.3f}")
            rows[lam] = dict(treatment=tm, treatment_sd=ts, baseline=base, retention=ret)
        # the cliff is the largest drop in retention between adjacent lambdas
        drops = [(rows[LAMBDAS[i - 1]]["retention"] / rows[LAMBDAS[i]]["retention"], LAMBDAS[i])
                 for i in range(1, len(LAMBDAS)) if rows[LAMBDAS[i]]["retention"]]
        if drops:
            factor, crossing = max(drops)
            print(f"  가장 큰 낙차: λ{LAMBDAS[LAMBDAS.index(crossing) - 1]} → λ{crossing}  {factor:.1f}배")
        ref = basis[("0.0", "halfA")]
        print(f"  무작위 수준 {chance(ref):.3f} | adapter 분기 λ = {crossing or '없음'} | teacher 0% 도달 λ{zero}")
        results[f"{pair}/{student}"] = dict(rows=rows, crossing=crossing, teacher_zero=zero,
                                            chance=chance(ref), adapters=len(basis))

    Path(a.output).write_text(json.dumps(results, indent=2))
    print(f"\n{a.output}")
    if results:
        ok = sum(1 for v in results.values() if v["crossing"] == v["teacher_zero"])
        print(f"분기 λ가 teacher 전환 구간의 끝과 일치: {ok}/{len(results)}")


if __name__ == "__main__":
    main()
