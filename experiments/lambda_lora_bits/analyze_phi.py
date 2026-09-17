#!/usr/bin/env python3
"""Subspace overlap between LoRA adapters trained on different teacher lambdas, judged against a seed null.

phi(dW_i, dW_j) = ||U_i^T U_j||_F^2 / r over the top-r left singular vectors of dW = (alpha/r) B A,
averaged over layer/module with ||dW||_F as the weight (Hu et al. 2021, section 7.2).

For every (pair, student) cell and every lambda this prints
  phi(lambda0, lambda)  averaged over the three seed-matched adapter pairs, with its spread
  the seed null          same lambda, same data, different seed: the three seed pairs at that lambda
  the verdict            whether phi sits below the null by more than the two spreads combined

The crossing lambda is where a cell first drops below its own null, which is the quantity compared
against the lambda at which that teacher pair stops emitting the reasoning format.
"""

import argparse
import json
import math
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

EXP = Path(__file__).resolve().parent
LAMBDAS = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
SEEDS = [0, 1, 2]
CELLS = [
    ("A", "Qwen2.5-0.5B", "sweep"),
    ("A", "Llama-3.2-1B", "sweep_llama"),
    ("A", "OLMo-2-1B", "sweep_olmo2"),
    ("B", "Qwen2.5-0.5B", "pairB/qwen0.5b"),
    ("B", "Llama-3.2-1B", "pairB/llama1b"),
    ("B", "OLMo-2-1B", "pairB/olmo1b"),
    ("D", "Qwen2.5-0.5B", "pairD/qwen0.5b"),
    ("D", "Llama-3.2-1B", "pairD/llama1b"),
    ("D", "OLMo-2-1B", "pairD/olmo1b"),
]
# lambda interval in which each teacher pair stops emitting </think>, from the generations
# lambda at which each pair's </think> rate reaches zero, measured by measure_think_rate.py
TEACHER_ZERO = {"A": "0.4", "B": "0.8", "D": "0.8"}


def load_basis(path):
    """Top-r left singular vectors and Frobenius norm of dW, per layer/module."""
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
    ap.add_argument("--output", default=str(EXP.parent / "phi_table.json"))
    a = ap.parse_args()

    results = {}
    for pair, student, tag in CELLS:
        runs = EXP / tag / "runs"
        basis = {}
        for lam in LAMBDAS:
            for seed in SEEDS:
                p = runs / f"lambda{lam}_seed{seed}" / "adapter" / "adapter_model.safetensors"
                if p.exists():
                    basis[(lam, seed)] = load_basis(p)
        have = len(basis)
        if have < 12:
            print(f"\n=== 쌍 {pair} / {student}: adapter {have}/18 — 건너뜀")
            continue

        zero = TEACHER_ZERO[pair]
        print(f"\n=== 쌍 {pair} / {student}  (adapter {have}/18, teacher 0% 도달 λ{zero})")
        print(f"{'lam':5s} {'phi(l0,l)':>16s} {'seed 기준선':>16s} {'판정':>6s}")
        rows, crossing = {}, None
        for lam in LAMBDAS:
            cross = [overlap(basis[("0.0", s)], basis[(lam, s)])
                     for s in SEEDS if ("0.0", s) in basis and (lam, s) in basis]
            null = [overlap(basis[(lam, i)], basis[(lam, j)])
                    for i in SEEDS for j in SEEDS if i < j and (lam, i) in basis and (lam, j) in basis]
            if not cross or not null:
                continue
            cm, cs = st.mean(cross), (st.stdev(cross) if len(cross) > 1 else 0.0)
            nm, ns = st.mean(null), (st.stdev(null) if len(null) > 1 else 0.0)
            below = cm + cs < nm - ns
            if below and crossing is None and lam != "0.0":
                crossing = lam
            mark = "아래" if below else ("--" if lam == "0.0" else "위")
            print(f"{lam:5s} {cm:10.3f} +-{cs:4.3f} {nm:10.3f} +-{ns:4.3f} {mark:>6s}")
            rows[lam] = dict(phi=cm, phi_sd=cs, null=nm, null_sd=ns, below=below)
        ref = basis[("0.0", 0)]
        print(f"  무작위 수준 {chance(ref):.3f} | adapter 분기 λ = {crossing or '없음'} | teacher 0% 도달 λ{zero}")
        results[f"{pair}/{student}"] = dict(rows=rows, crossing=crossing,
                                           teacher_zero=zero, chance=chance(ref), adapters=have)

    Path(a.output).write_text(json.dumps(results, indent=2))
    print(f"\n{a.output}")
    match = [(k, v["crossing"], v["teacher_zero"]) for k, v in results.items()]
    ok = sum(1 for _, c, (lo, hi) in match if c == hi)
    print(f"분기 λ가 teacher 전환 구간의 끝과 일치: {ok}/{len(match)}")


if __name__ == "__main__":
    main()
