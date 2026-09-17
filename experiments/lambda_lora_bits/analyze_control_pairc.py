#!/usr/bin/env python3
"""Pair C: the control that separates the reasoning format's tag from the reasoning itself.

Pair C's instruct endpoint uses a template that never emits </think>, so the tag rate is 0% at every
lambda while the reflective-vocabulary density still collapses. If the adapter subspace cliff lands
where that density falls, the cliff cannot be an artefact of counting a literal string.

Only seed 0 exists for this pair, so both adapters in every comparison share one LoRA initialisation.
That lifts the whole curve and makes the absolute values incomparable with the three main pairs -- but
it is constant along lambda, so it cannot manufacture a cliff. Read the position of the drop, not its
height.
"""

import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

EXP = Path(__file__).resolve().parent
LAMBDAS = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
CELLS = [("Qwen2.5-0.5B", "pairC/qwen0.5b"), ("Llama-3.2-1B", "pairC/llama1b"), ("OLMo-2-1B", "pairC/olmo1b")]
# hedging and self-correction markers; their density stands in for reasoning the tag cannot show
REFLECTIVE = re.compile(
    r"\b(wait|hmm|actually|but let me|let me check|recheck|verify|reconsider|alternatively|on second thought)\b",
    re.I)


def load_basis(path):
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
        out[key] = (Qb @ U, float((S ** 2).sum().sqrt()))
    return out


def overlap(x, y):
    keys = set(x) & set(y)
    weight = sum(x[k][1] for k in keys)
    return sum(float((x[k][0].T @ y[k][0]).pow(2).sum() / x[k][0].shape[1]) * x[k][1] for k in keys) / weight


def main():
    print("teacher 쪽 (원본 생성물)")
    print(f"{'λ':5s} {'</think>':>9s} {'반성어휘/1k':>12s} {'평균토큰':>9s}")
    teacher = {}
    for lam in LAMBDAS:
        rows = [json.loads(line) for line in (EXP / "pairC" / "cot" / f"lambda{lam}.jsonl").open()]
        tokens = sum(r["generated_tokens"] for r in rows)
        rate = sum("</think>" in r["cot_response"] for r in rows) / len(rows)
        density = sum(len(REFLECTIVE.findall(r["cot_response"])) for r in rows) / tokens * 1000
        teacher[lam] = dict(think_rate=rate, reflective_per_1k=density, avg_tokens=tokens / len(rows))
        print(f"{lam:5s} {rate:9.0%} {density:12.2f} {tokens/len(rows):9.0f}")

    print("\nadapter 쪽 φ(λ0, λ)")
    print(f"{'학생':14s} " + "".join(f"{('λ' + l):>8s}" for l in LAMBDAS) + "   최대 낙차")
    results = {"teacher": teacher, "cells": {}}
    for student, tag in CELLS:
        basis = {}
        for lam in LAMBDAS:
            for seed in (0, 1, 2):
                p = EXP / tag / "runs" / f"lambda{lam}_seed{seed}" / "adapter" / "adapter_model.safetensors"
                if p.exists():
                    basis[lam] = load_basis(p)
                    break
        if len(basis) < len(LAMBDAS):
            print(f"{student:14s} {len(basis)}/{len(LAMBDAS)} — 건너뜀")
            continue
        phi = {lam: overlap(basis["0.0"], basis[lam]) for lam in LAMBDAS}
        drops = [(phi[LAMBDAS[i - 1]] / phi[LAMBDAS[i]], LAMBDAS[i]) for i in range(1, len(LAMBDAS)) if phi[LAMBDAS[i]]]
        factor, at = max(drops)
        print(f"{student:14s} " + "".join(f"{phi[l]:8.3f}" for l in LAMBDAS)
              + f"   λ{LAMBDAS[LAMBDAS.index(at) - 1]}→λ{at} {factor:.1f}배")
        results["cells"][student] = dict(phi=phi, cliff=at, factor=factor)

    path = EXP.parent / "control_pairc.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"\n{path}")


if __name__ == "__main__":
    main()
