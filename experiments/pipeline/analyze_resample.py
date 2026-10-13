#!/usr/bin/env python3
import argparse
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path
import torch
from safetensors import safe_open
from config import CELL_DIR, EXP, LAMBDAS, PAIR_KEY, PAIRS, RESULTS, SEEDS, STUDENT_SHORT, STUDENTS

HALVES = ["halfA", "halfB"]
CELLS = [(PAIR_KEY[p], STUDENT_SHORT[s], CELL_DIR[f"{p}/{s}"]) for p in PAIRS for s in STUDENTS]
TEACHER_ZERO = {"A": "0.4", "B": "0.8", "C": "0.8"}


def load_basis(path):
    config = json.loads((path.parent / "adapter_config.json").read_text())
    scale = config["lora_alpha"] / config["r"]
    parts = defaultdict(dict)
    with safe_open(path, "pt") as handle:
        for key in handle.keys():
            m = re.search("layers\\.(\\d+)\\.(?:self_attn|mlp)\\.(\\w+_proj)", key)
            if m:
                parts[int(m.group(1)), m.group(2)]["A" if "lora_A" in key else "B"] = handle.get_tensor(
                    key
                ).float()
    out = {}
    for key, v in parts.items():
        if len(v) != 2:
            continue
        B, A = (v["B"] * scale, v["A"])
        Qb, Rb = torch.linalg.qr(B)
        Qa, Ra = torch.linalg.qr(A.T)
        U, S, _ = torch.linalg.svd(Rb @ Ra.T)
        out[key] = (Qb @ U, float((S**2).sum().sqrt()), B.shape[0])
    return out


def overlap(x, y):
    keys = set(x) & set(y)
    weight = sum((x[k][1] for k in keys))
    if not weight:
        return float("nan")
    return (
        sum((float((x[k][0].T @ y[k][0]).pow(2).sum() / x[k][0].shape[1]) * x[k][1] for k in keys)) / weight
    )


def chance(x):
    weight = sum((v[1] for v in x.values()))
    return sum((v[0].shape[1] / v[2] * v[1] for v in x.values())) / weight


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=str(RESULTS / "phi_resample_s012.json"))
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    a = ap.parse_args()
    results = {}
    for pair, student, tag in CELLS:
        per_seed = {}
        for seed in a.seeds:
            r = cell_rows(pair, student, tag, seed)
            if r is not None:
                per_seed[seed] = r
        if len(per_seed) < len(a.seeds):
            continue
        rows = {}
        for lam in LAMBDAS:
            vals = {
                k: [per_seed[s][0][lam][k] for s in a.seeds] for k in ("treatment", "baseline", "retention")
            }
            rows[lam] = dict(
                treatment=st.mean(vals["treatment"]),
                baseline=st.mean(vals["baseline"]),
                retention=st.mean(vals["retention"]),
                retention_sd=st.stdev(vals["retention"]) if len(a.seeds) > 1 else 0.0,
                retention_seeds=vals["retention"],
            )
        if len(a.seeds) > 1:
            print(f"\n=== pair {pair} / {student}: mean R over seeds {a.seeds}")
            print(
                "  "
                + "  ".join(
                    (
                        f"λ{lam} {rows[lam]['retention']:.2f}±{rows[lam]['retention_sd']:.2f}"
                        for lam in LAMBDAS
                    )
                )
            )
        results[f"{pair}/{student}"] = dict(
            rows=rows,
            seeds=a.seeds,
            teacher_zero=TEACHER_ZERO[pair],
            chance=st.mean((per_seed[s][1] for s in a.seeds)),
            per_seed={str(s): per_seed[s][0] for s in a.seeds},
        )
    Path(a.output).write_text(json.dumps(results, indent=2))
    print(f"\n{a.output}")


def cell_rows(pair, student, tag, seed):
    runs = EXP / tag / "runs"
    suffix = "" if seed == 0 else f"_seed{seed}"
    basis = {}
    for lam in LAMBDAS:
        for half in HALVES:
            p = runs / f"lambda{lam}_{half}{suffix}" / "adapter" / "adapter_model.safetensors"
            if p.exists():
                basis[lam, half] = load_basis(p)
    if len(basis) < 12:
        print(f"\n=== pair {pair} / {student} seed {seed}: {len(basis)}/12 half adapters, skipped")
        return None
    zero = TEACHER_ZERO[pair]
    print(f"\n=== pair {pair} / {student} seed {seed}  (teacher think rate reaches 0% at λ{zero})")
    print(f"{'lam':5s} {'treatment phi':>16s} {'data baseline':>14s} {'R':>9s}")
    rows, crossing = ({}, None)
    for lam in LAMBDAS:
        treat = [
            overlap(basis["0.0", "halfA"], basis[lam, "halfB"]),
            overlap(basis["0.0", "halfB"], basis[lam, "halfA"]),
        ]
        base = overlap(basis[lam, "halfA"], basis[lam, "halfB"])
        tm = st.mean(treat)
        ts = st.stdev(treat) if len(treat) > 1 else 0.0
        ret = tm / base if base else float("nan")
        print(f"{lam:5s} {tm:10.3f} +-{ts:4.3f} {base:14.3f} {ret:9.3f}")
        rows[lam] = dict(treatment=tm, treatment_sd=ts, baseline=base, retention=ret)
    drops = [
        (rows[LAMBDAS[i - 1]]["retention"] / rows[LAMBDAS[i]]["retention"], LAMBDAS[i])
        for i in range(1, len(LAMBDAS))
        if rows[LAMBDAS[i]]["retention"]
    ]
    if drops:
        factor, crossing = max(drops)
        print(f"  largest drop: λ{LAMBDAS[LAMBDAS.index(crossing) - 1]} → λ{crossing}  x{factor:.1f}")
    ref = basis["0.0", "halfA"]
    print(
        f"  chance {chance(ref):.3f} | adapter switch λ = {crossing or 'none'} | teacher reaches 0% at λ{zero}"
    )
    return (rows, chance(ref))


if __name__ == "__main__":
    main()
