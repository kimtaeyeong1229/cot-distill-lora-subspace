#!/usr/bin/env python3
"""How much of the shared region does the two-seed region recover?

6.2 builds a region without interpolated teachers: only the other two seeds' lambda-0 adapters, with
the 0.75 eigenvalue cut (ablate_subspace.py --region seed). This compares it, module by module, with
the shared region V built the usual way (all pre-switch adapters, 0.5 cut), both leave-one-out for the
seed-0 lambda-0 adapter that the edits are applied to. The overlap is ||V_seed^T V||_F^2 / k_V, the
share of V's dimensions that the seed region spans, averaged over modules with weight ||dW||_F of the
edited adapter (as for phi in analyze_resample.py).

Writes experiments/seed_region_overlap.json.
"""

import json
from pathlib import Path

import torch

from ablate_subspace import LAMBDAS, SEEDS, consensus, left_basis, read_adapter

from config import CELL_DIR, CELL_NAMES, EXP as E, RESULTS, SWITCH

CELLS = {cell: (CELL_DIR[cell], SWITCH[cell[0]]) for cell in CELL_NAMES}   # cell -> (directory, switch lambda)


def bases(tag, lam, seed):
    path = E / tag / "runs" / f"lambda{lam}_seed{seed}" / "adapter"
    config = json.loads((path / "adapter_config.json").read_text())
    scale = config["lora_alpha"] / config["r"]
    raw, parts = read_adapter(path / "adapter_model.safetensors")
    out, norms = {}, {}
    for key, names in parts.items():
        if len(names) == 2:
            B, A = raw[names["B"]].float(), raw[names["A"]].float()
            out[key] = left_basis(B, A, scale)
            norms[key] = float(torch.linalg.matrix_norm(scale * B @ A))
    return out, norms


def main():
    result = {}
    for cell, (tag, cliff) in CELLS.items():
        pre = [(l, s) for l in LAMBDAS[:LAMBDAS.index(cliff)] for s in SEEDS if (l, s) != ("0.0", 0)]
        seed_members = [("0.0", s) for s in SEEDS if s != 0]
        loaded = {m: bases(tag, *m)[0] for m in set(pre) | set(seed_members)}
        _, weight = bases(tag, "0.0", 0)
        num = den = dims_v = dims_s = 0.0
        for key, w in weight.items():
            V, kv = consensus([loaded[m][key] for m in pre], threshold=0.5)
            S, ks = consensus([loaded[m][key] for m in seed_members], threshold=0.75)
            num += w * float((S.T @ V).pow(2).sum()) / kv
            den += w
            dims_v += kv
            dims_s += ks
        n = len(weight)
        result[cell] = {"overlap": num / den, "dims_shared": dims_v / n, "dims_seed": dims_s / n}
        print(f"{cell:8s} overlap {num / den:.2f}  dims shared {dims_v / n:.1f}  seed {dims_s / n:.1f}", flush=True)
    vals = [r["overlap"] for r in result.values()]
    print(f"range {min(vals):.2f}-{max(vals):.2f}  mean {sum(vals) / len(vals):.2f}")
    (RESULTS / "seed_region_overlap.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
