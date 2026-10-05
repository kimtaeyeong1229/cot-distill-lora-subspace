#!/usr/bin/env python3
"""Where the shared region sits inside the adapters: by module type and depth, and its share of update energy.

The region is the one of Table 5 (eigenvalue cut 0.5 over every seed of the pre-switch lambdas), built per
layer and module. For each module the script records the region's dimension, and for the lambda = 0 and
lambda = 1 adapters the share of the update's energy (squared Frobenius norm of dW = scale * B A) that lies
in the region, ||V^T dW||^2 / ||dW||^2, next to the share k / d a random k-dimensional subspace would hold.
Results are averaged over the three seeds and grouped by module type and by thirds of depth.

Writes experiments/region_anatomy.json.
"""

import json
import statistics as st
from collections import defaultdict

import torch

from ablate_subspace import consensus, left_basis, read_adapter
from config import CELL_DIR, EXP as E, LAMBDAS as LAM, PAIR_DIR, RESULTS as X, SEEDS, STUDENT_SHORT

PRE = {"A": ["0.0", "0.2"], "B": ["0.0", "0.2", "0.4"], "C": ["0.0", "0.2", "0.4"]}
RUNS = {p: {s: CELL_DIR[f"{p}/{s}"] for s in STUDENT_SHORT} for p in PAIR_DIR}   # pair -> student -> cell dir


def load(tag, lam, seed):
    path = E / tag / "runs" / f"lambda{lam}_seed{seed}" / "adapter"
    config = json.loads((path / "adapter_config.json").read_text())
    scale = config["lora_alpha"] / config["r"]
    raw, parts = read_adapter(path / "adapter_model.safetensors")
    out = {}
    for key, names in parts.items():
        if len(names) == 2:
            B, A = raw[names["B"]].float(), raw[names["A"]].float()
            out[key] = (left_basis(B, A, scale), scale * B, A)
    return out


def energy_in(V, sB, A):
    """Share of ||sB A||_F^2 inside span(V), without forming the full update."""
    G = A @ A.T                                  # r x r
    total = float(torch.trace(sB.T @ sB @ G))
    VB = V.T @ sB
    return float(torch.trace(VB.T @ VB @ G)) / total, total


def main():
    out = {}
    for pair, cells in RUNS.items():
        for short, tag in cells.items():
            ads = {(l, s): load(tag, l, s) for l in PRE[pair] + ["1.0"] for s in SEEDS}
            if "0.0" not in PRE[pair]:
                raise SystemExit("lambda 0 must be pre-switch")
            keys = sorted(set.intersection(*[set(a) for a in ads.values()]))
            layers = sorted({k[0] for k in keys})
            third = {l: min(2, 3 * i // len(layers)) for i, l in enumerate(layers)}
            rows = []
            for k in keys:
                V, kv = consensus([ads[(l, s)][k][0] for l in PRE[pair] for s in SEEDS], threshold=0.5)
                d = V.shape[0]
                row = {"layer": k[0], "module": k[1], "third": ["early", "middle", "late"][third[k[0]]],
                       "dims": kv, "chance_share": kv / d}
                for lam in ("0.0", "1.0"):
                    shares, norms = zip(*(energy_in(V, ads[(lam, s)][k][1], ads[(lam, s)][k][2]) for s in SEEDS))
                    row[f"share_{lam}"] = st.mean(shares)
                    row[f"energy_{lam}"] = st.mean(norms)
                rows.append(row)

            def group(field):
                g = defaultdict(list)
                for r in rows:
                    g[r[field]].append(r)
                res = {}
                for name, rs in g.items():
                    res[name] = {"dims": st.mean(r["dims"] for r in rs),
                                 "share_0.0": sum(r["share_0.0"] * r["energy_0.0"] for r in rs) / sum(r["energy_0.0"] for r in rs),
                                 "share_1.0": sum(r["share_1.0"] * r["energy_1.0"] for r in rs) / sum(r["energy_1.0"] for r in rs),
                                 "chance_share": st.mean(r["chance_share"] for r in rs),
                                 "energy_frac_0.0": sum(r["energy_0.0"] for r in rs) / sum(r["energy_0.0"] for r in rows)}
                return res

            tot0 = sum(r["energy_0.0"] for r in rows)
            tot1 = sum(r["energy_1.0"] for r in rows)
            cell = {"overall": {"dims": st.mean(r["dims"] for r in rows),
                                "share_0.0": sum(r["share_0.0"] * r["energy_0.0"] for r in rows) / tot0,
                                "share_1.0": sum(r["share_1.0"] * r["energy_1.0"] for r in rows) / tot1,
                                "chance_share": st.mean(r["chance_share"] for r in rows)},
                    "by_module": group("module"), "by_depth": group("third"), "modules": rows}
            out[f"{pair}/{short}"] = cell
            o = cell["overall"]
            print(f"{pair}/{short:6s} dims {o['dims']:.2f} | energy in region: lambda0 {o['share_0.0']:.3f} "
                  f"lambda1 {o['share_1.0']:.3f} chance {o['chance_share']:.4f} | depth "
                  + " ".join(f"{n}:{v['dims']:.1f}/{v['share_0.0']:.2f}" for n, v in sorted(cell['by_depth'].items()))
                  + " | module " + " ".join(f"{n}:{v['dims']:.1f}/{v['share_0.0']:.2f}" for n, v in sorted(cell['by_module'].items())),
                  flush=True)
    (X / "region_anatomy.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
