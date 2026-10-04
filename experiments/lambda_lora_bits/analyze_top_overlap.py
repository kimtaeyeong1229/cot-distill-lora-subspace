#!/usr/bin/env python3
"""How much does the shared region V overlap the energy-matched top singular subspace it is compared with?

Table 6's "Top removed (energy-matched)" control (ablate_subspace.py, variant gs_energy) removes the top
singular components of the lambda0 adapter's own update until the removed energy matches the energy that
removing V takes. In pairs B and C that control also turns markers almost off, so this script measures,
per cell and seed, how much the two removed subspaces coincide. Everything is recomputed from the
existing adapters exactly as ablate_subspace.py builds the edits (--loo, source lambda0 of that seed):

  V      the shared region built without the edited adapter (eigenvalue cut 0.5)
  U_top  the top-r left singular vectors of the edited lambda0 update, r chosen as in gs_energy

and reports, averaged over modules with weight ||dW||_F of the edited adapter,

  cover_region = ||V^T U_top||_F^2 / k   share of V's k dimensions that U_top spans
  cover_top    = ||V^T U_top||_F^2 / r   share of U_top's r dimensions that lie inside V
  chance       = r / d                   what a random r-dimensional subspace would give for cover_region

Region and top-subspace sizes (dims_region, dims_top) are averaged with the same ||dW||_F weights, as the
region sizes of Table 5 are.

No training or generation (CPU is fine); writes experiments/top_overlap.json.
"""

import json
import statistics as st
from pathlib import Path

import torch

from ablate_subspace import LAMBDAS, SEEDS, consensus, left_basis, read_adapter, spectrum

from config import CELL_DIR, CELL_NAMES, EXP as E, RESULTS, SWITCH

CELLS = {cell: (CELL_DIR[cell], SWITCH[cell[0]]) for cell in CELL_NAMES}   # cell -> (directory, switch lambda)


def load(tag, lam, seed):
    path = E / tag / "runs" / f"lambda{lam}_seed{seed}" / "adapter"
    config = json.loads((path / "adapter_config.json").read_text())
    scale = config["lora_alpha"] / config["r"]
    raw, parts = read_adapter(path / "adapter_model.safetensors")
    return {key: (raw[n["B"]].float(), raw[n["A"]].float(), scale) for key, n in parts.items() if len(n) == 2}


def main():
    out = {}
    for cell, (tag, cliff) in CELLS.items():
        pre_all = [(l, s) for l in LAMBDAS[:LAMBDAS.index(cliff)] for s in SEEDS]
        bases = {m: {k: left_basis(*v) for k, v in load(tag, *m).items()} for m in pre_all}
        per_seed = {}
        for seed in SEEDS:
            pre = [m for m in pre_all if m != ("0.0", seed)]
            src = load(tag, "0.0", seed)
            num_v = num_t = chance = dims_k = dims_r = wsum = 0.0
            for key, (B, A, scale) in src.items():
                V, k = consensus([bases[m][key] for m in pre], threshold=0.5)
                U, S = spectrum(B, A, scale)
                e2 = S.pow(2)
                target = float(((V.T @ U) * S).pow(2).sum())          # dW energy that removing V takes
                r = max(1, min(int((e2.cumsum(0) < target).sum()) + 1, U.shape[1]))
                inter = float((V.T @ U[:, :r]).pow(2).sum())
                w = float(e2.sum().sqrt())                               # ||dW||_F of the edited module
                num_v += w * inter / k
                num_t += w * inter / r
                chance += w * r / U.shape[0]
                dims_k += w * k
                dims_r += w * r
                wsum += w
            per_seed[seed] = {"cover_region": num_v / wsum, "cover_top": num_t / wsum, "chance": chance / wsum,
                              "dims_region": dims_k / wsum, "dims_top": dims_r / wsum}
        row = {f: st.mean(per_seed[s][f] for s in SEEDS) for f in per_seed[0]}
        row["cover_region_sd"] = st.stdev(per_seed[s]["cover_region"] for s in SEEDS)
        row["cover_top_sd"] = st.stdev(per_seed[s]["cover_top"] for s in SEEDS)
        row["per_seed"] = per_seed
        out[cell] = row
        print(f"{cell:8s} V covered by top {row['cover_region']:.3f} +- {row['cover_region_sd']:.3f} | "
              f"top inside V {row['cover_top']:.3f} | dims V {row['dims_region']:.1f} top {row['dims_top']:.1f} "
              f"| chance {row['chance']:.4f}", flush=True)
    (RESULTS / "top_overlap.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
