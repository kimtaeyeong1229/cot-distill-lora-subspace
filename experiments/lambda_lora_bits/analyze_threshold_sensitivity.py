#!/usr/bin/env python3
"""Sensitivity of the region to its eigenvalue cuts, and how consistent the post-switch adapters are.

1. Shared region (Eq. 4, cut 0.5). Re-run analyze_consensus.py's construction at cuts 0.4, 0.5 and 0.6:
   region size, coverage by held-out pre-switch adapters (leave-one-out), coverage by post-switch
   adapters and the random-adapter chance level r/d. One SVD per (module, left-out adapter) serves all
   cuts, since only the number of kept eigenvectors changes.
2. Two-seed region (cut 0.75). Re-run analyze_seed_region_overlap.py at cuts 0.65, 0.75 and 0.85: the
   share of the shared region's dimensions (cut 0.5, built without the seed-0 lambda0 adapter) that the
   region from the seed-1 and seed-2 lambda0 adapters spans.
3. Post-switch consistency. Mean pairwise overlap phi (Eq. 2) between adapters of DIFFERENT seeds,
   within the pre-switch group, within the post-switch group, and across the two groups, so a reader can
   check whether short-answer adapters agree with each other or are merely scattered.

All overlaps are averaged over modules with weight ||dW||_F averaged over the adapters involved, as in
analyze_consensus.py. No training or generation (CPU is fine); writes experiments/threshold_sensitivity.json.
"""

import itertools
import json
import statistics as st
from pathlib import Path

import torch

from ablate_subspace import LAMBDAS, SEEDS, left_basis, read_adapter

from config import CELL_DIR, CELL_NAMES, EXP as E, RESULTS, SWITCH

CELLS = {cell: (CELL_DIR[cell], SWITCH[cell[0]]) for cell in CELL_NAMES}   # cell -> (directory, switch lambda)
CUTS = [0.4, 0.5, 0.6]
SEED_CUTS = [0.65, 0.75, 0.85]


def load(tag, lam, seed):
    path = E / tag / "runs" / f"lambda{lam}_seed{seed}" / "adapter"
    config = json.loads((path / "adapter_config.json").read_text())
    scale = config["lora_alpha"] / config["r"]
    raw, parts = read_adapter(path / "adapter_model.safetensors")
    out = {}
    for key, n in parts.items():
        if len(n) == 2:
            B, A = raw[n["B"]].float(), raw[n["A"]].float()
            out[key] = (left_basis(B, A, scale), float(torch.linalg.matrix_norm(scale * B @ A)))
    return out


def eig(members):
    """Eigenvectors / eigenvalues of mean(U U^T) through the stacked thin SVD."""
    Q, S, _ = torch.linalg.svd(torch.cat(members, dim=1), full_matrices=False)
    return Q, (S ** 2) / len(members)


def region(Q, ev, cut):
    k = max(1, int((ev > cut).sum()))
    return Q[:, :k], k


def main():
    out = {}
    for cell, (tag, cliff) in CELLS.items():
        i = LAMBDAS.index(cliff)
        pre = [(l, s) for l in LAMBDAS[:i] for s in SEEDS]
        post = [(l, s) for l in LAMBDAS[i:] for s in SEEDS]
        ads = {m: load(tag, *m) for m in pre + post}
        keys = sorted(set.intersection(*[set(a) for a in ads.values()]))
        acc = {c: {"dims": 0.0, "pre_loo": 0.0, "post": 0.0} for c in CUTS}
        seed_acc = {c: {"overlap": 0.0, "dims_seed": 0.0} for c in SEED_CUTS}
        chance = wsum = seed_w = 0.0
        groups = {"within_pre": 0.0, "within_post": 0.0, "across": 0.0}
        for key in keys:
            U = {m: ads[m][key][0] for m in ads}
            w = st.mean(ads[m][key][1] for m in ads)
            r, d = U[pre[0]].shape[1], U[pre[0]].shape[0]
            # 1. shared region at several cuts
            Qf, evf = eig([U[m] for m in pre])
            loo = {m: eig([U[u] for u in pre if u != m]) for m in pre}
            for c in CUTS:
                V, k = region(Qf, evf, c)
                acc[c]["dims"] += w * k
                acc[c]["post"] += w * st.mean(float((U[m].T @ V).pow(2).sum()) / k for m in post)
                vals = []
                for m in pre:
                    V2, k2 = region(*loo[m], c)
                    vals.append(float((U[m].T @ V2).pow(2).sum()) / k2)
                acc[c]["pre_loo"] += w * st.mean(vals)
            chance += w * r / d
            wsum += w
            # 2. two-seed region against the shared region without the seed-0 lambda0 adapter
            V0, k0 = region(*loo[("0.0", 0)], 0.5)
            Qs, evs = eig([U[("0.0", 1)], U[("0.0", 2)]])
            w0 = ads[("0.0", 0)][key][1]
            for c in SEED_CUTS:
                Sk, ks = region(Qs, evs, c)
                seed_acc[c]["overlap"] += w0 * float((Sk.T @ V0).pow(2).sum()) / k0
                seed_acc[c]["dims_seed"] += w0 * ks
            seed_w += w0
            # 3. cross-seed pairwise overlap within and across the groups
            def mean_phi(pairs):
                return st.mean(float((U[a].T @ U[b]).pow(2).sum()) / r for a, b in pairs if a[1] != b[1])
            groups["within_pre"] += w * mean_phi(itertools.combinations(pre, 2))
            groups["within_post"] += w * mean_phi(itertools.combinations(post, 2))
            groups["across"] += w * mean_phi(itertools.product(pre, post))
        row = {"transition": cliff, "chance": chance / wsum,
               "shared_region": {str(c): {f: v / wsum for f, v in acc[c].items()} for c in CUTS},
               "two_seed_region": {str(c): {f: v / seed_w for f, v in seed_acc[c].items()} for c in SEED_CUTS},
               "cross_seed_phi": {g: v / wsum for g, v in groups.items()}}
        out[cell] = row
        s = row["shared_region"]
        print(f"{cell:8s} " + " | ".join(f"cut {c}: {s[str(c)]['dims']:.1f}d pre {s[str(c)]['pre_loo']:.3f} "
                                         f"post {s[str(c)]['post']:.3f}" for c in CUTS)
              + " || seed " + " ".join(f"{c}:{row['two_seed_region'][str(c)]['overlap']:.2f}" for c in SEED_CUTS)
              + " || phi " + " ".join(f"{g} {v:.3f}" for g, v in row["cross_seed_phi"].items()), flush=True)
    (RESULTS / "threshold_sensitivity.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
