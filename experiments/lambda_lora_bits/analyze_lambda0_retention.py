#!/usr/bin/env python3
"""Share of the lambda = 0 region kept at each lambda, against a same-teacher reference (Figure 2, bottom).

The region is built from the three full-data lambda = 0 seeds (eigenvalue cut 0.5, as in Table 5). Its
coverage by the six lambda = 0 half-data adapters (same teacher, different data, never part of the build
set) is the reference; the coverage by the three full-data adapters at each lambda > 0, divided by that
reference, is the share of the region kept. At lambda = 0.2 this equals the first row of the half-data
reference in Table 18. A random adapter of the same rank gives the chance level, divided by the same
reference. Modules are weighted by the mean update norm of the cell's adapters, as elsewhere.

Run from the repo root; writes experiments/lambda0_retention.json.
"""

import json
import statistics as st
import sys
from pathlib import Path

import torch

X = Path("experiments")
E = X / "lambda_lora_bits"
sys.path.insert(0, str(E))
from analyze_length_confound import LAM, RUNS, SEEDS  # noqa: E402
from ablate_subspace import consensus, left_basis, read_adapter  # noqa: E402

HALVES = ["halfA", "halfA_seed1", "halfA_seed2", "halfB", "halfB_seed1", "halfB_seed2"]


def load(tag, run):
    path = E / tag / "runs" / run / "adapter"
    config = json.loads((path / "adapter_config.json").read_text())
    scale = config["lora_alpha"] / config["r"]
    raw, parts = read_adapter(path / "adapter_model.safetensors")
    bases, norms = {}, {}
    for key, names in parts.items():
        if len(names) == 2:
            B, A = raw[names["B"]].float(), raw[names["A"]].float()
            bases[key] = left_basis(B, A, scale)
            norms[key] = float(torch.linalg.matrix_norm(scale * B @ A))
    return bases, norms


def main():
    out = {}
    for pair, cells in RUNS.items():
        for short, tag in cells.items():
            full = {(l, s): load(tag, f"lambda{l}_seed{s}") for l in LAM for s in SEEDS}
            half = [load(tag, f"lambda0.0_{h}")[0] for h in HALVES]
            keys = set.intersection(*[set(b) for b, _ in full.values()])
            weight = {k: st.mean(n[k] for _, n in full.values()) for k in keys}
            den = sum(weight.values())
            ref, chance = 0.0, 0.0
            cov = {l: {s: 0.0 for s in SEEDS} for l in LAM[1:]}
            for k in keys:
                V, kv = consensus([full[("0.0", s)][0][k] for s in SEEDS], threshold=0.5)
                ref += weight[k] * st.mean(float((h[k].T @ V).pow(2).sum()) / kv for h in half) / den
                U = full[("0.0", 0)][0][k]
                chance += weight[k] * U.shape[1] / U.shape[0] / den
                for l in LAM[1:]:
                    for s in SEEDS:
                        cov[l][s] += weight[k] * float((full[(l, s)][0][k].T @ V).pow(2).sum()) / kv / den
            kept = {"0.0": 1.0}
            kept.update({l: st.mean(v.values()) / ref for l, v in cov.items()})
            out[f"{pair}/{short}"] = {"kept": kept, "kept_by_seed": {l: {s: x / ref for s, x in v.items()}
                                                                     for l, v in cov.items()},
                                      "reference": ref, "chance_kept": chance / ref}
            print(f"{pair}/{short:6s} " + " ".join(f"{kept[l]:.2f}" for l in LAM) + f" | chance {chance / ref:.3f}",
                  flush=True)
    (X / "lambda0_retention.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
