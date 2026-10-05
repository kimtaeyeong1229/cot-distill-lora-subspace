#!/usr/bin/env python3
"""What the energy-matched edits actually achieve, read from the saved edited adapters (read only).

The removal control "Other removed" (ablate_subspace.py, rm_decisive) takes, module by module, the
highest-energy directions of the adapter's own span orthogonal to the shared region V until it has removed
as much ||B||^2 as region removal (rm_pre) does. Where the region holds more of ||B||^2 than everything
outside it, the target cannot be reached and the control removes every direction outside V. The
fixed-norm amplification (steer_subspace.py, ampn / ampV) multiplies the region component of B by
(1 + gamma) and rescales each module back to its original ||dW||, so the region's own amplitude rises by
less than (1 + gamma).

For every cell and all three edited seeds (runs_edit_s{seed}, built with the edit-time leave-one-out
region, i.e. all pre-switch adapters except the edited lambda0 seed, eigenvalue cut 0.5):
  region_B / other_B   ||B||^2 removed by rm_pre and rm_decisive, as a share of the adapter's ||B||^2,
                       computed as ||B||^2 - ||B_edited||^2 (exact for an orthogonal projection removal)
  ratio_B              other_B / region_B, the achieved energy match in ||B||^2 (1 = matched)
  ratio_dW             the same ratio measured in ||dW||^2 = ||B A||^2
  unreachable          share of modules where rm_decisive removes less than 99.9% of the region's ||B||^2
  dims_region / dims_other   mean directions removed per module (ablation.json, dims_removed_mean)
  gain_ampn1 / gain_ampn4 / gain_ampV   effective gain of the region's amplitude in dW,
                       sqrt(sum ||V^T B' A||^2 / sum ||V^T B A||^2) over modules, with V rebuilt exactly as
                       steer_subspace.py --loo builds it

Run inside the container; writes experiments/energy_match.json.
"""

import json
import statistics as st

import torch

from ablate_subspace import consensus, left_basis, read_adapter
from config import CELL_DIR, CELL_NAMES, EXP as E, LAMBDAS, RESULTS, SEEDS, SWITCH

CELLS = {cell: (CELL_DIR[cell], SWITCH[cell[0]]) for cell in CELL_NAMES}   # cell -> (directory, switch lambda)


def load(path):
    config = json.loads((path / "adapter_config.json").read_text())
    raw, parts = read_adapter(path / "adapter_model.safetensors")
    out = {key: (raw[n["B"]].float(), raw[n["A"]].float()) for key, n in parts.items() if len(n) == 2}
    return out, config["lora_alpha"] / config["r"]


def main():
    result = {}
    for cell, (tag, cliff) in CELLS.items():
        pre_all = [(l, s) for l in LAMBDAS[:LAMBDAS.index(cliff)] for s in SEEDS]
        bases = {m: load(E / tag / "runs" / f"lambda{m[0]}_seed{m[1]}" / "adapter") for m in pre_all}
        bases = {m: {k: left_basis(B, A, sc) for k, (B, A) in ad.items()} for m, (ad, sc) in bases.items()}
        per_seed = {}
        for seed in SEEDS:
            pre = [m for m in pre_all if m != ("0.0", seed)]
            edits = E / tag / f"runs_edit_s{seed}"
            base, _ = load(E / tag / "runs" / f"lambda0.0_seed{seed}" / "adapter")
            rm, _ = load(edits / "lambda0.0_rm_pre" / "adapter")
            dec, _ = load(edits / "lambda0.0_rm_decisive" / "adapter")
            amp = {name: load(edits / name / "adapter")[0] for name in ("ampn1", "ampn4", "ampV")}
            tot = reg = oth = reg_dw = oth_dw = 0.0
            short = n = 0
            num = {name: 0.0 for name in amp}
            den = 0.0
            for key, (B, A) in base.items():
                if key not in rm or key not in bases[pre[0]]:
                    continue
                V, _ = consensus([bases[m][key] for m in pre], threshold=0.5)
                e_b = float(B.pow(2).sum())
                r_b = e_b - float(rm[key][0].pow(2).sum())
                o_b = e_b - float(dec[key][0].pow(2).sum())
                e_dw = float((B @ A).pow(2).sum())
                r_dw = e_dw - float((rm[key][0] @ A).pow(2).sum())
                o_dw = e_dw - float((dec[key][0] @ A).pow(2).sum())
                tot += e_b
                reg += r_b
                oth += o_b
                reg_dw += r_dw
                oth_dw += o_dw
                short += o_b < 0.999 * r_b
                n += 1
                region_before = float((V.T @ B @ A).pow(2).sum())
                den += region_before
                for name, ad in amp.items():
                    num[name] += float((V.T @ ad[key][0] @ ad[key][1]).pow(2).sum())
            meta = {v: json.loads((edits / f"lambda0.0_{v}" / "ablation.json").read_text())["dims_removed_mean"]
                    for v in ("rm_pre", "rm_decisive")}
            per_seed[seed] = {"region_B": reg / tot, "other_B": oth / tot, "ratio_B": oth / reg,
                              "ratio_dW": oth_dw / reg_dw, "unreachable": short / n, "modules": n,
                              "dims_region": meta["rm_pre"], "dims_other": meta["rm_decisive"],
                              **{f"gain_{name}": (num[name] / den) ** 0.5 for name in amp}}
        keys = per_seed[0].keys()
        result[cell] = {"seeds": per_seed,
                        "mean": {k: st.mean(per_seed[s][k] for s in SEEDS) for k in keys},
                        "min": {k: min(per_seed[s][k] for s in SEEDS) for k in keys},
                        "max": {k: max(per_seed[s][k] for s in SEEDS) for k in keys}}
        m = result[cell]["mean"]
        print(f"{cell:8s} region {m['region_B']:.3f} other {m['other_B']:.3f} ratio_B {result[cell]['min']['ratio_B']:.2f}-"
              f"{result[cell]['max']['ratio_B']:.2f} ratio_dW {m['ratio_dW']:.2f} unreachable {m['unreachable']:.2f} "
              f"dims {m['dims_region']:.1f}/{m['dims_other']:.1f} gain ampn1 {m['gain_ampn1']:.2f} "
              f"ampn4 {m['gain_ampn4']:.2f} ampV {m['gain_ampV']:.2f}", flush=True)
    every = [result[c]["seeds"][s] for c in result for s in SEEDS]
    result["all"] = {k: {"min": min(x[k] for x in every), "max": max(x[k] for x in every)} for k in every[0]}
    result["all_cell_means"] = {k: {"min": min(result[c]["mean"][k] for c in CELLS),
                                    "max": max(result[c]["mean"][k] for c in CELLS)} for k in every[0]}
    print(json.dumps(result["all"], indent=1))
    (RESULTS / "energy_match.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
