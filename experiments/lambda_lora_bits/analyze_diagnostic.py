#!/usr/bin/env python3
"""Can the weights alone tell whether a distilled student writes in the reasoning format?

For every adapter of a cell -- the unedited ones (6 lambdas x 3 seeds) and the section-5 edits (13
conditions x 3 seeds) -- this computes, from the adapter file only (no generation):

  region_dir     share of the consensus region V the adapter's directions cover, ||U^T V||_F^2 / k
  region_energy  share of the update's energy that lies in V, ||V^T U S||_F^2 / ||S||_F^2
  sim_lambda0    baseline: overlap phi with the same seed's lambda0 adapter ("looks like a reasoning adapter")
  norm           baseline: size of the update, ||dW||_F

all averaged over layers and modules with the same ||dW|| weights as analyze_consensus.py. V is built as
there (projectors of the pre-transition adapters, eigenvalue > 0.5) and never contains the adapter being
scored: pre-transition adapters are scored leave-one-out, and a seed's edits against the region built
without that seed's lambda0 adapter, which is the region the edits were made with.

Each predictor is then rank-correlated with what the adapter actually writes: the reflective-marker
density (first sample, first 200 MATH-500 problems, seed-specific; experiments/seed_spread.json) and, for
the Llama cells, the SAE index (experiments/sae_llama.json). The beta = 2 grafts loop, so their density
is taken over the first 300 words and they are also reported separately.

Held-out test (written down before the half-data evaluations were read, 2026-10-01): the data-half
adapters (lambda x halfA/halfB, seed 0, each trained on half of that lambda's traces) never enter V, which
is built from all pre-transition seed adapters. Prediction: within each cell, region_energy rank-orders
their density (rho > 0), and does so at least as well as sim_lambda0 and norm. Their density is counted
exactly as for the seed adapters (first sample of the first 200 MATH-500 problems, eval_halves/).

Writes experiments/diagnostic.json.
"""

import gzip
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

from config import EXP as E, REFLECTIVE, RESULTS
LAM = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
SEEDS = [0, 1, 2]
CELLS = {"A/Qwen": ("sweep", "0.4"), "A/Llama": ("sweep_llama", "0.4"), "A/OLMo": ("sweep_olmo2", "0.4"),
         "B/Qwen": ("pairB/qwen0.5b", "0.6"), "B/Llama": ("pairB/llama1b", "0.6"), "B/OLMo": ("pairB/olmo1b", "0.6"),
         "C/Qwen": ("pairD/qwen0.5b", "0.6"), "C/Llama": ("pairD/llama1b", "0.6"), "C/OLMo": ("pairD/olmo1b", "0.6")}
EDITS = {"rm_pre": "lambda0.0_rm_pre", "rm_other": "lambda0.0_rm_decisive", "rm_scale": "lambda0.0_rm_scale",
         "rm_pre_l1": "lambda1.0_rm_pre", "graft1": "graft1", "graft_other1": "graft_orth1",
         "graft_whole1": "graft_full1", "graft2": "graft2", "graft_other2": "graft_orth2",
         "graft_whole2": "graft_full2", "ampn1": "ampn1", "ampn4": "ampn4", "ampV": "ampV"}
LOOPING = {"graft2", "graft_other2", "graft_whole2"}
SAE_CONDS = EDITS   # sae_llama.py scored every section-5 condition
PREDICTORS = ("region_dir", "region_energy", "sim_lambda0", "norm")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def basis(p):
    """Per (layer, module): left singular vectors U, singular values S, output dim d."""
    c = json.loads((p.parent / "adapter_config.json").read_text()); s = c["lora_alpha"] / c["r"]
    parts = defaultdict(dict)
    with safe_open(p, "pt") as h:
        for k in h.keys():
            m = re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+_proj)", k)
            if m:
                parts[(int(m.group(1)), m.group(2))]["A" if "lora_A" in k else "B"] = h.get_tensor(k).float().to(DEV)
    out = {}
    for key, v in parts.items():
        if len(v) != 2:
            continue
        B, A = v["B"] * s, v["A"]
        Qb, Rb = torch.linalg.qr(B); Qa, Ra = torch.linalg.qr(A.T)
        U, S, _ = torch.linalg.svd(Rb @ Ra.T)
        out[key] = (Qb @ U, S)
    return out


def consensus(members, key):
    M = torch.cat([m[key][0] for m in members], dim=1)
    Q, S, _ = torch.linalg.svd(M, full_matrices=False)
    k = max(1, int(((S ** 2) / len(members) > 0.5).sum()))
    return Q[:, :k]


def region(members):
    keys = set.intersection(*[set(m) for m in members])
    return {key: consensus(members, key) for key in keys}


def score(ad, V, ref, weights):
    """Weighted averages over (layer, module) of the four predictors."""
    tot = {"region_dir": 0.0, "region_energy": 0.0, "sim_lambda0": 0.0, "norm": 0.0}; w = 0.0
    for key, Vk in V.items():
        if key not in ad:
            continue
        U, S = ad[key]; wt = weights[key]; k = Vk.shape[1]
        tot["region_dir"] += wt * float((U.T @ Vk).pow(2).sum() / k)
        tot["region_energy"] += wt * float(((Vk.T @ U) * S).pow(2).sum() / S.pow(2).sum())
        Ur = ref[key][0]
        tot["sim_lambda0"] += wt * float((U[:, :Ur.shape[1]].T @ Ur).pow(2).sum() / Ur.shape[1])
        tot["norm"] += wt * float(S.pow(2).sum().sqrt())
        w += wt
    return {k: v / w for k, v in tot.items()}


def half_density(path, samples=2):
    """Markers per 1k words over the first sample of each problem, as analyze_main_edits.density."""
    with gzip.open(path, "rt") as f:
        first = [json.loads(l)["text"] for i, l in enumerate(f) if i % samples == 0]
    return 1000 * sum(len(REFLECTIVE.findall(t)) for t in first) / max(sum(len(t.split()) for t in first), 1)


def spearman(xs, ys):
    rank = lambda v: [sorted(v).index(x) + (v.count(x) - 1) / 2 for x in v]
    return st.correlation(rank(xs), rank(ys)) if len(set(xs)) > 1 and len(set(ys)) > 1 else float("nan")


def main():
    spread = json.loads((RESULTS / "seed_spread.json").read_text())
    sae = json.loads((RESULTS / "sae_llama.json").read_text()) if (RESULTS / "sae_llama.json").exists() else {}
    out = {}
    for cell, (tag, cliff) in CELLS.items():
        runs = E / tag / "runs"
        base = {(l, s): basis(runs / f"lambda{l}_seed{s}" / "adapter" / "adapter_model.safetensors") for l in LAM for s in SEEDS}
        i = LAM.index(cliff)
        pre = [(l, s) for l in LAM[:i] for s in SEEDS]
        keys = set.intersection(*[set(b) for b in base.values()])
        weights = {key: st.mean(float(b[key][1].pow(2).sum().sqrt()) for b in base.values()) for key in keys}
        V_full = region([base[t] for t in pre])
        rows = []
        for (l, s), ad in base.items():
            V = region([base[t] for t in pre if t != (l, s)]) if (l, s) in pre else V_full
            sc = score(ad, V, base[("0.0", s)] if (l, s) != ("0.0", s) else base[("0.2", s)], weights)
            dens = spread[cell]["table1"][l]["seeds"][s]
            rows.append({"adapter": f"lambda{l}", "seed": s, "edited": False, "density": dens, **sc})
        for s in SEEDS:
            V = region([base[t] for t in pre if t != ("0.0", s)])
            for name, d in EDITS.items():
                ad = basis(E / tag / f"runs_edit_s{s}" / d / "adapter" / "adapter_model.safetensors")
                sc = score(ad, V, base[("0.0", s)], weights)
                q = "head_markers_per_1k" if name in LOOPING else "markers_per_1k"
                dens = spread[cell]["edits"][name][q]["seeds"][s]
                rows.append({"adapter": name, "seed": s, "edited": True, "density": dens, **sc})
        if cell.endswith("Llama") and sae:
            idx = {f"lambda{l}": f"base{l}" for l in LAM} | SAE_CONDS
            for r in rows:
                c = idx.get(r["adapter"])
                if c:
                    r["sae"] = sae[cell[0]][c]["sae"]["seeds"][r["seed"]]
        halves = []
        for l in LAM:
            for h in ("halfA", "halfB"):
                ev = E / tag / "eval_halves" / f"lambda{l}_{h}.samples.jsonl.gz"
                if ev.exists():
                    ad = basis(runs / f"lambda{l}_{h}" / "adapter" / "adapter_model.safetensors")
                    halves.append({"adapter": f"lambda{l}_{h}", "density": half_density(ev),
                                   **score(ad, V_full, base[("0.0", 0)], weights)})
        res = {"rows": rows, "halves": halves}
        if len(halves) == 2 * len(LAM):
            res["held_out"] = {p: spearman([r[p] for r in halves], [r["density"] for r in halves]) for p in PREDICTORS}
        for subset, keep in (("all", lambda r: True), ("no_loop", lambda r: r["adapter"] not in LOOPING),
                             ("unedited", lambda r: not r["edited"])):
            sel = [r for r in rows if keep(r)]
            res[subset] = {p: spearman([r[p] for r in sel], [r["density"] for r in sel])
                           for p in PREDICTORS}
        if any("sae" in r for r in rows):
            sel = [r for r in rows if "sae" in r]
            res["sae"] = {p: spearman([r[p] for r in sel], [r["sae"] for r in sel]) for p in PREDICTORS}
        out[cell] = res
        f = lambda d: "  ".join(f"{k} {v:+.2f}" for k, v in d.items())
        print(f"{cell:8s} all      {f(res['all'])}\n{'':8s} no_loop  {f(res['no_loop'])}\n{'':8s} unedited {f(res['unedited'])}"
              + (f"\n{'':8s} sae      {f(res['sae'])}" if "sae" in res else "")
              + (f"\n{'':8s} held-out {f(res['held_out'])}" if "held_out" in res else f"\n{'':8s} held-out {len(halves)}/12 halves"), flush=True)
    (RESULTS / "diagnostic.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
