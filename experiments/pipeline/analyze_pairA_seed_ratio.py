#!/usr/bin/env python3
import json
import math
import statistics as st
import torch
from analyze_length_confound import RUNS, SEEDS
from ablate_subspace import consensus, left_basis, read_adapter
from config import EXP as E, RESULTS as X

HALVES = ["halfA", "halfA_seed1", "halfA_seed2", "halfB", "halfB_seed1", "halfB_seed2"]
INTERVALS = {"held": ("0.0", "0.2"), "switch": ("0.2", "0.4")}


def load(tag, run):
    path = E / tag / "runs" / run / "adapter"
    config = json.loads((path / "adapter_config.json").read_text())
    scale = config["lora_alpha"] / config["r"]
    raw, parts = read_adapter(path / "adapter_model.safetensors")
    bases, norms = ({}, {})
    for key, names in parts.items():
        if len(names) == 2:
            B, A = (raw[names["B"]].float(), raw[names["A"]].float())
            bases[key] = left_basis(B, A, scale)
            norms[key] = float(torch.linalg.matrix_norm(scale * B @ A))
    return (bases, norms)


def main():
    lens = json.loads((X / "length_confound.json").read_text())["pairs"]["A"]["intervals"]
    loglen = {r["interval"]: r["log_len_drop"] for r in lens}
    out = {}
    for short, tag in RUNS["A"].items():
        full = {(l, s): load(tag, f"lambda{l}_seed{s}") for l in ("0.0", "0.2", "0.4") for s in SEEDS}
        keys = set.intersection(*[set(b) for b, _ in full.values()])
        weight = {k: st.mean((n[k] for _, n in full.values())) for k in keys}
        den = sum(weight.values())
        cell = {}
        for name, (a, b) in INTERVALS.items():
            half = [load(tag, f"lambda{a}_{h}")[0] for h in HALVES]
            ref, nxt = (0.0, {s: 0.0 for s in SEEDS})
            for k in keys:
                V, kv = consensus([full[a, s][0][k] for s in SEEDS], threshold=0.5)
                ref += weight[k] * st.mean((float((h[k].T @ V).pow(2).sum()) / kv for h in half)) / den
                for s in SEEDS:
                    nxt[s] += weight[k] * float((full[b, s][0][k].T @ V).pow(2).sum()) / kv / den
            L = loglen[f"{a}->{b}"]
            cell[name] = {
                "kept": {s: nxt[s] / ref for s in SEEDS},
                "loss_per_loglen": {s: -math.log(nxt[s] / ref) / L for s in SEEDS},
            }
        ratios = [
            cell["switch"]["loss_per_loglen"][i] / cell["held"]["loss_per_loglen"][j]
            for i in SEEDS
            for j in SEEDS
        ]
        cell["switch_over_held"] = {"min": min(ratios), "max": max(ratios), "all": ratios}
        out[f"A/{short}"] = cell
        print(
            f"A/{short}: held kept "
            + " ".join((f"{v:.2f}" for v in cell["held"]["kept"].values()))
            + " | switch kept "
            + " ".join((f"{v:.2f}" for v in cell["switch"]["kept"].values()))
            + f" | ratio {min(ratios):.2f}-{max(ratios):.2f}",
            flush=True,
        )
    (X / "pairA_seed_ratio.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
