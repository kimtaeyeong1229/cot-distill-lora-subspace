#!/usr/bin/env python3
"""Pair A's held-vs-switch contrast with no shared LoRA initialization between region and scored adapters.

Same-seed adapters share their LoRA initialization, which inflates subspace overlap. Here, for each seed s,
the region at an interval's start is built from the two full-data adapters of the other seeds (eigenvalue
cut 0.75, the paper's two-adapter cut; with two adapters a cut of 0.5 keeps nearly the whole update subspace), and both the reference (the two half-data adapters of seed s at the start) and the scored
adapter (the full-data adapter of seed s at the end) carry seed s, so neither shares an initialization with
the builders. Kept share = coverage(end, seed s) / mean coverage(half-data, seed s); loss per unit of
log-length drop = -ln(kept) / log_len_drop. Reported per seed and as the switch/held ratio.
The same loss is also divided by the drop in log teacher reflective markers over the interval
(experiments/teacher_density.json, A.teacher at lambda 0, 0.2 and 0.4), giving a per-unit-of-marker-drop
switch/held ratio, and the region size (directions kept per module, unweighted and update-norm weighted) is
recorded for each held-out seed.

Run from the repo root; writes experiments/pairA_crossseed.json.
"""

import json
import math
import statistics as st
import sys
from pathlib import Path

import torch

X = Path("experiments")
E = X / "lambda_lora_bits"
sys.path.insert(0, str(E))
from analyze_length_confound import RUNS, SEEDS  # noqa: E402
from ablate_subspace import consensus, left_basis, read_adapter  # noqa: E402

HALF = {0: ["halfA", "halfB"], 1: ["halfA_seed1", "halfB_seed1"], 2: ["halfA_seed2", "halfB_seed2"]}
INTERVALS = {"held": ("0.0", "0.2"), "switch": ("0.2", "0.4")}


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
    lens = {r["interval"]: r["log_len_drop"]
            for r in json.loads((X / "length_confound.json").read_text())["pairs"]["A"]["intervals"]}
    teacher = json.loads((X / "teacher_density.json").read_text())["A"]["teacher"]
    mdrop = {f"{a}->{b}": math.log(teacher[a] / teacher[b]) for a, b in INTERVALS.values()}
    out = {"teacher_log_marker_drop": mdrop}
    for short, tag in RUNS["A"].items():
        full = {(l, s): load(tag, f"lambda{l}_seed{s}") for l in ("0.0", "0.2", "0.4") for s in SEEDS}
        keys = set.intersection(*[set(b) for b, _ in full.values()])
        weight = {k: st.mean(n[k] for _, n in full.values()) for k in keys}
        den = sum(weight.values())
        cell = {}
        for name, (a, b) in INTERVALS.items():
            halves = {s: [load(tag, f"lambda{a}_{h}")[0] for h in HALF[s]] for s in SEEDS}
            per = {}
            for s in SEEDS:
                builders = [t for t in SEEDS if t != s]
                ref = nxt = 0.0
                dims = {}
                for k in keys:
                    V, kv = consensus([full[(a, t)][0][k] for t in builders], threshold=0.75)
                    dims[k] = kv
                    ref += weight[k] * st.mean(float((h[k].T @ V).pow(2).sum()) / kv for h in halves[s]) / den
                    nxt += weight[k] * float((full[(b, s)][0][k].T @ V).pow(2).sum()) / kv / den
                kept = nxt / ref
                per[s] = {"kept": kept, "loss_per_log_len": -math.log(kept) / lens[f"{a}->{b}"],
                          "loss_per_log_marker": -math.log(kept) / mdrop[f"{a}->{b}"],
                          "region_dims_mean": st.mean(dims.values()),
                          "region_dims_weighted": sum(weight[k] * dims[k] for k in keys) / den}
            cell[name] = per
        ratios = [cell["switch"][s]["loss_per_log_len"] / cell["held"][s]["loss_per_log_len"] for s in SEEDS]
        cell["switch_over_held_by_seed"] = ratios
        cell["switch_over_held_per_marker_by_seed"] = [
            cell["switch"][s]["loss_per_log_marker"] / cell["held"][s]["loss_per_log_marker"] for s in SEEDS]
        out[f"A/{short}"] = cell
        print(f"A/{short}: held kept " + " ".join(f"{cell['held'][s]['kept']:.2f}" for s in SEEDS)
              + " | switch kept " + " ".join(f"{cell['switch'][s]['kept']:.2f}" for s in SEEDS)
              + " | ratio " + " ".join(f"{r:.2f}" for r in ratios)
              + " | per marker " + " ".join(f"{r:.2f}" for r in cell["switch_over_held_per_marker_by_seed"])
              + " | dims " + " ".join(f"{cell[n][s]['region_dims_mean']:.1f}" for n in INTERVALS for s in SEEDS),
              flush=True)
    (X / "pairA_crossseed.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
