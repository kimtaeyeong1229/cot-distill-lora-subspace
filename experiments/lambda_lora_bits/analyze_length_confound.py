#!/usr/bin/env python3
"""Does the shared region follow the teacher's thinking or the length of its training solutions?

Training solutions shorten as lambda rises, and they shorten most where the teacher stops thinking, so
over the whole sweep length and think rate are nearly collinear. Pooling cell x lambda points into one
correlation cannot separate them, and those points are pseudo-replicated: the 45 points carry only 15
teacher-level values of length and think rate. This script therefore looks at the lambda intervals one at
a time, where the two come apart: in some intervals the training solutions shorten while the teacher keeps
thinking, in others the teacher stops thinking.

Teacher level (one value per pair and lambda, never pooled over students):
  tokens    mean solution tokens per training problem (split_stats.json: train_tokens / n_train, counted
            with the Qwen2.5-0.5B tokenizer used to filter every pair's splits)
  think     teacher think rate in % (experiments/think_rate.json, pair C under key "D")
  markers   reflective markers per 1,000 words in the teacher's solutions (experiments/teacher_density.json)
Student level (per cell, mean of three seeds):
  coverage  shared-region coverage (experiments/coverage_by_lambda.json, pair C under "D")
  R         overlap ratio (experiments/phi_resample_s012.json, pair C under "D")

Interval classes, from the teacher alone:
  switch          think rate falls by >= 50 points, or, directly after a switch interval, falls from >= 15%
                  by >= 15 points (pair C's second step, 0.6 -> 0.8, 26.8% -> 0%)
  thinking held   think rate changes by < 15 points, stays >= 50% at both ends, and the teacher's markers at
                  the end of the interval stay at least twice their lambda = 1 level (not collapsed)
  after switch    think rate < 15% at both ends
For each interval the script reports the log drop in training length, ln(tokens_start / tokens_end), the
think-rate and marker changes, and per student the change (end - start) in coverage and in R. The summary
contrasts thinking-held and switch intervals by the range of the change over students and intervals and
by the change per unit log-length drop.

Pre-switch coverage is scored leave-one-out against a region built from the other pre-switch adapters,
which include the other seeds at the same lambda, whereas post-switch adapters never help build it. A
held interval therefore compares two adapters from the build set and a switch interval crosses out of it.
As a check free of that asymmetry, the script also scores every lambda against the two-seed lambda = 0
region of Section 6.2 (two lambda = 0 seeds, eigenvalue cut 0.75), with the third seed scored at every
lambda, so that no scored adapter helped build the region. Modules are weighted by the mean update norm
of the cell's 18 adapters, as in analyze_coverage_by_lambda.py, and the result is averaged over the three
choices of the held-out seed. A second, local check repeats this per interval with the region built from
two seeds at the interval's start (cut 0.75) and the third seed scored at both ends. Both ends are then
outside the build set, but the end adapter also differs in its training data, so even an interval in
which neither length nor thinking changes loses some coverage (compare the after-switch intervals).

Run from the repo root; writes experiments/length_confound.json.
"""

import json
import math
import statistics as st
import sys
from pathlib import Path

X = Path("experiments")
E = X / "lambda_lora_bits"
sys.path.insert(0, str(E))
LAM = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
SEEDS = [0, 1, 2]
PAIRS = {"A": ("sweep", "A"), "B": ("pairB", "B"), "C": ("pairD", "D")}   # paper name -> (split dir, json key)
STUDENTS = {"Qwen": "Qwen2.5-0.5B", "Llama": "Llama-3.2-1B", "OLMo": "OLMo-2-1B"}
RUNS = {"A": {"Qwen": "sweep", "Llama": "sweep_llama", "OLMo": "sweep_olmo2"},
        "B": {"Qwen": "pairB/qwen0.5b", "Llama": "pairB/llama1b", "OLMo": "pairB/olmo1b"},
        "C": {"Qwen": "pairD/qwen0.5b", "Llama": "pairD/llama1b", "OLMo": "pairD/olmo1b"}}
SWITCH_DROP, HELD_CHANGE, ON_LEVEL, OFF_LEVEL, MARKER_FLOOR = 50.0, 15.0, 50.0, 15.0, 2.0


def classify(think, markers):
    """Label the five intervals of one pair from the teacher's think rate (%) and marker density."""
    labels = []
    for i in range(5):
        t0, t1 = think[LAM[i]], think[LAM[i + 1]]
        if t0 - t1 >= SWITCH_DROP:
            labels.append("switch")
        elif labels and labels[-1] == "switch" and t0 >= OFF_LEVEL and t0 - t1 >= OFF_LEVEL:
            labels.append("switch")
        elif (abs(t1 - t0) < HELD_CHANGE and min(t0, t1) >= ON_LEVEL
              and markers[LAM[i + 1]] >= MARKER_FLOOR * markers["1.0"]):
            labels.append("held")
        elif max(t0, t1) < OFF_LEVEL:
            labels.append("after")
        else:
            labels.append("unclassified")
    return labels


def two_seed_coverage():
    """Coverage of the two-seed lambda = 0 region (cut 0.75) by the held-out third seed at every lambda."""
    import torch
    from ablate_subspace import consensus, left_basis, read_adapter

    def load(tag, lam, seed):
        path = E / tag / "runs" / f"lambda{lam}_seed{seed}" / "adapter"
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

    out = {}
    for pair, cells in RUNS.items():
        for short, tag in cells.items():
            ads = {(l, s): load(tag, l, s) for l in LAM for s in SEEDS}
            keys = set.intersection(*[set(b) for b, _ in ads.values()])
            weight = {k: st.mean(n[k] for _, n in ads.values()) for k in keys}
            cov = {l: [] for l in LAM}
            local = {f"{LAM[i]}->{LAM[i + 1]}": [] for i in range(5)}
            chance = 0.0
            for held in SEEDS:
                members = [s for s in SEEDS if s != held]
                num = {l: 0.0 for l in LAM}
                den = 0.0
                for k in keys:
                    V, kv = consensus([ads[("0.0", s)][0][k] for s in members], threshold=0.75)
                    for l in LAM:
                        num[l] += weight[k] * float((ads[(l, held)][0][k].T @ V).pow(2).sum()) / kv
                    den += weight[k]
                    if held == SEEDS[0]:
                        U = ads[("0.0", held)][0][k]
                        chance += weight[k] * U.shape[1] / U.shape[0]
                for l in LAM:
                    cov[l].append(num[l] / den)
                for i in range(5):
                    a, b = LAM[i], LAM[i + 1]
                    na = nb = 0.0
                    for k in keys:
                        V, kv = consensus([ads[(a, s)][0][k] for s in members], threshold=0.75)
                        na += weight[k] * float((ads[(a, held)][0][k].T @ V).pow(2).sum()) / kv
                        nb += weight[k] * float((ads[(b, held)][0][k].T @ V).pow(2).sum()) / kv
                    local[f"{a}->{b}"].append([na / den, nb / den])
            cell = f"{pair}/{short}"
            out[cell] = {"coverage": {l: st.mean(v) for l, v in cov.items()},
                         "coverage_by_held_seed": cov, "chance": chance / den,
                         "local_start_end": {iv: [st.mean(x for x, _ in v), st.mean(y for _, y in v)]
                                             for iv, v in local.items()},
                         "local_change": {iv: st.mean(y - x for x, y in v) for iv, v in local.items()}}
            print(f"two-seed region {cell:8s} " + " ".join(f"{out[cell]['coverage'][l]:.3f}" for l in LAM),
                  flush=True)
    return out


def summarize(values):
    return {"min": min(values), "max": max(values), "mean": st.mean(values), "n": len(values)}


def main():
    think_all = json.loads((X / "think_rate.json").read_text())
    teacher = json.loads((X / "teacher_density.json").read_text())
    res = json.loads((X / "phi_resample_s012.json").read_text())
    cov_all = json.loads((X / "coverage_by_lambda.json").read_text())
    seed_cov = two_seed_coverage()

    out = {"criteria": {"switch_drop_points": SWITCH_DROP, "held_change_points": HELD_CHANGE,
                        "held_min_think": ON_LEVEL, "after_max_think": OFF_LEVEL,
                        "held_marker_floor_x_lambda1": MARKER_FLOOR,
                        "switch_continuation": "directly after a switch, think rate falls from >= 15% by >= 15 points"},
           "pairs": {}, "two_seed_region": seed_cov, "summary": {}}
    pooled = {c: {"coverage": [], "coverage_per_log_len": [], "R": [], "R_per_log_len": [], "seed_cov": [],
                  "local": [], "log_len": []} for c in ("held", "switch", "after")}
    for p, (split, key) in PAIRS.items():
        stats = json.loads((E / split / "splits" / "split_stats.json").read_text())
        n = stats["n_train"]
        tokens = {l: stats["per_lambda"][l]["train_tokens"] / n for l in LAM}
        think = {l: 100 * think_all[key]["rates"][l] for l in LAM}
        markers = {l: teacher[p]["teacher"][l] for l in LAM}
        labels = classify(think, markers)
        cov = {s: cov_all[f"{key}/{full}"]["coverage"] for s, full in STUDENTS.items()}
        R = {s: {l: res[f"{key}/{full}"]["rows"][l]["retention"] for l in LAM} for s, full in STUDENTS.items()}
        sc = {s: seed_cov[f"{p}/{s}"]["coverage"] for s in STUDENTS}
        intervals = []
        for i in range(5):
            a, b = LAM[i], LAM[i + 1]
            d_len = math.log(tokens[a] / tokens[b])
            row = {"interval": f"{a}->{b}", "class": labels[i], "tokens": [tokens[a], tokens[b]],
                   "log_len_drop": d_len, "think": [think[a], think[b]], "think_change": think[b] - think[a],
                   "markers": [markers[a], markers[b]], "marker_change": markers[b] - markers[a],
                   "marker_log_ratio": math.log(markers[a] / markers[b]),
                   "coverage_change": {s: cov[s][b] - cov[s][a] for s in STUDENTS},
                   "R_change": {s: R[s][b] - R[s][a] for s in STUDENTS},
                   "two_seed_coverage_change": {s: sc[s][b] - sc[s][a] for s in STUDENTS},
                   "local_coverage_change": {s: seed_cov[f"{p}/{s}"]["local_change"][f"{a}->{b}"]
                                             for s in STUDENTS}}
            row["coverage_change_per_log_len"] = {s: v / d_len for s, v in row["coverage_change"].items()}
            row["R_change_per_log_len"] = {s: v / d_len for s, v in row["R_change"].items()}
            intervals.append(row)
            c = labels[i]
            if c in pooled:
                pooled[c]["coverage"] += row["coverage_change"].values()
                pooled[c]["coverage_per_log_len"] += row["coverage_change_per_log_len"].values()
                pooled[c]["R"] += row["R_change"].values()
                pooled[c]["R_per_log_len"] += row["R_change_per_log_len"].values()
                pooled[c]["seed_cov"] += row["two_seed_coverage_change"].values()
                pooled[c]["local"] += row["local_coverage_change"].values()
                pooled[c]["log_len"].append(d_len)
            print(f"{p} {a}->{b} {c:7s} loglen {d_len:.2f} think {think[a]:5.1f}->{think[b]:5.1f} "
                  f"markers {markers[a]:5.2f}->{markers[b]:5.2f} | cov "
                  + " ".join(f"{v:+.3f}" for v in row["coverage_change"].values()) + " | R "
                  + " ".join(f"{v:+.3f}" for v in row["R_change"].values()) + " | 2-seed cov "
                  + " ".join(f"{v:+.3f}" for v in row["two_seed_coverage_change"].values()) + " | local "
                  + " ".join(f"{v:+.3f}" for v in row["local_coverage_change"].values()))
        out["pairs"][p] = {"n_train": n, "tokens": tokens, "think_rate": think, "teacher_markers": markers,
                           "coverage": cov, "R": R, "two_seed_coverage": sc, "intervals": intervals}
    for c, v in pooled.items():
        out["summary"][c] = {"intervals": sum(1 for p in out["pairs"].values() for r in p["intervals"]
                                              if r["class"] == c),
                             "log_len_drop": summarize(v["log_len"]),
                             "coverage_change": summarize(v["coverage"]),
                             "coverage_change_per_log_len": summarize(v["coverage_per_log_len"]),
                             "R_change": summarize(v["R"]), "R_change_per_log_len": summarize(v["R_per_log_len"]),
                             "two_seed_coverage_change": summarize(v["seed_cov"]),
                             "local_coverage_change": summarize(v["local"])}
        s = out["summary"][c]
        print(f"{c:7s} log-len drop {s['log_len_drop']['min']:.2f}..{s['log_len_drop']['max']:.2f} | "
              f"coverage {s['coverage_change']['min']:+.3f}..{s['coverage_change']['max']:+.3f} "
              f"(per log-len {s['coverage_change_per_log_len']['min']:+.2f}..{s['coverage_change_per_log_len']['max']:+.2f}) | "
              f"R {s['R_change']['min']:+.3f}..{s['R_change']['max']:+.3f} | "
              f"2-seed cov {s['two_seed_coverage_change']['min']:+.3f}..{s['two_seed_coverage_change']['max']:+.3f} | "
              f"local {s['local_coverage_change']['min']:+.3f}..{s['local_coverage_change']['max']:+.3f}")
    (X / "length_confound.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
