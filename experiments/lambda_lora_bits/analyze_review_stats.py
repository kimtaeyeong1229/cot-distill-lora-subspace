#!/usr/bin/env python3
"""Derived statistics from existing result files, for the cycle-5 manuscript revision (read only).

  graft_recovery_early   recovery of the early-marker gap by the region graft, (graft - lambda1) /
                         (lambda0 - lambda1), with early = reflective markers in the first 300 words
                         (main_edits.json head_markers_per_1k), at beta = 1 and 2, per cell
  sae_subsets            Llama SAE feature subsets (sae_llama.json): the cue-driven subset and the exploratory
                         verification subset, read on the beta = 2 region graft and on the lambda1 adapter, as a
                         share of the lambda0 level, and the share of the lambda1 -> lambda0 gap recovered
  cell_interval          the removal accuracy differences (paired_bootstrap.json cells.*.diff_pp) with the nine
                         cells as the unit: the t interval of the nine-cell mean, mean +- t(0.975, 8) * sd / 3
  thresholds             the interval classification's slack (length_confound.json): the largest think-rate
                         change in held intervals and the range of first-step switch drops
  amplification          accuracy and completion of the lambda0 adapter and its fixed-norm amplifications
                         (main_edits.json), per cell and as nine-cell means
  correlation_points     how many points each reported correlation pools
  graft1_whole_recovery  per-seed recovery of the whole-response marker gap by the beta = 1 region graft, with
                         seed-matched lambda0 / lambda1 adapters (seed_spread.json), and its range over seeds
  graft1_lengthening     beta = 1 region graft against the lambda1 adapter: tokens, completion, early-marker
                         recovery and loop rate (main_edits.json, degeneration.json)
  split_difficulty       MATH level of the shared training and held-out problems against all 4,000 problems
                         (split_ids.json, cot/lambda0.0.jsonl level field); a problem is kept only if every
                         teacher in the split run solved it completely, correctly and within 8,192 tokens

Run from the repo root; writes experiments/review_stats.json.
"""

import json
import statistics as st
from pathlib import Path

X = Path("experiments")
CELLS = ["A/Qwen", "A/Llama", "A/OLMo", "B/Qwen", "B/Llama", "B/OLMo", "C/Qwen", "C/Llama", "C/OLMo"]


def main():
    m = json.loads((X / "main_edits.json").read_text())
    sae = json.loads((X / "sae_llama.json").read_text())
    boot = json.loads((X / "paired_bootstrap.json").read_text())
    lc = json.loads((X / "length_confound.json").read_text())
    out = {}

    rec = {}
    for c in CELLS:
        h = {k: m[c][k]["head_markers_per_1k"] for k in ("base0", "base1", "graft1", "graft2")}
        gap = h["base0"] - h["base1"]
        rec[c] = {"beta1": (h["graft1"] - h["base1"]) / gap, "beta2": (h["graft2"] - h["base1"]) / gap}
    out["graft_recovery_early"] = rec

    subsets = {}
    for pair in ("A", "B", "C"):
        row = {}
        for k in ("cue", "verify", "verify_masked", "sae"):
            l0, l1, g = (sae[pair][c][k]["mean"] for c in ("base0.0", "base1.0", "graft2"))
            row[k] = {"lambda0": l0, "lambda1": l1, "graft2": g, "graft2_level": g / l0, "lambda1_level": l1 / l0,
                      "graft2_gap_recovery": (g - l1) / (l0 - l1)}
        row["points"] = len(sae[pair]["base0.0"]["sae"]["seeds"]) * sum(
            1 for k, v in sae[pair].items() if isinstance(v, dict) and "sae" in v)
        subsets[pair] = row
    out["sae_subsets"] = subsets

    cb = {}
    for comp in ("cut", "top", "other"):
        d = [boot["cells"][c][comp]["diff_pp"] for c in CELLS]
        half = 2.306004 * st.stdev(d) / len(d) ** 0.5          # t(0.975, df = 8)
        cb[comp] = {"mean": st.mean(d), "ci95": [st.mean(d) - half, st.mean(d) + half],
                    "cells_positive": sum(x > 0 for x in d)}
    out["cell_interval"] = cb

    held = [abs(r["think_change"]) for p in lc["pairs"].values() for r in p["intervals"] if r["class"] == "held"]
    first = [-r["think_change"] for p in lc["pairs"].values() for r in p["intervals"]
             if r["class"] == "switch" and r["think"][0] >= 50]
    second = [-r["think_change"] for p in lc["pairs"].values() for r in p["intervals"]
              if r["class"] == "switch" and r["think"][0] < 50]
    after = [abs(r["think_change"]) for p in lc["pairs"].values() for r in p["intervals"] if r["class"] == "after"]
    out["thresholds"] = {"held_max_change": max(held), "switch_first_step_drops": [min(first), max(first)],
                         "switch_second_step_drops": second, "after_max_change": max(after)}

    amp = {}
    for c in CELLS:
        amp[c] = {k: {"accuracy": m[c][k]["accuracy"],
                      "completed": m[c][k]["completed"] if "completed" in m[c][k] else 1 - m[c][k]["truncated"]}
                  for k in ("base0", "ampn1", "ampn4", "ampV")}
    amp["nine_cell"] = {k: {q: st.mean(amp[c][k][q] for c in CELLS) for q in ("accuracy", "completed")}
                        for k in ("base0", "ampn1", "ampn4", "ampV")}
    out["amplification"] = amp

    out["correlation_points"] = {
        "student_vs_teacher_markers_per_cell": 6, "student_vs_teacher_markers_pooled": 54,
        "R_drop_vs_teacher_drop_per_cell": 5, "sae_vs_markers_per_pair": subsets["A"]["points"]}

    spread = json.loads((X / "seed_spread.json").read_text())
    deg = json.loads((X / "degeneration.json").read_text())
    g1 = {}
    for c in CELLS:
        b0 = spread[c]["base"]["base0_markers_per_1k"]["seeds"]
        b1 = spread[c]["base"]["base1_markers_per_1k"]["seeds"]
        g = spread[c]["edits"]["graft1"]["markers_per_1k"]["seeds"]
        per = [(gi - b1i) / (b0i - b1i) for gi, b0i, b1i in zip(g, b0, b1)]
        g1[c] = {"seeds": per, "min": min(per), "max": max(per)}
    out["graft1_whole_recovery"] = g1
    out["graft1_lengthening"] = {c: {"tokens_lambda1": m[c]["base1"]["avg_tokens"], "tokens_graft1": m[c]["graft1"]["avg_tokens"],
                                     "token_ratio": m[c]["graft1"]["avg_tokens"] / m[c]["base1"]["avg_tokens"],
                                     "completed_graft1": m[c]["graft1"]["completed"],
                                     "completed_lambda1": 1 - m[c]["base1"]["truncated"],
                                     "early_recovery": rec[c]["beta1"], "loop_graft1": deg[c]["graft1"]["mean"]}
                                 for c in CELLS}
    diff = {}
    E = X / "lambda_lora_bits"
    for pair, d in (("A", "sweep"), ("B", "pairB"), ("C", "pairD")):
        rows = [json.loads(l) for l in open(E / d / "cot" / "lambda0.0.jsonl")]
        ids = json.loads((E / d / "splits" / "split_ids.json").read_text())
        keep = set(ids["train"]) | set(ids["held"])
        lv = lambda r: int(str(r.get("level", "")).replace("Level ", "")) if str(r.get("level", "")).replace("Level ", "").isdigit() else None
        allv = [lv(r) for r in rows if lv(r)]
        kv = [lv(r) for r in rows if lv(r) and r["unique_id"] in keep]
        diff[pair] = {"problems": len(rows), "kept": len(keep), "mean_level_all": st.mean(allv), "mean_level_kept": st.mean(kv),
                      "level5_share_all": sum(v == 5 for v in allv) / len(allv), "level5_share_kept": sum(v == 5 for v in kv) / len(kv),
                      "split_lambdas": list(json.loads((E / d / "splits" / "split_stats.json").read_text())["per_lambda"])}
    out["split_difficulty"] = diff
    (X / "review_stats.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"recovery": {c: {k: round(v, 3) for k, v in r.items()} for c, r in rec.items()},
                      "cell_interval": cb, "thresholds": out["thresholds"],
                      "amp9": amp["nine_cell"], "points": out["correlation_points"]}, indent=1))
    for p, r in subsets.items():
        print(p, {k: {q: round(v, 3) for q, v in r[k].items()} for k in ("cue", "verify", "verify_masked")})


if __name__ == "__main__":
    main()
