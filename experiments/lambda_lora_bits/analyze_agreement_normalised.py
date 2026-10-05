#!/usr/bin/env python3
"""How much of the region's emptying over a lambda interval is a general loss of overlap between adapters?
Read-only re-analysis of two existing JSONs (no adapters are loaded).

Inputs
  experiments/length_confound.json  pairs.<P>.intervals[i]: interval a->b, class, log_len_drop and
      kept_vs_half_reference.<student> = coverage of the region built at lambda a by the lambda b adapters,
      relative to half-data adapters at lambda a (the Table 1 / Table 8 "kept"; post hoc reference).
  experiments/phi_xseed.json  <P'>/<student>.<lambda> (P' = A, B, D for pairs A, B, C), from analyze_phi_xseed.py,
      with phi the norm-weighted subspace overlap of Eq. 2:
        xseed = mean phi(lambda=0 adapter of seed i, lambda adapter of seed j), i != j
                (overlap of the whole lambda update with lambda = 0 directions, no shared LoRA initialisation);
        null  = mean phi(lambda adapter of seed i, lambda adapter of seed j), i < j
                (cross-seed self-agreement of adapters at the same lambda).

Definitions, per cell and interval a->b
  overlap_ratio   = xseed(b) / xseed(a)    how much of its overlap with the lambda = 0 directions the whole
                                            update keeps over the interval. This is the reviewer's "cross-seed
                                            agreement kept" (c14 meta 0.2); at lambda = 0 xseed is self-agreement,
                                            elsewhere it is overlap with lambda = 0 adapters.
  kept_over_overlap = kept / overlap_ratio  1 means the region lost no more than the whole update's overlap with
                                            lambda = 0 did; 1 - value is the region's excess loss.
  selfagree_ratio = null(b) / null(a)       how much cross-seed agreement among same-lambda adapters falls.
  kept_over_selfagree = kept / selfagree_ratio
  per-unit loss of the overlap = -ln(overlap_ratio) / log_len_drop (for the pair-A switch/held comparison).

Writes experiments/agreement_normalised.json and prints the summary ranges.
"""

import json
import math

from config import LAMBDAS as LAMS, PAIR_KEY as PKEY, RESULTS as X

STUD = {"Qwen": "Qwen-0.5B", "Llama": "Llama-1B", "OLMo": "OLMo-1B"}


def main():
    lc = json.loads((X / "length_confound.json").read_text())
    px = json.loads((X / "phi_xseed.json").read_text())
    cells = {}
    for p in "ABC":
        for s, full in STUD.items():
            phi = px[f"{PKEY[p]}/{full}"]
            rows = []
            for i, iv in enumerate(lc["pairs"][p]["intervals"]):
                a, b = LAMS[i], LAMS[i + 1]
                kept = iv["kept_vs_half_reference"][s]
                ro = phi[b]["xseed"] / phi[a]["xseed"]
                rs = phi[b]["null"] / phi[a]["null"]
                d = iv["log_len_drop"]
                rows.append({"interval": iv["interval"], "class": iv["class"], "log_len_drop": d, "kept": kept,
                             "overlap_ratio": ro, "kept_over_overlap": kept / ro,
                             "selfagree_ratio": rs, "kept_over_selfagree": kept / rs,
                             "overlap_loss_per_unit": (-math.log(ro) / d) if abs(d) >= 0.05 else None})
            cells[f"{p}/{s}"] = rows

    def rng(p, i, key):
        v = [cells[f"{p}/{s}"][i][key] for s in STUD]
        return [min(v), max(v)]

    summary = {
        "A_held": {k: rng("A", 0, k) for k in ("kept", "overlap_ratio", "kept_over_overlap", "selfagree_ratio", "kept_over_selfagree")},
        "A_switch": {k: rng("A", 1, k) for k in ("kept", "overlap_ratio", "kept_over_overlap", "selfagree_ratio", "kept_over_selfagree")},
        "A_switch_excess_loss_vs_overlap": [1 - x for x in reversed(rng("A", 1, "kept_over_overlap"))],
        "A_overlap_switch_over_held_per_unit": [min(r), max(r)] if (r := [cells[f"A/{s}"][1]["overlap_loss_per_unit"] / cells[f"A/{s}"][0]["overlap_loss_per_unit"] for s in STUD]) else None,
        "B_switch": {k: rng("B", 2, k) for k in ("kept", "overlap_ratio", "kept_over_overlap", "selfagree_ratio", "kept_over_selfagree")},
        "B_after": {k: rng("B", 3, k) for k in ("kept", "overlap_ratio", "kept_over_overlap", "selfagree_ratio", "kept_over_selfagree")},
        "C_first_step": {k: rng("C", 2, k) for k in ("kept", "overlap_ratio", "kept_over_overlap", "selfagree_ratio", "kept_over_selfagree")},
        "C_second_step": {k: rng("C", 3, k) for k in ("kept", "overlap_ratio", "kept_over_overlap", "selfagree_ratio", "kept_over_selfagree")},
    }
    bc = rng("B", 2, "kept_over_overlap") + rng("C", 2, "kept_over_overlap")
    summary["BC_first_step_kept_over_overlap"] = [min(bc), max(bc)]
    bs = rng("B", 2, "selfagree_ratio") + rng("C", 2, "selfagree_ratio")
    summary["BC_first_step_selfagree_ratio"] = [min(bs), max(bs)]
    out = {"cells": cells, "summary": summary}
    (X / "agreement_normalised.json").write_text(json.dumps(out, indent=1))
    for k, v in summary.items():
        print(k, json.dumps(v, default=lambda x: round(x, 3)) if not isinstance(v, list) else [round(x, 3) for x in v])


if __name__ == "__main__":
    main()
