#!/usr/bin/env python3
"""Summarise the SAE pilot on the three Llama cells (sae_llama.py score) next to the keyword metric.

For every condition and seed: the keyword density (reflective markers per 1k words), the SAE index of the
selected features on the responses as written, the same index with the reflective phrases deleted, and
the index of the cue-driven features. Tables report the mean +- SD over seeds 0-2, as in the paper.
Also reported: the correlation between the SAE index and the keyword density across condition x seed,
and a verification subset (features whose top contexts are self-checking; chosen by reading them, so it
is exploratory).

Run from the repo root; writes experiments/sae_llama.json.
"""

import json
import statistics as st
from pathlib import Path

D = Path("experiments/lambda_lora_bits/sae_llama")
CONDS = ["base0.0", "base0.2", "base0.4", "base0.6", "base0.8", "base1.0", "lambda0.0_rm_pre", "graft2", "graft_orth2"]
VERIFY = [29679, 123803, 74538, 17318, 72260, 76271]   # self-questioning / checking / trying another way


def per_seed(rec, key, feats=None, allf=None):
    if feats is None:
        return st.mean(r[key] for r in rec)
    j = [allf.index(f) for f in feats]
    return st.mean(st.mean(r[key][k] for k in j) for r in rec)


def main():
    out = {}
    for cell in "ABC":
        row = {}
        for cond in CONDS:
            vals = {k: [] for k in ("keyword", "sae", "sae_masked", "cue", "verify", "verify_masked", "sae_head")}
            for s in (0, 1, 2):
                d = json.loads((D / cell / f"{cond}_s{s}.json").read_text()); rec, allf = d["records"], d["features"]
                vals["keyword"].append(1000 * sum(r["markers"] for r in rec) / max(sum(r["words"] for r in rec), 1))
                vals["sae"].append(per_seed(rec, "orig_selected")); vals["sae_masked"].append(per_seed(rec, "masked_selected"))
                vals["cue"].append(per_seed(rec, "orig_cue")); vals["sae_head"].append(per_seed(rec, "orig_selected_head"))
                vals["verify"].append(per_seed(rec, "orig_per_feature", VERIFY, allf))
                vals["verify_masked"].append(per_seed(rec, "masked_per_feature", VERIFY, allf))
            row[cond] = {k: {"mean": st.mean(v), "sd": st.stdev(v), "seeds": v} for k, v in vals.items()}
        xs = [v for c in CONDS for v in row[c]["keyword"]["seeds"]]
        for k in ("sae", "sae_masked", "cue", "verify"):
            ys = [v for c in CONDS for v in row[c][k]["seeds"]]
            row[f"corr_keyword_{k}"] = st.correlation(xs, ys)
        out[cell] = row
        print(f"== {cell}/Llama   corr(keyword, sae)={row['corr_keyword_sae']:.2f}  masked={row['corr_keyword_sae_masked']:.2f}  cue={row['corr_keyword_cue']:.2f}  verify={row['corr_keyword_verify']:.2f}")
        for c in CONDS:
            r = row[c]; f = lambda k: f"{r[k]['mean']:.3f}±{r[k]['sd']:.3f}"
            print(f"  {c:18s} kw {r['keyword']['mean']:5.1f}±{r['keyword']['sd']:4.1f} | sae {f('sae')} masked {f('sae_masked')} | cue {f('cue')} | verify {f('verify')} vmask {f('verify_masked')}")
    Path("experiments/sae_llama.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
