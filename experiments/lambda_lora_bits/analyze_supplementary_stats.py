#!/usr/bin/env python3
"""Three supplementary statistics from existing files (no training, no generation).

1. train_think_rate   the teacher's </think> rate on the training subset only (split_ids.json "train"),
                      counted in the raw generations (cot/lambda*.jsonl, field cot_response), next to the
                      rate over all 4,000 problems in experiments/think_rate.json.
2. olmo_over_4096     the share of training examples (prompt + solution + EOS, OLMo-2 chat template and
                      tokenizer, exactly as common.encode builds them) longer than OLMo-2's 4,096-token
                      training length, i.e. the examples whose solution end is cut during training.
3. two_seed_energy    the share of the update energy (||V^T dW||^2 / ||dW||^2, summed over modules) that the
                      two-seed lambda0 region (two lambda0 seeds, eigenvalue cut 0.75) holds in the held-out
                      third lambda0 adapter and in the same seed's lambda1 adapter, with its mean dimension and
                      the share a random subspace of that size would hold; averaged over the three held-out
                      seeds. The shared region's own shares are in experiments/region_anatomy.json.

4. per_unit_loss      from experiments/length_confound.json: the share of the region lost per unit of log
                      training-length drop, -ln(kept) / log_len_drop, with kept the half-reference retention
                      (pairs.*.intervals[].kept_vs_half_reference), and per unit of log teacher-marker drop,
                      -ln(kept) / marker_log_ratio; plus, for pair A, the switch/held ratio of the length-
                      normalised loss per student. The measure assumes no loss at zero length change.

Run from the repo root inside the container; writes experiments/supplementary_stats.json.
"""

import json
import statistics as st
import sys
from pathlib import Path

import torch

X = Path("experiments")
E = X / "lambda_lora_bits"
sys.path.insert(0, str(E))
from ablate_subspace import consensus, left_basis, read_adapter  # noqa: E402
from common import build_prompt, load_tokenizer, read_jsonl  # noqa: E402

LAM = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
SEEDS = [0, 1, 2]
PAIRS = {"A": "sweep", "B": "pairB", "C": "pairD"}
RUNS = {"A": {"Qwen": "sweep", "Llama": "sweep_llama", "OLMo": "sweep_olmo2"},
        "B": {"Qwen": "pairB/qwen0.5b", "Llama": "pairB/llama1b", "OLMo": "pairB/olmo1b"},
        "C": {"Qwen": "pairD/qwen0.5b", "Llama": "pairD/llama1b", "OLMo": "pairD/olmo1b"}}
OLMO = Path("models/OLMo-2-1B-Instruct")


def train_think_rate():
    out = {}
    for pair, d in PAIRS.items():
        train = set(json.loads((E / d / "splits" / "split_ids.json").read_text())["train"])
        out[pair] = {}
        for lam in LAM:
            rows = [r for r in read_jsonl(E / d / "cot" / f"lambda{lam}.jsonl") if r["unique_id"] in train]
            out[pair][lam] = {"rate": sum("</think>" in r["cot_response"] for r in rows) / len(rows), "n": len(rows)}
        print("train think rate", pair, " ".join(f"{l}:{v['rate']:.3f}" for l, v in out[pair].items()), flush=True)
    return out


def olmo_over_4096():
    tok = load_tokenizer(str(OLMO))
    out = {}
    for pair, d in PAIRS.items():
        out[pair] = {}
        for lam in LAM:
            rows = read_jsonl(E / d / "splits" / f"lambda{lam}" / "train.jsonl")
            over = 0
            for r in rows:
                n = len(tok(build_prompt(tok, r["problem"]), add_special_tokens=False)["input_ids"])
                n += len(tok(r["response"] + tok.eos_token, add_special_tokens=False)["input_ids"])
                over += n > 4096
            out[pair][lam] = {"share": over / len(rows), "n": len(rows)}
        print("OLMo over 4096", pair, " ".join(f"{l}:{v['share']:.3f}" for l, v in out[pair].items()), flush=True)
    return out


def two_seed_energy():
    def load(tag, lam, seed):
        path = E / tag / "runs" / f"lambda{lam}_seed{seed}" / "adapter"
        scale = (lambda c: c["lora_alpha"] / c["r"])(json.loads((path / "adapter_config.json").read_text()))
        raw, parts = read_adapter(path / "adapter_model.safetensors")
        return {k: (raw[n["B"]].float(), raw[n["A"]].float(), scale) for k, n in parts.items() if len(n) == 2}

    out = {}
    for pair, cells in RUNS.items():
        for short, tag in cells.items():
            ads = {(l, s): load(tag, l, s) for l in ("0.0", "1.0") for s in SEEDS}
            per = []
            for held in SEEDS:
                members = [s for s in SEEDS if s != held]
                in0 = tot0 = in1 = tot1 = dims = chance = 0.0
                n = 0
                for key, (B, A, sc) in ads[("0.0", held)].items():
                    V, k = consensus([left_basis(*ads[("0.0", s)][key]) for s in members], threshold=0.75)
                    dw0 = sc * B @ A
                    B1, A1, sc1 = ads[("1.0", held)][key]
                    dw1 = sc1 * B1 @ A1
                    in0 += float((V.T @ dw0).pow(2).sum())
                    tot0 += float(dw0.pow(2).sum())
                    in1 += float((V.T @ dw1).pow(2).sum())
                    tot1 += float(dw1.pow(2).sum())
                    dims += k
                    chance += k / B.shape[0]
                    n += 1
                per.append({"share_0.0": in0 / tot0, "share_1.0": in1 / tot1, "dims": dims / n, "chance_share": chance / n})
            cell = f"{pair}/{short}"
            out[cell] = {k: st.mean(p[k] for p in per) for k in per[0]}
            print("two-seed region", cell, {k: round(v, 3) for k, v in out[cell].items()}, flush=True)
    return out


def per_unit_loss():
    import math
    lc = json.loads((X / "length_confound.json").read_text())
    out = {}
    for pair, v in lc["pairs"].items():
        out[pair] = {}
        for r in v["intervals"]:
            out[pair][r["interval"]] = {
                "class": r["class"], "log_len_drop": r["log_len_drop"], "marker_log_drop": r["marker_log_ratio"],
                "per_log_len": {s: -math.log(k) / r["log_len_drop"] for s, k in r["kept_vs_half_reference"].items()},
                "per_log_marker": {s: -math.log(k) / r["marker_log_ratio"] for s, k in r["kept_vs_half_reference"].items()}}
    a = out["A"]
    out["A_switch_over_held"] = {s: a["0.2->0.4"]["per_log_len"][s] / a["0.0->0.2"]["per_log_len"][s]
                                 for s in a["0.0->0.2"]["per_log_len"]}
    print("per-unit loss A switch/held", {s: round(x, 2) for s, x in out["A_switch_over_held"].items()}, flush=True)
    return out


def main():
    result = {"train_think_rate": train_think_rate(), "olmo_over_4096": olmo_over_4096(),
              "two_seed_energy": two_seed_energy(), "per_unit_loss": per_unit_loss()}
    (X / "supplementary_stats.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
