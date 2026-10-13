#!/usr/bin/env python3
import json
import statistics as st
import torch
from ablate_subspace import consensus, left_basis, read_adapter
from common import build_prompt, load_tokenizer, read_jsonl
from config import CELL_DIR, EXP as E, LAMBDAS as LAM, MODELS as MODEL_DIR, PAIR_DIR, RESULTS as X, SEEDS
from config import STUDENT_SHORT

PAIRS = dict(PAIR_DIR)
RUNS = {p: {s: CELL_DIR[f"{p}/{s}"] for s in STUDENT_SHORT} for p in PAIR_DIR}
OLMO = MODEL_DIR / "OLMo-2-1B-Instruct"


def train_think_rate():
    out = {}
    for pair, d in PAIRS.items():
        train = set(json.loads((E / d / "splits" / "split_ids.json").read_text())["train"])
        out[pair] = {}
        for lam in LAM:
            rows = [r for r in read_jsonl(E / d / "cot" / f"lambda{lam}.jsonl") if r["unique_id"] in train]
            out[pair][lam] = {
                "rate": sum(("</think>" in r["cot_response"] for r in rows)) / len(rows),
                "n": len(rows),
            }
        print(
            "train think rate",
            pair,
            " ".join((f"{l}:{v['rate']:.3f}" for l, v in out[pair].items())),
            flush=True,
        )
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
        print(
            "OLMo over 4096",
            pair,
            " ".join((f"{l}:{v['share']:.3f}" for l, v in out[pair].items())),
            flush=True,
        )
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
                for key, (B, A, sc) in ads["0.0", held].items():
                    V, k = consensus([left_basis(*ads["0.0", s][key]) for s in members], threshold=0.75)
                    dw0 = sc * B @ A
                    B1, A1, sc1 = ads["1.0", held][key]
                    dw1 = sc1 * B1 @ A1
                    in0 += float((V.T @ dw0).pow(2).sum())
                    tot0 += float(dw0.pow(2).sum())
                    in1 += float((V.T @ dw1).pow(2).sum())
                    tot1 += float(dw1.pow(2).sum())
                    dims += k
                    chance += k / B.shape[0]
                    n += 1
                per.append(
                    {
                        "share_0.0": in0 / tot0,
                        "share_1.0": in1 / tot1,
                        "dims": dims / n,
                        "chance_share": chance / n,
                    }
                )
            cell = f"{pair}/{short}"
            out[cell] = {k: st.mean((p[k] for p in per)) for k in per[0]}
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
                "class": r["class"],
                "log_len_drop": r["log_len_drop"],
                "marker_log_drop": r["marker_log_ratio"],
                "per_log_len": {
                    s: -math.log(k) / r["log_len_drop"] for s, k in r["kept_vs_half_reference"].items()
                },
                "per_log_marker": {
                    s: -math.log(k) / r["marker_log_ratio"] for s, k in r["kept_vs_half_reference"].items()
                },
            }
    a = out["A"]
    out["A_switch_over_held"] = {
        s: a["0.2->0.4"]["per_log_len"][s] / a["0.0->0.2"]["per_log_len"][s]
        for s in a["0.0->0.2"]["per_log_len"]
    }
    print(
        "per-unit loss A switch/held",
        {s: round(x, 2) for s, x in out["A_switch_over_held"].items()},
        flush=True,
    )
    return out


def main():
    result = {
        "train_think_rate": train_think_rate(),
        "olmo_over_4096": olmo_over_4096(),
        "two_seed_energy": two_seed_energy(),
        "per_unit_loss": per_unit_loss(),
    }
    (X / "supplementary_stats.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
