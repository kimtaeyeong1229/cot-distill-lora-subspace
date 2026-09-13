#!/usr/bin/env python3
"""Measure what a LoRA adapter wrote, in bits (after Tan et al., arXiv:2607.21351).

Behaviour-write bits on the final policy:
    W(D)  = sum_{(x,y) in D} [ H(y | x, base) - H(y | x, base+adapter) ],  H = -log2 p
    dW    = W(D_train) - (n_train / n_held) * W(D_held)        (count-scaled, as in the paper)
    dW_tok= W(D_train) - (T_train / T_held) * W(D_held)        (token-scaled robustness check)
    clipped variants sum max(0, per-sample saving), matching the min() in mem_U.

Parameter-write cost CL: each LoRA tensor's update from the shared step-0 init
(identical across arms, so a decoder already has it) is uniformly quantised with
step c * RMS(update) and costed by the empirical entropy of its integer symbols
plus a 32-bit step. Wider steps round small entries to zero and cost fewer bits.
CL is the cheapest step multiplier c* whose quantised adapter keeps >= 99% of W
on a fixed train subset. Max-scaled bit-width quantisation was tried first, but
it kept W even at 2 bits and so only measured parameter count times symbol
entropy. The paper's supplement defining CL is not public; this is our
operationalisation.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from peft import PeftModel, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from common import encode, label_count, load_tokenizer, read_jsonl, sample_bits


def load_set(tokenizer, path, max_len):
    rows = read_jsonl(path)
    encoded = [encode(tokenizer, r["problem"], r["response"], max_len) for r in rows]
    return rows, encoded, [label_count(labels) for _, labels in encoded]


def base_bits(model, tokenizer, path, encoded, cache_dir, args):
    path = Path(path).resolve()
    key = hashlib.sha1(f"{path}:{path.stat().st_mtime_ns}:{Path(args.base).resolve()}:{args.max_len}".encode()).hexdigest()[:12]
    cache = Path(cache_dir) / f"{path.parent.name}_{path.stem}_{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    with model.disable_adapter():
        bits = sample_bits(model, encoded, tokenizer.pad_token_id, args.token_budget)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(bits))
    return bits


def summarize(base, tuned, tokens):
    savings = [b - t for b, t in zip(base, tuned)]
    H, W, T = sum(base), sum(savings), sum(tokens)
    return {
        "n": len(base),
        "tokens": T,
        "H_base_bits": H,
        "H_tuned_bits": sum(tuned),
        "W_bits": W,
        "W_clipped_bits": sum(max(0.0, s) for s in savings),
        "H_base_bits_per_token": H / T,
        "W_bits_per_token": W / T,
        "W_bits_per_sample": W / len(base),
        "W_over_H": W / H,
    }


def quantize_step(x: torch.Tensor, multiplier: float):
    """Round x to a grid of step multiplier * RMS(x); return (dequantised, cost in bits, zero fraction)."""
    rms = x.pow(2).mean().sqrt()
    if rms == 0:
        return torch.zeros_like(x), 32.0, 1.0
    step = multiplier * rms
    q = torch.round(x / step)
    _, counts = torch.unique(q, return_counts=True)
    p = counts.double() / q.numel()
    cost = float(-(counts.double() * torch.log2(p)).sum()) + 32.0
    return q * step, cost, float((q == 0).double().mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--run_dir", required=True, help="Directory with adapter/ and adapter_init/.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--held", required=True)
    parser.add_argument("--cross", nargs="*", default=[], help="name=path.jsonl sets for the cross-lambda matrix.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache_dir", default="experiments/lambda_lora_bits/cache/base_bits")
    # Adam updates are dense and flat, so the cost collapses somewhere between c=1 and c=4;
    # the grid is fine there to locate the cheapest step that still reproduces W.
    parser.add_argument("--step_multipliers", default="0.25 0.5 0.75 1 1.25 1.5 2 2.5 3 4")
    parser.add_argument("--fidelity_samples", type=int, default=128)
    parser.add_argument("--fidelity_threshold", type=float, default=0.99)
    parser.add_argument("--max_len", type=int, default=8192)
    parser.add_argument("--token_budget", type=int, default=8192)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    tokenizer = load_tokenizer(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model = PeftModel.from_pretrained(model, run_dir / "adapter").eval()
    pad = tokenizer.pad_token_id
    result = {"run_dir": str(run_dir), "train_file": args.train, "held_file": args.held}

    # --- behaviour-write bits -------------------------------------------------
    sets = {}
    for name, path in [("train", args.train), ("held", args.held)]:
        rows, encoded, tokens = load_set(tokenizer, path, args.max_len)
        base = base_bits(model, tokenizer, path, encoded, args.cache_dir, args)
        tuned = sample_bits(model, encoded, pad, args.token_budget)
        sets[name] = (encoded, base, tuned)
        result[name] = summarize(base, tuned, tokens)
        print(name, json.dumps(result[name]))

    tr, he = result["train"], result["held"]
    result["delta_W_bits"] = tr["W_bits"] - tr["n"] / he["n"] * he["W_bits"]
    result["delta_W_token_scaled_bits"] = tr["W_bits"] - tr["tokens"] / he["tokens"] * he["W_bits"]
    result["delta_W_clipped_bits"] = tr["W_clipped_bits"] - tr["n"] / he["n"] * he["W_clipped_bits"]
    result["delta_W_bits_per_train_token"] = result["delta_W_bits"] / tr["tokens"]

    # --- parameter-write cost ------------------------------------------------
    final = {k: v.float().cuda() for k, v in load_file(run_dir / "adapter" / "adapter_model.safetensors").items()}
    init = {k: v.float().cuda() for k, v in load_file(run_dir / "adapter_init" / "adapter_model.safetensors").items()}
    n_params = sum(v.numel() for v in final.values())
    result["trainable_params"] = n_params

    subset = sets["train"][0][: args.fidelity_samples]
    sub_base = sum(sets["train"][1][: args.fidelity_samples])
    sub_W = sub_base - sum(sets["train"][2][: args.fidelity_samples])

    sweep = {}
    for multiplier in [float(m) for m in args.step_multipliers.split()]:
        state, cost, zeros = {}, 0.0, 0.0
        for key in final:
            deq_update, bits, zero_fraction = quantize_step(final[key] - init[key], multiplier)
            state[key] = init[key] + deq_update
            cost += bits
            zeros += zero_fraction * final[key].numel()
        set_peft_model_state_dict(model, state)
        W_q = sub_base - sum(sample_bits(model, subset, pad, args.token_budget))
        sweep[multiplier] = {
            "bits": cost,
            "bits_per_param": cost / n_params,
            "zero_fraction": zeros / n_params,
            "retained_W": W_q / sub_W if sub_W else float("nan"),
        }
        print(f"c={multiplier:g}: CL={cost:.0f} bits ({cost / n_params:.3f}/param), "
              f"zeros={zeros / n_params:.3f}, retained W={sweep[multiplier]['retained_W']:.4f}")
    set_peft_model_state_dict(model, final)

    passing = [m for m, v in sweep.items() if v["retained_W"] >= args.fidelity_threshold]
    c_star = min(passing, key=lambda m: sweep[m]["bits"]) if passing else min(sweep)
    cl = {
        "sweep": sweep,
        "fidelity_met": bool(passing),
        "fidelity_subset_W_bits": sub_W,
        "c_star": c_star,
        "CL_update_bits": sweep[c_star]["bits"],
        "CL_update_bits_per_param": sweep[c_star]["bits_per_param"],
        "zero_fraction": sweep[c_star]["zero_fraction"],
    }
    result["CL"] = cl

    # --- cross-lambda matrix --------------------------------------------------
    result["cross"] = {}
    for spec in args.cross:
        name, path = spec.split("=", 1)
        _, encoded, tokens = load_set(tokenizer, path, args.max_len)
        base = base_bits(model, tokenizer, path, encoded, args.cache_dir, args)
        tuned = sample_bits(model, encoded, pad, args.token_budget)
        result["cross"][name] = summarize(base, tuned, tokens)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ["delta_W_bits", "delta_W_token_scaled_bits", "trainable_params"]}))
    print(json.dumps({k: cl[k] for k in ["c_star", "fidelity_met", "CL_update_bits", "CL_update_bits_per_param"]}))


if __name__ == "__main__":
    main()
