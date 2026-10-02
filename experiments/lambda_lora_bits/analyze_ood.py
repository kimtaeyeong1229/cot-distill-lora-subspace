#!/usr/bin/env python3
"""The section-5 removal outside MATH (eval_bench_edits.py): GPQA-Diamond and GSM8K.

Per cell, benchmark and condition (lambda0, shared region removed, same energy removed elsewhere, lambda1):
accuracy, mean tokens and the reflective-marker density of the first sample, as mean +- SD over seeds 0-2.
GPQA-Diamond is four-way multiple choice, so its accuracy is only read against the 25% guessing rate.

Run from the repo root; writes experiments/ood.json.
"""

import json
import statistics as st
from pathlib import Path

E = Path("experiments/lambda_lora_bits")
CELLS = {"A/Qwen": "sweep", "A/Llama": "sweep_llama", "A/OLMo": "sweep_olmo2",
         "B/Qwen": "pairB/qwen0.5b", "B/Llama": "pairB/llama1b", "B/OLMo": "pairB/olmo1b",
         "C/Qwen": "pairD/qwen0.5b", "C/Llama": "pairD/llama1b", "C/OLMo": "pairD/olmo1b"}
CONDS = {"base0": "lambda0.0", "rm_pre": "lambda0.0_rm_pre", "rm_other": "lambda0.0_rm_decisive", "base1": "lambda1.0"}
KEYS = {"accuracy": "accuracy", "tokens": "avg_tokens", "markers": "reflective_per_1k_words", "truncated": "truncated_rate"}


def main():
    out = {}
    for bench in ("gpqa_diamond", "gsm8k"):
        out[bench] = {}
        for cell, tag in CELLS.items():
            row = {}
            for c, name in CONDS.items():
                v = [json.loads((E / tag / f"eval_{bench}_s{s}" / f"{name}.json").read_text()) for s in (0, 1, 2)]
                row[c] = {k: {"mean": st.mean(x[f] for x in v), "sd": st.stdev(x[f] for x in v), "seeds": [x[f] for x in v]}
                          for k, f in KEYS.items()}
            out[bench][cell] = row
        mean = lambda c, k: st.mean(r[c][k]["mean"] for r in out[bench].values())
        print(f"== {bench}")
        for c in CONDS:
            print(f"  {c:9s} acc {100 * mean(c, 'accuracy'):5.1f}  tokens {mean(c, 'tokens'):6.0f}  markers {mean(c, 'markers'):5.1f}"
                  f"  trunc {mean(c, 'truncated'):.2f}")
    Path("experiments/ood.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
