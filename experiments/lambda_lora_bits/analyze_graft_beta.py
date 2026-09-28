#!/usr/bin/env python3
"""Table 8: grafting the shared region into the lambda1 adapter at beta = 2, 4, 8.

At beta >= 2 most grafted students run to the token limit repeating one sentence, so the marker density
over the whole response counts the loop, not the reasoning format. The density here is taken over the
first 300 words of each response, before a loop can pile up, and is reported next to the share of
responses that finished on their own and the accuracy. The lambda0 reference is the whole-response
density of the unedited lambda0 adapter (eval_ablation_loo), which stored no generations to recount.

Run from the repo root; writes experiments/graft_beta.json.
"""

import gzip
import json
import re
from pathlib import Path

# same pattern as eval_ablation.REFLECTIVE (not imported: that module pulls in vllm)
REFLECTIVE = re.compile(
    r"\b(wait|hmm|actually|but let me|let me check|recheck|verify|reconsider|alternatively|"
    r"on second thought|let me re|double-check|double check|is that right|hold on)\b", re.I)

E = Path("experiments/lambda_lora_bits")
CELLS = {"A/Qwen": "sweep", "A/Llama": "sweep_llama", "A/OLMo": "sweep_olmo2",
         "B/Qwen": "pairB/qwen0.5b", "B/Llama": "pairB/llama1b", "B/OLMo": "pairB/olmo1b",
         "C/Qwen": "pairD/qwen0.5b", "C/Llama": "pairD/llama1b", "C/OLMo": "pairD/olmo1b"}
HEAD = 300


def summarise(path):
    records = [json.loads(l) for l in gzip.open(path, "rt")]
    markers = words = 0
    for r in records:
        head = r["text"].split()[:HEAD]
        markers += len(REFLECTIVE.findall(" ".join(head)))
        words += len(head)
    return {"head_markers_per_1k": markers / max(words, 1) * 1000,
            "completed": sum(r["finish_reason"] == "stop" for r in records) / len(records)}


def main():
    out = {}
    for cell, tag in CELLS.items():
        base = json.loads((E / tag / "eval_ablation_loo" / "lambda0.0_base.json").read_text())
        row = {"lambda0_markers_per_1k": base["reflective_per_1k_words"]}
        for variant in ("graft", "graft_orth", "graft_full"):
            for beta in (2, 4, 8):
                name = f"{variant}{beta}"
                samples = E / tag / "eval_graft_beta" / f"{name}.samples.jsonl.gz"
                if not samples.exists():
                    continue
                summary = json.loads(samples.with_name(f"{name}.json").read_text())
                row[name] = {**summarise(samples), "accuracy": summary["accuracy"],
                             "truncated": summary["truncated_rate"]}
        out[cell] = row
        g, o = row.get("graft2", {}), row.get("graft_orth2", {})
        print(f"{cell:8} shared {g.get('head_markers_per_1k', float('nan')):6.2f} "
              f"({g.get('completed', float('nan')):.2f} done) | other "
              f"{o.get('head_markers_per_1k', float('nan')):5.2f} ({o.get('completed', float('nan')):.2f} done)")
    Path("experiments/graft_beta.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
