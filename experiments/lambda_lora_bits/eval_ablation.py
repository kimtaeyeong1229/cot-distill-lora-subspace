#!/usr/bin/env python3
"""Evaluate ablated adapters on the behaviour the removed subspace is supposed to carry.

The students never emit </think>: the training responses have the tags stripped, so the tag rate is 0
for every adapter and cannot be the read-out. What does separate a long-CoT student from a short-answer
one is how often it hedges and re-checks -- 32.5 reflective markers per 1k words at lambda0 against
0.17 at lambda1, a 190x gap on the same student. That density, together with response length, is what
this measures.

Per adapter it reports accuracy (mean over samples, MI-Distillation grader), mean tokens, truncation
rate, reflective-marker density, and the tag rate, which is kept only to show it stays at zero.
Generations are also written to <name>.samples.jsonl.gz so that a high marker density can be checked
against the text (a looping response repeats "wait" without re-checking anything).
"""

import argparse
import gzip
import json
import re
from pathlib import Path

import pandas as pd
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from common import build_prompt, is_correct, load_tokenizer, read_jsonl
from config import REFLECTIVE   # hedging and self-correction markers (see config.py)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapters", nargs="+", required=True, help="name=adapter_dir pairs")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--n_samples", type=int, default=2)
    ap.add_argument("--max_new_tokens", type=int, default=7680)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = (pd.read_parquet(a.data).to_dict("records") if a.data.endswith(".parquet")
            else read_jsonl(a.data))
    problems = rows[:a.limit]
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for spec in a.adapters:
        name, _, path = spec.partition("=")
        if (out_dir / f"{name}.json").exists():
            continue
        jobs.append((name, path))
    if not jobs:
        print("모두 완료됨")
        return

    tokenizer = load_tokenizer(a.base)
    prompts = [build_prompt(tokenizer, r["problem"]) for r in problems]
    llm = LLM(model=a.base, enable_lora=True, max_loras=1, max_lora_rank=32,
              max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_memory_utilization,
              seed=a.seed)
    params = SamplingParams(n=a.n_samples, temperature=0.6, top_p=0.95, top_k=-1,
                            max_tokens=a.max_new_tokens, seed=a.seed)

    for lora_id, (name, path) in enumerate(jobs, start=1):
        request = LoRARequest(name, lora_id, str(path)) if path else None
        outputs = llm.generate(prompts, params, lora_request=request)
        correct = tokens = truncated = markers = words = tagged = total = 0
        records = []
        for row, out in zip(problems, outputs):
            for c in out.outputs:
                records.append({"problem": row["problem"], "text": c.text,
                                "finish_reason": c.finish_reason, "tokens": len(c.token_ids)})
                total += 1
                correct += is_correct(row["problem"], c.text, str(row["answer"]))
                tokens += len(c.token_ids)
                truncated += c.finish_reason == "length"
                markers += len(REFLECTIVE.findall(c.text))
                words += len(c.text.split())
                tagged += "</think>" in c.text
        summary = {
            "name": name, "adapter": path, "problems": len(problems), "samples": total,
            "accuracy": correct / total,
            "avg_tokens": tokens / total,
            "truncated_rate": truncated / total,
            "reflective_per_1k_words": markers / max(words, 1) * 1000,
            "think_tag_rate": tagged / total,
        }
        with gzip.open(out_dir / f"{name}.samples.jsonl.gz", "wt") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        (out_dir / f"{name}.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
