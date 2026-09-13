#!/usr/bin/env python3
"""Evaluate the base student and LoRA adapters on a math set with one vLLM engine.

Accuracy is the mean over n sampled answers per problem (temperature 0.6,
top-p 0.95), graded with the MI-Distillation grader. Adapters whose result file
already exists are skipped.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from common import build_prompt, is_correct, load_tokenizer, read_jsonl


def load_problems(path: str, limit):
    rows = pd.read_parquet(path).to_dict("records") if path.endswith(".parquet") else read_jsonl(path)
    return rows[:limit] if limit else rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapters", nargs="*", default=[], help="name=adapter_dir pairs.")
    parser.add_argument("--include_base", action="store_true")
    parser.add_argument("--data", required=True, help="JSONL or parquet with problem, answer (and level).")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--max_model_len", type=int, default=9216)
    parser.add_argument("--max_lora_rank", type=int, default=16)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    jobs = [("base", None)] if args.include_base else []
    jobs += [tuple(spec.split("=", 1)) for spec in args.adapters]
    jobs = [(name, path) for name, path in jobs if not (out_dir / f"{name}.json").exists()]
    if not jobs:
        print("all results exist")
        return

    problems = load_problems(args.data, args.limit)
    tokenizer = load_tokenizer(args.base)
    prompts = [build_prompt(tokenizer, row["problem"]) for row in problems]
    llm = LLM(
        model=args.base,
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
        max_loras=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )
    params = SamplingParams(
        n=args.n_samples,
        temperature=0.6,
        top_p=0.95,
        top_k=-1,
        repetition_penalty=1.0,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    for lora_id, (name, path) in enumerate(jobs, start=1):
        request = LoRARequest(name, lora_id, str(path)) if path else None
        outputs = llm.generate(prompts, params, lora_request=request)
        records, per_run = [], [0] * args.n_samples
        for row, out in zip(problems, outputs):
            grades = [is_correct(row["problem"], c.text, str(row["answer"])) for c in out.outputs]
            for k, ok in enumerate(grades):
                per_run[k] += ok
            records.append({
                "unique_id": row.get("unique_id"),
                "level": str(row.get("level", "")),
                "correct": grades,
                "tokens": [len(c.token_ids) for c in out.outputs],
                "truncated": [c.finish_reason == "length" for c in out.outputs],
                "response_0": out.outputs[0].text,
            })

        n = len(problems)
        run_acc = [c / n for c in per_run]
        all_tokens = [t for r in records for t in r["tokens"]]
        by_level = {}
        for level in sorted({r["level"] for r in records}):
            group = [ok for r in records if r["level"] == level for ok in r["correct"]]
            by_level[level] = sum(group) / len(group)
        summary = {
            "name": name,
            "adapter": path,
            "data": args.data,
            "problems": n,
            "n_samples": args.n_samples,
            "accuracy": sum(run_acc) / len(run_acc),
            "accuracy_runs": run_acc,
            "avg_tokens": sum(all_tokens) / len(all_tokens),
            "truncated_rate": sum(t for r in records for t in r["truncated"]) / len(all_tokens),
            "accuracy_by_level": by_level,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{name}.json").write_text(json.dumps({"summary": summary, "records": records}, ensure_ascii=False))
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
