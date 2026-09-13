#!/usr/bin/env python3
"""Extraction test: does an adapter reproduce the teacher CoTs it was trained on?

Each model greedy-decodes the training problems and, as a control, the held-out
problems, and the output is compared with the teacher CoT the adapter saw. A
memorising adapter should match its training references far more closely than it
matches held-out references, and far more closely than the untuned base does.
"""

import argparse
import difflib
import json
import statistics
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from common import build_prompt, load_tokenizer, read_jsonl

COMPARE_TOKENS = 512  # cap for the similarity ratio; verbatim reuse shows up early


def prefix_length(a, b):
    count = 0
    for x, y in zip(a, b):
        if x != y:
            break
        count += 1
    return count


def compare(outputs, references):
    prefixes = [prefix_length(a, b) for a, b in zip(outputs, references)]
    ratios = [
        difflib.SequenceMatcher(a=a[:COMPARE_TOKENS], b=b[:COMPARE_TOKENS], autojunk=False).ratio()
        for a, b in zip(outputs, references)
    ]
    n = len(prefixes)
    return {
        "n": n,
        "prefix_mean": statistics.mean(prefixes),
        "prefix_median": statistics.median(prefixes),
        "prefix_max": max(prefixes),
        "frac_prefix_ge_32": sum(x >= 32 for x in prefixes) / n,
        "ratio_mean": statistics.mean(ratios),
        "ratio_median": statistics.median(ratios),
        "frac_ratio_ge_0.8": sum(x >= 0.8 for x in ratios) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--arms", nargs="+", required=True, help="name=adapter_dir=split_dir triples.")
    parser.add_argument("--include_base", action="store_true")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--max_lora_rank", type=int, default=16)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    arms = [spec.split("=") for spec in args.arms]
    tokenizer = load_tokenizer(args.base)

    references, prompts = {}, {}
    for name, _, split_dir in arms:
        for split in ("train", "held"):
            rows = read_jsonl(Path(split_dir) / f"{split}.jsonl")[: args.limit]
            references[(name, split)] = [tokenizer(r["response"], add_special_tokens=False)["input_ids"] for r in rows]
            prompts[split] = [build_prompt(tokenizer, r["problem"]) for r in rows]  # identical across arms

    llm = LLM(
        model=args.base,
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
        max_loras=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=0,
    )
    params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    jobs = ([("base", None)] if args.include_base else []) + [(name, adapter) for name, adapter, _ in arms]
    outputs = {}
    for lora_id, (name, adapter) in enumerate(jobs, start=1):
        request = LoRARequest(name, lora_id, adapter) if adapter else None
        for split in ("train", "held"):
            generated = llm.generate(prompts[split], params, lora_request=request)
            outputs[(name, split)] = [list(out.outputs[0].token_ids) for out in generated]

    results = {}
    for name, _ in jobs:
        # every model against its own reference; the base against all of them as a floor
        targets = [arm[0] for arm in arms] if name == "base" else [name]
        for ref_name in targets:
            for split in ("train", "held"):
                key = f"{name} vs {ref_name} CoT [{split}]"
                results[key] = compare(outputs[(name, split)], references[(ref_name, split)])
                v = results[key]
                print(f"{key:<48} prefix mean={v['prefix_mean']:6.1f} median={v['prefix_median']:5.0f} "
                      f"max={v['prefix_max']:5d} ≥32={v['frac_prefix_ge_32']:.3f} | "
                      f"ratio mean={v['ratio_mean']:.3f} ≥0.8={v['frac_ratio_ge_0.8']:.3f}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "extraction.json").write_text(json.dumps(results, indent=2))
    samples = {
        f"{name}|{split}": tokenizer.decode(outputs[(name, split)][0])[:600]
        for name, _ in jobs for split in ("train", "held")
    }
    samples["reference|train"] = tokenizer.decode(references[(arms[-1][0], "train")][0])[:600]
    (out_dir / "samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2))
    print(f"saved {out_dir}/extraction.json")


if __name__ == "__main__":
    main()
