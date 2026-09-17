#!/usr/bin/env python3
"""Generate answers for MI-Distillation's seven benchmarks with one vLLM engine per student.

Loaders, prompt formatting and decoding settings come from MI-Distillation's
evaluation/evaluate.py. The repeats their shell script runs as separate
invocations (16 for AIME24/AMC23, 4 otherwise) are drawn here as n samples per
prompt from a single engine, so the base model and every adapter share one
start-up. Raw generations are written gzipped for bench_score.py.
"""

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "MI-Distillation" / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import DATASET_LOADERS, DEFAULT_PATHS, format_prompt  # noqa: E402
from common import load_tokenizer  # noqa: E402

BENCHMARKS = ["math500", "gsm8k", "amc23", "aime24", "minervamath", "olympiadbench", "gpqa_diamond"]
SMALL = {"aime24", "amc23"}  # MI-Distillation repeats these 16 times, the rest 4 times
CONDITIONS = ["base"] + [f"lambda{lam}_seed0" for lam in ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--student_dir", required=True, help="Directory holding runs/<condition>/adapter.")
    parser.add_argument("--benchmarks", nargs="+", default=BENCHMARKS)
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    parser.add_argument("--data_root", default=str(Path(__file__).resolve().parent / "benchmarks"))
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Items per benchmark (smoke tests only).")
    parser.add_argument("--raw_dir", default="bench_raw")
    args = parser.parse_args()

    student_dir = Path(args.student_dir)
    raw_root = student_dir / args.raw_dir
    jobs = []
    for bench in args.benchmarks:
        for cond in args.conditions:
            out = raw_root / bench / f"{cond}.json.gz"
            adapter = None if cond == "base" else student_dir / "runs" / cond / "adapter"
            if out.exists():
                continue
            if adapter is not None and not (adapter / "adapter_model.safetensors").exists():
                print(f"[skip] missing adapter {adapter}")
                continue
            jobs.append((bench, cond, adapter, out))
    if not jobs:
        print("nothing to generate")
        return

    tokenizer = load_tokenizer(args.base)
    llm = LLM(model=args.base, dtype="bfloat16", enable_lora=True, max_lora_rank=16, max_loras=1,
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization, seed=args.seed)
    lora_ids = {}

    data = {}
    for bench, cond, adapter, out in jobs:
        if bench not in data:
            path = Path(args.data_root) / DEFAULT_PATHS[bench].replace("data/raw/", "")
            items = DATASET_LOADERS[bench](str(path))
            data[bench] = items[: args.limit] if args.limit else items
        items = data[bench]
        runs = 16 if bench in SMALL else 4
        prompts = [format_prompt(it["question"], tokenizer, dataset=bench) for it in items]
        # Short-context students (OLMo-2, 4096) cannot fit prompt + max_new_tokens; cap to what fits.
        longest = max(len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts)
        max_tokens = min(args.max_new_tokens, args.max_model_len - longest)
        params = SamplingParams(n=runs, temperature=0.6, top_p=0.95, top_k=-1, repetition_penalty=1.0,
                                max_tokens=max_tokens, seed=args.seed)
        request = None
        if adapter is not None:
            lora_ids.setdefault(str(adapter), len(lora_ids) + 1)
            request = LoRARequest(cond, lora_ids[str(adapter)], str(adapter))

        began = time.time()
        outputs = llm.generate(prompts, params, lora_request=request)
        records = []
        n_tokens = 0
        for it, o in zip(items, outputs):
            toks = [len(c.token_ids) for c in o.outputs]
            n_tokens += sum(toks)
            records.append({"question": it["question"], "answer": it["answer"], "groups": it.get("groups", {}),
                            "responses": [c.text for c in o.outputs], "tokens": toks,
                            "finish": [c.finish_reason for c in o.outputs]})
        out.parent.mkdir(parents=True, exist_ok=True)
        meta = {"benchmark": bench, "condition": cond, "base": args.base, "adapter": str(adapter) if adapter else None,
                "runs": runs, "temperature": 0.6, "top_p": 0.95, "max_new_tokens": max_tokens,
                "max_model_len": args.max_model_len, "seed": args.seed, "items": len(items)}
        with gzip.open(out, "wt", encoding="utf-8") as fh:
            json.dump({"meta": meta, "records": records}, fh, ensure_ascii=False)
        elapsed = time.time() - began
        print(f"[gen] {bench:<14} {cond:<16} items={len(items):5d} x{runs:<2d} "
              f"avg_tokens={n_tokens / (len(items) * runs):6.0f} {n_tokens / elapsed:6.0f} tok/s {elapsed / 60:5.1f} min", flush=True)


if __name__ == "__main__":
    main()
