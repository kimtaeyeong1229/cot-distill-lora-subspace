#!/usr/bin/env python3
import argparse
import gzip
import json
from pathlib import Path
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from common import build_prompt, is_correct, load_tokenizer, read_jsonl
from config import REFLECTIVE

NOTHINK = "Okay, I think I have finished thinking.\n\n"
FORCE = "\n\n**Final Answer**\n\\[\n\\boxed{"
FORCE_TOKENS = 64


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--mode", choices=["nothink", "budget"], required=True)
    ap.add_argument("--budget", type=int, help="new-token budget for --mode budget")
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--n_samples", type=int, default=2)
    ap.add_argument("--max_new_tokens", type=int, default=7680)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / f"{a.name}.json").exists():
        print("all conditions already done")
        return
    rows = pd.read_parquet(a.data).to_dict("records") if a.data.endswith(".parquet") else read_jsonl(a.data)
    problems = rows[: a.limit]
    tokenizer = load_tokenizer(a.base)
    prompts = [build_prompt(tokenizer, r["problem"]) for r in problems]
    if a.mode == "nothink":
        prompts = [p + NOTHINK for p in prompts]
    llm = LLM(
        model=a.base,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=32,
        max_model_len=a.max_model_len,
        gpu_memory_utilization=a.gpu_memory_utilization,
        seed=a.seed,
    )
    limit = a.budget if a.mode == "budget" else a.max_new_tokens
    params = SamplingParams(
        n=a.n_samples, temperature=0.6, top_p=0.95, top_k=-1, max_tokens=limit, seed=a.seed
    )
    lora = LoRARequest(a.name, 1, a.adapter)
    outputs = llm.generate(prompts, params, lora_request=lora)
    samples = []
    for row, prompt, out in zip(problems, prompts, outputs):
        for c in out.outputs:
            samples.append([row, prompt, c.text, c.finish_reason, len(c.token_ids)])
    forced = 0
    if a.mode == "budget":
        cut = [i for i, s in enumerate(samples) if s[3] == "length"]
        if cut:
            force = SamplingParams(
                n=1,
                temperature=0.6,
                top_p=0.95,
                top_k=-1,
                max_tokens=FORCE_TOKENS,
                stop=["\\]"],
                include_stop_str_in_output=True,
                seed=a.seed,
            )
            outs = llm.generate(
                [samples[i][1] + samples[i][2] + FORCE for i in cut], force, lora_request=lora
            )
            for i, o in zip(cut, outs):
                c = o.outputs[0]
                samples[i][2] += FORCE + c.text
                samples[i][3] = "forced"
                samples[i][4] += len(tokenizer(FORCE, add_special_tokens=False)["input_ids"]) + len(
                    c.token_ids
                )
            forced = len(cut)
    correct = tokens = truncated = markers = words = tagged = 0
    records = []
    for row, _, text, reason, n in samples:
        records.append({"problem": row["problem"], "text": text, "finish_reason": reason, "tokens": n})
        correct += is_correct(row["problem"], text, str(row["answer"]))
        tokens += n
        truncated += reason == "length"
        markers += len(REFLECTIVE.findall(text))
        words += len(text.split())
        tagged += "</think>" in text
    total = len(samples)
    summary = {
        "name": a.name,
        "adapter": a.adapter,
        "mode": a.mode,
        "budget": a.budget,
        "problems": len(problems),
        "samples": total,
        "accuracy": correct / total,
        "avg_tokens": tokens / total,
        "truncated_rate": truncated / total,
        "forced_rate": forced / total,
        "reflective_per_1k_words": markers / max(words, 1) * 1000,
        "think_tag_rate": tagged / total,
    }
    with gzip.open(out_dir / f"{a.name}.samples.jsonl.gz", "wt") as f:
        f.writelines((json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    (out_dir / f"{a.name}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
