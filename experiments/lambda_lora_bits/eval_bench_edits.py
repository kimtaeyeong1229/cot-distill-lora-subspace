#!/usr/bin/env python3
"""The section-5 removal outside MATH: GPQA-Diamond (science, multiple choice) and GSM8K.

The adapters are the MATH-trained ones; only the evaluation set changes. Prompts, answer extraction and
grading are MI-Distillation's (evaluation/evaluate.py), the decoding is the shared student setting:
first 200 items x 2 samples, temperature 0.6, top-p 0.95, sampling seed 0, 7,680 new tokens (OLMo
3,840), capped where a long prompt would not fit the context. Writes the same summary and samples
files as eval_ablation.py, with the reflective-marker density of each item's first sample.

Grading (extraction included) runs in a forked child that is killed after its timeout, as in
common.is_correct, so a pathological response cannot stall the run; a killed grade counts as wrong.
"""

import argparse
import gzip
import json
import os
import select
import signal
import sys
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "MI-Distillation" / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import (DATASET_LOADERS, DEFAULT_PATHS, extract_prediction, format_prompt,  # noqa: E402
                      normalize_prediction_for_answer, score_prediction)
from common import load_tokenizer  # noqa: E402
from config import EXP, REFLECTIVE  # noqa: E402

DATA = EXP / "benchmarks"


def grade(bench, question, response, answer, timeout=10):
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            pred = extract_prediction(question, response, dataset=bench)
            pred = normalize_prediction_for_answer(pred, answer, dataset=bench)
            ok, _ = score_prediction(bench, pred, answer, grader_timeout=5.0)
        except BaseException:
            ok = False
        os.write(w, b"1" if ok else b"0")
        os._exit(0)
    os.close(w)
    try:
        ready, _, _ = select.select([r], [], [], timeout)
        if ready:
            return os.read(r, 1) == b"1"
        os.kill(pid, signal.SIGKILL)
        return False
    finally:
        os.close(r)
        os.waitpid(pid, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapters", nargs="+", required=True, help="name=adapter_dir pairs")
    ap.add_argument("--bench", choices=["gpqa_diamond", "gsm8k"], required=True)
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
    jobs = [(n, p) for n, _, p in (s.partition("=") for s in a.adapters) if not (out_dir / f"{n}.json").exists()]
    if not jobs:
        print("모두 완료됨")
        return
    items = DATASET_LOADERS[a.bench](str(DATA / DEFAULT_PATHS[a.bench].replace("data/raw/", "")))[:a.limit]
    tokenizer = load_tokenizer(a.base)
    prompts = [format_prompt(it["question"], tokenizer, dataset=a.bench) for it in items]
    longest = max(len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts)
    max_tokens = min(a.max_new_tokens, a.max_model_len - longest)
    llm = LLM(model=a.base, enable_lora=True, max_loras=1, max_lora_rank=32, max_model_len=a.max_model_len,
              gpu_memory_utilization=a.gpu_memory_utilization, seed=a.seed)
    params = SamplingParams(n=a.n_samples, temperature=0.6, top_p=0.95, top_k=-1, max_tokens=max_tokens, seed=a.seed)

    for lora_id, (name, path) in enumerate(jobs, start=1):
        outputs = llm.generate(prompts, params, lora_request=LoRARequest(name, lora_id, path))
        records, correct, tokens, truncated, first = [], 0, 0, 0, []
        for it, out in zip(items, outputs):
            for k, c in enumerate(out.outputs):
                ok = grade(a.bench, it["question"], c.text, it["answer"])
                records.append({"question": it["question"], "text": c.text, "finish_reason": c.finish_reason,
                                "tokens": len(c.token_ids), "correct": ok})
                correct += ok
                tokens += len(c.token_ids)
                truncated += c.finish_reason == "length"
                if k == 0:
                    first.append(c.text)
        total = len(records)
        summary = {"name": name, "adapter": path, "bench": a.bench, "items": len(items), "samples": total,
                   "max_new_tokens": max_tokens, "accuracy": correct / total, "avg_tokens": tokens / total,
                   "truncated_rate": truncated / total,
                   "reflective_per_1k_words": 1000 * sum(len(REFLECTIVE.findall(t)) for t in first)
                   / max(sum(len(t.split()) for t in first), 1)}
        with gzip.open(out_dir / f"{name}.samples.jsonl.gz", "wt") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        (out_dir / f"{name}.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
