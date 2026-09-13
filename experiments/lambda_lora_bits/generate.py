#!/usr/bin/env python3
"""Sample one CoT per problem from a (possibly interpolated) teacher with vLLM.

The prompt uses the chat template of the tokenizer stored in the teacher
directory. Output is JSONL keyed by unique_id and appended per chunk, so an
interrupted run resumes where it stopped.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from common import STOP_TOKEN_IDS, build_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--problems", required=True, help="Parquet with unique_id, problem, answer, level, type.")
    parser.add_argument("--output", required=True, help="JSONL output path.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--max_model_len", type=int, default=9216)
    parser.add_argument("--max_num_seqs", type=int, default=256)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=1000)
    args = parser.parse_args()

    problems = pd.read_parquet(args.problems)
    if args.limit:
        problems = problems.head(args.limit)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output.exists():
        with output.open(encoding="utf-8") as handle:
            done = {json.loads(line)["unique_id"] for line in handle if line.strip()}
    pending = problems[~problems["unique_id"].isin(done)].to_dict("records")
    print(f"{args.model}: {len(done)} done, {len(pending)} pending")
    if not pending:
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )
    params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
        stop_token_ids=STOP_TOKEN_IDS,
        skip_special_tokens=False,
    )
    print("Example prompt:\n" + build_prompt(tokenizer, pending[0]["problem"]))

    for start in range(0, len(pending), args.chunk):
        batch = pending[start:start + args.chunk]
        prompts = [
            {"prompt_token_ids": tokenizer(build_prompt(tokenizer, row["problem"]), add_special_tokens=False)["input_ids"]}
            for row in batch
        ]
        began = time.time()
        outputs = llm.generate(prompts, params)
        elapsed = time.time() - began
        n_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)

        with output.open("a", encoding="utf-8") as handle:
            for row, out in zip(batch, outputs):
                completion = out.outputs[0]
                record = {
                    "unique_id": int(row["unique_id"]),
                    "problem": row["problem"],
                    "answer": row["answer"],
                    "level": row.get("level", ""),
                    "type": row.get("type", ""),
                    "cot_response": completion.text,
                    "generated_tokens": len(completion.token_ids),
                    "finish_reason": completion.finish_reason,
                    "teacher": args.model,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"[{start + len(batch)}/{len(pending)}] {n_tokens / len(batch):.0f} tok/sample, "
            f"{n_tokens / elapsed:.0f} tok/s, {elapsed / 60:.1f} min"
        )


if __name__ == "__main__":
    main()
