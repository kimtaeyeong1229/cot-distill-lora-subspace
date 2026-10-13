#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from tqdm import tqdm
from common import clean_response, encode, is_correct, label_count, load_tokenizer, read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gen_dir", required=True)
    parser.add_argument("--pattern", default="lambda{lam}.jsonl", help="File name pattern inside gen_dir.")
    parser.add_argument("--lambdas", default="0.0 0.2 0.4 0.6 0.8 1.0")
    parser.add_argument("--student", required=True, help="Student tokenizer used for length filtering.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_held", type=int, default=300)
    parser.add_argument("--n_train", type=int, default=None, help="Default: every remaining shared problem.")
    parser.add_argument("--max_len", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report_only", action="store_true", help="Print per-lambda stats without writing splits."
    )
    args = parser.parse_args()
    lambdas = args.lambdas.split()
    tokenizer = load_tokenizer(args.student)
    kept, stats = ({}, {})
    for lam in lambdas:
        rows = read_jsonl(Path(args.gen_dir) / args.pattern.format(lam=lam))
        good, truncated, wrong, too_long = ({}, 0, 0, 0)
        for row in tqdm(rows, desc=f"lambda={lam}"):
            if row["finish_reason"] != "stop":
                truncated += 1
                continue
            response = clean_response(row["cot_response"])
            if not is_correct(row["problem"], response, row["answer"]):
                wrong += 1
                continue
            input_ids, labels = encode(tokenizer, row["problem"], response, max_len=10**9)
            if len(input_ids) >= args.max_len:
                too_long += 1
                continue
            good[row["unique_id"]] = {
                "unique_id": row["unique_id"],
                "problem": row["problem"],
                "answer": row["answer"],
                "level": row.get("level", ""),
                "response": response,
                "n_tokens": label_count(labels),
                "teacher_tokens": row["generated_tokens"],
            }
        kept[lam] = good
        n = len(rows)
        stats[lam] = {
            "generated": n,
            "teacher_accuracy": (n - truncated - wrong) / n,
            "truncated_rate": truncated / n,
            "too_long_for_student": too_long,
            "kept": len(good),
            "avg_teacher_tokens_all": sum((r["generated_tokens"] for r in rows)) / n,
            "avg_student_tokens_kept": sum((g["n_tokens"] for g in good.values())) / max(1, len(good)),
        }
        print(json.dumps({lam: stats[lam]}))
    shared = sorted(set.intersection(*(set(k) for k in kept.values())))
    print(f"Shared problems across {len(lambdas)} lambdas: {len(shared)}")
    if args.report_only:
        return
    rng = random.Random(args.seed)
    rng.shuffle(shared)
    held_ids = shared[: args.n_held]
    train_ids = (
        shared[args.n_held :] if args.n_train is None else shared[args.n_held : args.n_held + args.n_train]
    )
    if len(held_ids) < args.n_held or not train_ids:
        raise ValueError(f"Not enough shared problems ({len(shared)}) for n_held={args.n_held}.")
    out_dir = Path(args.out_dir)
    for lam in lambdas:
        write_jsonl(out_dir / f"lambda{lam}" / "train.jsonl", [kept[lam][i] for i in train_ids])
        write_jsonl(out_dir / f"lambda{lam}" / "held.jsonl", [kept[lam][i] for i in held_ids])
        stats[lam]["train_tokens"] = sum((kept[lam][i]["n_tokens"] for i in train_ids))
        stats[lam]["held_tokens"] = sum((kept[lam][i]["n_tokens"] for i in held_ids))
    summary = {"shared": len(shared), "n_train": len(train_ids), "n_held": len(held_ids), "per_lambda": stats}
    (out_dir / "split_stats.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "split_ids.json").write_text(json.dumps({"train": train_ids, "held": held_ids}))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
