#!/usr/bin/env python3
import json
from pathlib import Path
from config import CELL_DIR, CELL_NAMES, EVAL_PROBLEMS as PROBLEMS, EVAL_SAMPLES as SAMPLES, EXP as E
from config import HEAD_WORDS as HEAD, LAMBDAS, MATH500_EVAL, REFLECTIVE, RESULTS, SEEDS

CELLS = {cell: (CELL_DIR[cell], MATH500_EVAL[cell]) for cell in CELL_NAMES}


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--output", default=str(RESULTS / "main_density.json"))
    a = ap.parse_args()
    out = {}
    for cell, (tag, bench) in CELLS.items():
        row = {}
        for lam in LAMBDAS:
            texts, toks, trunc, correct = ([], [], [], [])
            for seed in a.seeds:
                path = E / tag / bench / f"lambda{lam}_seed{seed}.json"
                if not path.exists():
                    continue
                for rec in json.loads(path.read_text())["records"][:PROBLEMS]:
                    texts.append(rec["response_0"])
                    toks += rec["tokens"][:SAMPLES]
                    trunc += rec["truncated"][:SAMPLES]
                    correct += rec["correct"][:SAMPLES]
            if not texts:
                continue
            words = [t.split() for t in texts]
            n_words = sum(map(len, words))
            head = [w[:HEAD] for w in words]
            row[lam] = {
                "responses": len(texts),
                "markers_per_1k": sum((len(REFLECTIVE.findall(t)) for t in texts)) / max(n_words, 1) * 1000,
                "head_markers_per_1k": sum((len(REFLECTIVE.findall(" ".join(h))) for h in head))
                / max(sum(map(len, head)), 1)
                * 1000,
                "avg_tokens": sum(toks) / len(toks),
                "truncated": sum(trunc) / len(trunc),
                "accuracy": sum(map(bool, correct)) / len(correct),
                "accuracy_completed": sum((bool(c) for c, t in zip(correct, trunc) if not t))
                / max(sum((not t for t in trunc)), 1),
            }
        out[cell] = row
        print(
            f"{cell:8}"
            + "".join(
                (
                    f" {row[l]['markers_per_1k']:6.2f}({row[l]['responses']})" if l in row else "     --    "
                    for l in LAMBDAS
                )
            )
        )
    Path(a.output).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
