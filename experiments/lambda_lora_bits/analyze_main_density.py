#!/usr/bin/env python3
"""The unedited students under the shared student setting: Tables 1 and 2 and every section-5 base.

Every student number in the paper uses the first 200 MATH-500 problems x 2 samples x seeds 0-2
(7,680 new tokens; OLMo 3,840). The main MATH-500 evaluation ran 500 problems x 4 samples, so it is cut
down here to that setting: accuracy, length and truncation from the first two samples of the first 200
problems, and the reflective-marker density from the first sample, the only one whose text was kept
(section-5 densities are counted on the first sample too, so both sides see the same responses). This
one number per (cell, lambda) is what Table 1 reports and what every section-5 table uses as its
lambda0 / lambda1 base, so the same adapter never carries two different values. The first-300-word
density is the base for Table 8.

Pair A Qwen and Llama are read from eval_math500_7680, their re-run at the shared generation limit.
Writes experiments/main_density.json.
"""

import json
from pathlib import Path

from config import CELL_DIR, CELL_NAMES, EVAL_PROBLEMS as PROBLEMS, EVAL_SAMPLES as SAMPLES, EXP as E
from config import HEAD_WORDS as HEAD, LAMBDAS, MATH500_EVAL, REFLECTIVE, RESULTS, SEEDS

CELLS = {cell: (CELL_DIR[cell], MATH500_EVAL[cell]) for cell in CELL_NAMES}   # cell -> (directory, evaluation)


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
            texts, toks, trunc, correct = [], [], [], []
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
                "markers_per_1k": sum(len(REFLECTIVE.findall(t)) for t in texts) / max(n_words, 1) * 1000,
                "head_markers_per_1k": sum(len(REFLECTIVE.findall(" ".join(h))) for h in head)
                                       / max(sum(map(len, head)), 1) * 1000,
                "avg_tokens": sum(toks) / len(toks), "truncated": sum(trunc) / len(trunc),
                "accuracy": sum(map(bool, correct)) / len(correct),
                "accuracy_completed": (sum(bool(c) for c, t in zip(correct, trunc) if not t)
                                       / max(sum(not t for t in trunc), 1))}
        out[cell] = row
        print(f"{cell:8}" + "".join(f" {row[l]['markers_per_1k']:6.2f}({row[l]['responses']})" if l in row else "     --    " for l in LAMBDAS))
    Path(a.output).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
