#!/usr/bin/env python3
import json
from pathlib import Path
from config import EXP, LAMBDAS, PAIR_DIR, PAIR_KEY, RESULTS

PAIRS = [(PAIR_KEY[p], PAIR_DIR[p]) for p in PAIR_DIR]


def main():
    print(f"{'pair':4s} " + "".join((f"{'λ' + l:>8s}" for l in LAMBDAS)) + "   first 0%")
    out = {}
    for name, tag in PAIRS:
        rates, first_zero = ({}, None)
        for lam in LAMBDAS:
            path = EXP / tag / "cot" / f"lambda{lam}.jsonl"
            rows = [json.loads(line) for line in path.open()]
            rate = sum(("</think>" in r["cot_response"] for r in rows)) / len(rows)
            rates[lam] = rate
            if rate < 0.005 and first_zero is None:
                first_zero = lam
        print(f"{name:4s} " + "".join((f"{rates[l]:7.0%} " for l in LAMBDAS)) + f"   λ{first_zero}")
        out[name] = dict(
            rates=rates,
            first_zero=first_zero,
            n=len([1 for _ in (EXP / tag / "cot" / "lambda0.0.jsonl").open()]),
        )
    path = RESULTS / "think_rate.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\n{path}")


if __name__ == "__main__":
    main()
