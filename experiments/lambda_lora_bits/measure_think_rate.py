#!/usr/bin/env python3
"""Fraction of a teacher's generations that close the reasoning block, per pair and lambda.

The interpolated teachers are built from a reasoning checkpoint and an instruct checkpoint, and the
reasoning one's chat template opens <think> for it, so the model itself only ever emits the closing
</think>. Counting that tag in the raw generations (cot/lambda*.jsonl, field cot_response) is therefore
the cleanest read of whether a given teacher is still producing the reasoning format at all.

The lambda at which this reaches zero is the prediction the adapter analyses are tested against, so it
is measured here rather than carried around as a constant.
"""

import json
from pathlib import Path

from config import EXP, LAMBDAS, PAIR_DIR, PAIR_KEY, RESULTS

PAIRS = [(PAIR_KEY[p], PAIR_DIR[p]) for p in PAIR_DIR]   # (key in think_rate.json, directory); pair C is "D"


def main():
    print(f"{'pair':4s} " + "".join(f"{('λ' + l):>8s}" for l in LAMBDAS) + "   first 0%")
    out = {}
    for name, tag in PAIRS:
        rates, first_zero = {}, None
        for lam in LAMBDAS:
            path = EXP / tag / "cot" / f"lambda{lam}.jsonl"
            rows = [json.loads(line) for line in path.open()]
            rate = sum("</think>" in r["cot_response"] for r in rows) / len(rows)
            rates[lam] = rate
            if rate < 0.005 and first_zero is None:
                first_zero = lam
        print(f"{name:4s} " + "".join(f"{rates[l]:7.0%} " for l in LAMBDAS) + f"   λ{first_zero}")
        out[name] = dict(rates=rates, first_zero=first_zero,
                         n=len([1 for _ in (EXP / tag / "cot" / "lambda0.0.jsonl").open()]))
    path = RESULTS / "think_rate.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\n{path}")


if __name__ == "__main__":
    main()
