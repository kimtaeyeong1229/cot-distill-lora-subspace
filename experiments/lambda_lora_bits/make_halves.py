#!/usr/bin/env python3
"""Split every lambda's training set into two disjoint halves for the data-only baseline B(lambda).

halfA takes lines 0, 2, 4, ... of train.jsonl and halfB lines 1, 3, 5, ...; train.jsonl lists the same
problems in the same order at every lambda, so both halves hold the same problems at every lambda.
This reproduces the shipped train_half{A,B}.jsonl files byte for byte (checked on all 18 pair/lambda
directories). run_halves.sh trains one adapter per half.

  python experiments/lambda_lora_bits/make_halves.py experiments/lambda_lora_bits/pairB/splits
"""

import argparse
from pathlib import Path

LAMBDAS = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("splits", help="a pair's splits directory, holding lambda<l>/train.jsonl")
    ap.add_argument("--lambdas", nargs="+", default=LAMBDAS)
    a = ap.parse_args()
    for lam in a.lambdas:
        d = Path(a.splits) / f"lambda{lam}"
        lines = (d / "train.jsonl").read_bytes().splitlines(keepends=True)
        (d / "train_halfA.jsonl").write_bytes(b"".join(lines[0::2]))
        (d / "train_halfB.jsonl").write_bytes(b"".join(lines[1::2]))
        print(f"{d}: {len(lines)} -> {len(lines[0::2])} + {len(lines[1::2])}")


if __name__ == "__main__":
    main()
