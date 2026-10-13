#!/usr/bin/env python3
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
