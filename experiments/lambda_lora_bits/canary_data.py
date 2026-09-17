#!/usr/bin/env python3
"""Build canary splits: random strings that can only be memorized, mixed into a real training set.

Each canary pairs a unique lookup prompt with a uniformly random string over [a-z0-9]. Nothing in the
string is predictable from the prompt, so the only information a student can carry from training
canaries to held-out canaries is the format (alphabet, length, EOS). A sound memorization measure
should put canary bits written beyond that format on the memorized side (Morris et al. 2026;
Tan et al. 2026 use random data as ground truth in the same way).
"""

import argparse
import json
import math
import random
import string
from pathlib import Path

ALPHABET = string.ascii_lowercase + string.digits


def canaries(rng, ids, length):
    return [{"unique_id": f"canary-{i}", "problem": f"What is the access code stored under record #{i}?",
             "answer": "", "level": "canary", "response": "".join(rng.choice(ALPHABET) for _ in range(length))}
            for i in ids]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--splits", required=True, help="Existing lambda split dir with train.jsonl and held.jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=100, help="canaries per side (train and held)")
    p.add_argument("--length", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeat", type=int, default=1, help="copies of each training canary")
    a = p.parse_args()
    rng = random.Random(a.seed)
    ids = rng.sample(range(100000, 1000000), 2 * a.n)
    tr, he = canaries(rng, ids[: a.n], a.length), canaries(rng, ids[a.n:], a.length)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    real = Path(a.splits)
    mixed = [json.loads(l) for l in open(real / "train.jsonl")] + tr * a.repeat
    random.Random(a.seed + 1).shuffle(mixed)
    for name, rows in [("train_mixed", mixed), ("canary_train", tr), ("canary_held", he)]:
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    (out / "held.jsonl").write_text(open(real / "held.jsonl").read())
    (out / "train_real.jsonl").write_text(open(real / "train.jsonl").read())
    meta = {"n_per_side": a.n, "length": a.length, "alphabet": len(ALPHABET),
            "entropy_bits_per_canary": a.length * math.log2(len(ALPHABET)), "seed": a.seed, "repeat": a.repeat, "source_splits": str(real)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
