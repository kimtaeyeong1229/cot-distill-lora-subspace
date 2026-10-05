#!/usr/bin/env python3
"""Reflective-marker density of the teachers' own solutions, next to the students that learned from them.

The students' marker density is counted with the REFLECTIVE pattern of analyze_main_density.py; the same
pattern is run here on the raw teacher generations (cot/lambda*.jsonl, field cot_response, the 4,000
problems the training data were drawn from), so teacher and student are measured with one instrument.
It also reports, per cell, the correlation across lambda between the teacher's density and the
student's (Table 1, seed means from experiments/seed_spread.json).

Pair A's teacher still closes </think> in 91% of its answers at lambda 0.2, yet its reflective density
has already fallen there; this is what the think rate alone cannot show.

Writes experiments/teacher_density.json.
"""

import json
import statistics as st
from pathlib import Path

from config import EXP as E, LAMBDAS, PAIR_DIR as PAIRS, REFLECTIVE, RESULTS, STUDENTS


def main():
    spread = json.loads((RESULTS / "seed_spread.json").read_text())
    out = {}
    pooled_t, pooled_s = [], []
    for pair, tag in PAIRS.items():
        density = []
        for lam in LAMBDAS:
            markers = words = 0
            with open(E / tag / "cot" / f"lambda{lam}.jsonl") as f:
                for line in f:
                    text = json.loads(line)["cot_response"]
                    markers += len(REFLECTIVE.findall(text))
                    words += len(text.split())
            density.append(1000 * markers / max(words, 1))
        row = {"teacher": dict(zip(LAMBDAS, density)), "corr": {}}
        for s in STUDENTS:
            student = [spread[f"{pair}/{s}"]["table1"][lam]["mean"] for lam in LAMBDAS]
            row["corr"][s] = st.correlation(density, student)
            pooled_t += density
            pooled_s += student
        out[pair] = row
        print(pair, " ".join(f"{d:5.1f}" for d in density), "| r", " ".join(f"{v:.2f}" for v in row["corr"].values()),
              flush=True)
    out["pooled_corr"] = st.correlation(pooled_t, pooled_s)
    print(f"pooled r = {out['pooled_corr']:.3f}")
    (RESULTS / "teacher_density.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
