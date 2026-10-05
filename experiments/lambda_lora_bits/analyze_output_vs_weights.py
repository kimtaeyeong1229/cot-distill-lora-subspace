#!/usr/bin/env python3
"""Does the adapter's direction change simply mirror the student's output change?

For every cell this correlates, across the five lambda intervals, the teacher's drop (in </think> rate
and in reflective-marker density) with three student-side drops: R (phi_resample_s012.json), the
student's mean response length and its reflective-marker density (main_density.json). Pair A is the
only pair whose teacher signals come apart -- markers fall from lambda 0.2, the tag only at 0.4 -- so
it is where the adapter and the outputs can be told apart. Table 4 reports the pair means.

Writes experiments/output_vs_weights.json.
"""

import json
import statistics as st
from config import LAMBDAS, RESULTS

STUDENTS = {"Qwen": "Qwen2.5-0.5B", "Llama": "Llama-3.2-1B", "OLMo": "OLMo-2-1B"}
DIR = {"A": "A", "B": "B", "C": "D"}  # the adapter and think-rate files keep pair C's directory name


def drops(values):
    return [values[i] - values[i + 1] for i in range(len(values) - 1)]


def corr(a, b):
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return num / (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5


def main():
    exp = RESULTS
    density = json.loads((exp / "main_density.json").read_text())
    resample = json.loads((exp / "phi_resample_s012.json").read_text())
    think = json.loads((exp / "think_rate.json").read_text())
    teacher = json.loads((exp / "teacher_density.json").read_text())

    out = {"cells": {}, "pairs": {}}
    for pair, d in DIR.items():
        teacher_drop = {"think": drops([think[d]["rates"][l] for l in LAMBDAS]),
                        "markers": drops([teacher[pair]["teacher"][l] for l in LAMBDAS])}
        rows = []
        for short, full in STUDENTS.items():
            cell = density[f"{pair}/{short}"]
            student_drop = {"R": drops([resample[f"{d}/{full}"]["rows"][l]["retention"] for l in LAMBDAS]),
                            "tokens": drops([cell[l]["avg_tokens"] for l in LAMBDAS]),
                            "markers": drops([cell[l]["markers_per_1k"] for l in LAMBDAS])}
            row = {f"{s}_vs_{t}": corr(sv, tv) for s, sv in student_drop.items() for t, tv in teacher_drop.items()}
            out["cells"][f"{pair}/{short}"] = row
            rows.append(row)
        out["pairs"][pair] = {k: st.mean(r[k] for r in rows) for k in rows[0]}

    for pair, row in out["pairs"].items():
        print(pair, "  ".join(f"{k} {v:.2f}" for k, v in row.items()))
    (exp / "output_vs_weights.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
