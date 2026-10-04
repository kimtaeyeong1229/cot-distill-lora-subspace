#!/usr/bin/env python3
"""The untrained students under the shared student setting (the "Untrained" rows of Tables 1 and 15).

drivers/run_base_eval.sh evaluates each student without an adapter on the first 200 MATH-500 problems x 2
samples, three times with sampling seeds 0-2 (there is no training seed). Per student this reports the mean
and standard deviation over those three runs of accuracy, mean tokens, truncation rate and the marker
density over all samples, plus the marker density of each problem's first sample (markers_first_sample),
which is how every adapter's markers are counted elsewhere in the paper.

Reads base_eval/<student>/eval_main_s<seed>/base{.json,.samples.jsonl.gz}; writes experiments/base_students.json.
"""

import gzip
import json
import statistics as st

from config import EVAL_SAMPLES, EXP, REFLECTIVE, RESULTS, SEEDS, STUDENT_MODEL

KEYS = ("accuracy", "avg_tokens", "truncated_rate", "reflective_per_1k_words")


def first_sample_density(path):
    with gzip.open(path, "rt") as f:
        texts = [json.loads(line)["text"] for i, line in enumerate(f) if i % EVAL_SAMPLES == 0]
    return 1000 * sum(len(REFLECTIVE.findall(t)) for t in texts) / max(sum(len(t.split()) for t in texts), 1)


def main():
    out = {}
    for student in STUDENT_MODEL.values():
        runs = [EXP / "base_eval" / student / f"eval_main_s{s}" for s in SEEDS]
        summaries = [json.loads((d / "base.json").read_text()) for d in runs]
        values = {k: [s[k] for s in summaries] for k in KEYS}
        values["markers_first_sample"] = [first_sample_density(d / "base.samples.jsonl.gz") for d in runs]
        out[student] = {k: {"mean": st.mean(v), "sd": st.stdev(v)} for k, v in values.items()}
        r = out[student]
        print(f"{student:24s} acc {100 * r['accuracy']['mean']:5.1f}  tokens {r['avg_tokens']['mean']:5.0f}"
              f"  markers {r['markers_first_sample']['mean']:.2f}")
    (RESULTS / "base_students.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
