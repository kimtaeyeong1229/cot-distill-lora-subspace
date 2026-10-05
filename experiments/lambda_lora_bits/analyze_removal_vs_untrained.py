#!/usr/bin/env python3
"""Does region removal return the student to its untrained state? Per-problem comparison (read only).

Every model below answered the same first 200 MATH-500 problems, 2 samples per problem, under three seeds,
at the shared generation limit (7,680 new tokens; OLMo 3,840):

  untrained   the untrained instruct student (base_eval/<model>/eval_main_s<seed>/base.samples.jsonl.gz;
              these runs are the source of experiments/base_students.json)
  rm_pre      the lambda = 0 adapter with the shared region removed (<cell>/eval_main_s<seed>/lambda0.0_rm_pre)
  lambda0     the unedited lambda = 0 adapter          (the main_density.json records, Table 1)
  lambda1     the unedited lambda = 1 adapter          (same, the short-answer student)

Grading: untrained and rm_pre responses are regraded with common.is_correct. For lambda0 / lambda1 only the
first sample's text was kept, so both samples use the correctness stored by the evaluation (the same grader);
the first sample is regraded to check that the two agree (stored_vs_regraded).

Pairing: response (problem p, seed s, sample k) of a model is paired with response (p, s, k) of the untrained
student; the untrained seeds are only sampling seeds, so the pairing is arbitrary but fixed. Per cell:
  accuracy, tokens (mean over all responses), completed (share stopping before the limit),
  markers_per_1k (REFLECTIVE on the first sample of each problem, as in Tables 1 and 6-9),
  opens_okay (share of first samples whose first word is "Okay", the teachers' monologue opening),
  response level (1,200 pairs): agreement (both right or both wrong), cohen_kappa, exactly_one (share of
    pairs where exactly one is right = 1 - agreement), split into only_model / only_untrained;
  problem level (200 problems; a problem is "solved" when at least half of its 6 responses are right):
    agreement, cohen_kappa, exactly_one, and the Pearson r of the per-problem accuracies.
The sampling-noise ceiling is the untrained student against itself (seed s vs seed s+1 mod 3, response level).

Distances to the untrained student (|accuracy difference| in pp, |log2 token ratio|, |marker difference|,
kappa) are compared between rm_pre and lambda1, the question being whether removal lands nearer the untrained
student than the short-answer adapter does.

Run from the repo root; writes experiments/removal_vs_untrained.json.
"""

import gzip
import json
import math
import statistics as st
import sys
from pathlib import Path

import pandas as pd

E = Path("experiments/lambda_lora_bits")
sys.path.insert(0, str(E))
from common import is_correct  # noqa: E402
from analyze_main_density import CELLS as BASE_CELLS, REFLECTIVE  # noqa: E402

CELLS = {c: tag for c, (tag, _) in BASE_CELLS.items()}
MODELS = {"Qwen": "Qwen2.5-0.5B-Instruct", "Llama": "Llama-3.2-1B-Instruct", "OLMo": "OLMo-2-1B-Instruct"}
SEEDS, SAMPLES, PROBLEMS = [0, 1, 2], 2, 200


def density(texts):
    return 1000 * sum(len(REFLECTIVE.findall(t)) for t in texts) / max(sum(len(t.split()) for t in texts), 1)


def opens_okay(texts):
    return sum(t.lstrip().split(" ", 1)[0].rstrip(",.") == "Okay" for t in texts) / len(texts)


def load_samples(path, answers):
    """Generations stored per sample: correctness [p][k], tokens, completion and first-sample texts."""
    rows = [json.loads(line) for line in gzip.open(path, "rt")]
    assert len(rows) == PROBLEMS * SAMPLES, path
    correct = [[is_correct(rows[p * SAMPLES + k]["problem"], rows[p * SAMPLES + k]["text"], answers[p])
                for k in range(SAMPLES)] for p in range(PROBLEMS)]
    return {"correct": correct, "tokens": [r["tokens"] for r in rows],
            "completed": [r["finish_reason"] == "stop" for r in rows],
            "first": [rows[p * SAMPLES]["text"] for p in range(PROBLEMS)], "regrade_check": None}


def load_records(path, problems, answers):
    """Unedited adapters: stored correctness for both samples, first-sample text regraded as a check."""
    recs = json.loads(path.read_text())["records"][:PROBLEMS]
    first = [r["response_0"] for r in recs]
    regraded = [is_correct(problems[p], first[p], answers[p]) for p in range(PROBLEMS)]
    stored = [bool(r["correct"][0]) for r in recs]
    return {"correct": [[bool(c) for c in r["correct"][:SAMPLES]] for r in recs],
            "tokens": [t for r in recs for t in r["tokens"][:SAMPLES]],
            "completed": [not t for r in recs for t in r["truncated"][:SAMPLES]],
            "first": first, "regrade_check": sum(a == b for a, b in zip(stored, regraded)) / PROBLEMS}


def kappa(x, y):
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    px, py = sum(x) / n, sum(y) / n
    pe = px * py + (1 - px) * (1 - py)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def compare(model, ref):
    """model, ref: {seed: loaded}. Response-level and problem-level agreement with the reference."""
    x = [model[s]["correct"][p][k] for s in SEEDS for p in range(PROBLEMS) for k in range(SAMPLES)]
    y = [ref[s]["correct"][p][k] for s in SEEDS for p in range(PROBLEMS) for k in range(SAMPLES)]
    n = len(x)
    resp = {"pairs": n, "agreement": sum(a == b for a, b in zip(x, y)) / n, "cohen_kappa": kappa(x, y),
            "exactly_one": sum(a != b for a, b in zip(x, y)) / n,
            "only_model": sum(a and not b for a, b in zip(x, y)) / n,
            "only_untrained": sum(b and not a for a, b in zip(x, y)) / n}
    pm = [st.mean(float(model[s]["correct"][p][k]) for s in SEEDS for k in range(SAMPLES)) for p in range(PROBLEMS)]
    pr = [st.mean(float(ref[s]["correct"][p][k]) for s in SEEDS for k in range(SAMPLES)) for p in range(PROBLEMS)]
    bx, by = [v >= 0.5 for v in pm], [v >= 0.5 for v in pr]
    prob = {"agreement": sum(a == b for a, b in zip(bx, by)) / PROBLEMS, "cohen_kappa": kappa(bx, by),
            "exactly_one": sum(a != b for a, b in zip(bx, by)) / PROBLEMS,
            "solved_model": sum(bx) / PROBLEMS, "solved_untrained": sum(by) / PROBLEMS,
            "pearson_r_accuracy": st.correlation(pm, pr) if len(set(pm)) > 1 and len(set(pr)) > 1 else float("nan")}
    return {"response_level": resp, "problem_level": prob}


def summary(runs):
    correct = [c for s in SEEDS for p in runs[s]["correct"] for c in p]
    first = [t for s in SEEDS for t in runs[s]["first"]]
    tokens = [t for s in SEEDS for t in runs[s]["tokens"]]
    completed = [c for s in SEEDS for c in runs[s]["completed"]]
    out = {"accuracy": sum(correct) / len(correct), "tokens": st.mean(tokens),
           "completed": sum(completed) / len(completed), "markers_per_1k": density(first), "opens_okay": opens_okay(first),
           "accuracy_per_seed": [sum(c for p in runs[s]["correct"] for c in p) / (PROBLEMS * SAMPLES) for s in SEEDS]}
    if runs[SEEDS[0]]["regrade_check"] is not None:
        out["stored_vs_regraded"] = st.mean(runs[s]["regrade_check"] for s in SEEDS)
    return out


def main():
    rows = pd.read_parquet(E / "data" / "math500.parquet").to_dict("records")[:PROBLEMS]
    problems, answers = [r["problem"] for r in rows], [str(r["answer"]) for r in rows]
    untrained = {}
    for short, name in MODELS.items():
        untrained[short] = {s: load_samples(E / "base_eval" / name / f"eval_main_s{s}" / "base.samples.jsonl.gz", answers)
                            for s in SEEDS}
        assert untrained[short][0]["first"], name
    out = {"untrained": {}, "cells": {}}
    for short in MODELS:
        u = untrained[short]
        shifted = {s: u[SEEDS[(i + 1) % len(SEEDS)]] for i, s in enumerate(SEEDS)}
        out["untrained"][short] = {**summary(u), "self_ceiling": compare(shifted, u)["response_level"]}
        print(short, {k: round(v, 3) for k, v in out["untrained"][short].items() if isinstance(v, float)}, flush=True)
    for cell, (tag, bench) in BASE_CELLS.items():
        short = cell.split("/")[1]
        u = untrained[short]
        conds = {"rm_pre": {s: load_samples(E / tag / f"eval_main_s{s}" / "lambda0.0_rm_pre.samples.jsonl.gz", answers)
                            for s in SEEDS},
                 "lambda0": {s: load_records(E / tag / bench / f"lambda0.0_seed{s}.json", problems, answers) for s in SEEDS},
                 "lambda1": {s: load_records(E / tag / bench / f"lambda1.0_seed{s}.json", problems, answers) for s in SEEDS}}
        ut = out["untrained"][short]
        row = {}
        for c, runs in conds.items():
            r = {**summary(runs), "vs_untrained": compare(runs, u)}
            r["distance"] = {"abs_accuracy_pp": 100 * abs(r["accuracy"] - ut["accuracy"]),
                             "abs_log2_token_ratio": abs(math.log2(r["tokens"] / ut["tokens"])),
                             "abs_markers_per_1k": abs(r["markers_per_1k"] - ut["markers_per_1k"]),
                             "accuracy_minus_untrained_pp": 100 * (r["accuracy"] - ut["accuracy"])}
            row[c] = r
        d0, d1 = row["rm_pre"]["distance"], row["lambda1"]["distance"]
        row["rm_pre_closer_than_lambda1"] = {
            "accuracy": d0["abs_accuracy_pp"] < d1["abs_accuracy_pp"],
            "tokens": d0["abs_log2_token_ratio"] < d1["abs_log2_token_ratio"],
            "markers": d0["abs_markers_per_1k"] < d1["abs_markers_per_1k"],
            "response_kappa": row["rm_pre"]["vs_untrained"]["response_level"]["cohen_kappa"]
                              > row["lambda1"]["vs_untrained"]["response_level"]["cohen_kappa"]}
        out["cells"][cell] = row
        print(f"{cell:8} untrained {100 * ut['accuracy']:5.1f} | " + " | ".join(
            f"{c} {100 * row[c]['accuracy']:5.1f} k={row[c]['vs_untrained']['response_level']['cohen_kappa']:.2f}"
            f" tok {row[c]['tokens']:5.0f} mk {row[c]['markers_per_1k']:5.2f}" for c in conds), flush=True)
    cells = list(out["cells"])
    out["summary"] = {c: {"mean_abs_accuracy_pp": st.mean(out["cells"][k][c]["distance"]["abs_accuracy_pp"] for k in cells),
                          "mean_response_kappa": st.mean(out["cells"][k][c]["vs_untrained"]["response_level"]["cohen_kappa"]
                                                         for k in cells),
                          "mean_abs_log2_token_ratio": st.mean(out["cells"][k][c]["distance"]["abs_log2_token_ratio"]
                                                               for k in cells)}
                      for c in ("rm_pre", "lambda0", "lambda1")}
    out["summary"]["untrained_self_kappa"] = {m: out["untrained"][m]["self_ceiling"]["cohen_kappa"] for m in MODELS}
    out["summary"]["rm_pre_closer_than_lambda1_cells"] = {
        k: sum(out["cells"][c]["rm_pre_closer_than_lambda1"][k] for c in cells)
        for k in ("accuracy", "tokens", "markers", "response_kappa")}
    print(json.dumps(out["summary"], indent=1))
    Path("experiments/removal_vs_untrained.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
