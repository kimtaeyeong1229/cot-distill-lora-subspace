#!/usr/bin/env python3
"""Score bench_generate.py outputs with MI-Distillation's extraction and grading functions.

Accuracy per run is the fraction of items whose k-th sample is correct; the
summary reports the mean and standard deviation over runs, as MI-Distillation
does over its repeated invocations. Grader timeouts count as wrong.
"""

import argparse
import gzip
import json
import multiprocessing as mp
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "MI-Distillation" / "evaluation"))
from evaluate import extract_prediction, normalize_prediction_for_answer, score_prediction  # noqa: E402

BENCH = None
TIMEOUT = 5.0


def _init(bench, timeout):
    global BENCH, TIMEOUT
    BENCH, TIMEOUT = bench, timeout


def score_item(record):
    correct, timeouts = [], 0
    for response in record["responses"]:
        pred = extract_prediction(record["question"], response, dataset=BENCH)
        pred = normalize_prediction_for_answer(pred, record["answer"], dataset=BENCH)
        ok, timed_out = score_prediction(BENCH, pred, record["answer"], grader_timeout=TIMEOUT)
        correct.append(bool(ok))
        timeouts += int(timed_out)
    return correct, timeouts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student_dir", required=True)
    parser.add_argument("--raw_dir", default="bench_raw")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--grader_timeout", type=float, default=5.0)
    args = parser.parse_args()

    student_dir = Path(args.student_dir)
    for raw in sorted((student_dir / args.raw_dir).glob("*/*.json.gz")):
        bench, cond = raw.parent.name, raw.name.replace(".json.gz", "")
        out = student_dir / f"eval_{bench}" / f"{cond}.json"
        if out.exists() and out.stat().st_mtime >= raw.stat().st_mtime:
            continue
        with gzip.open(raw, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
        meta, records = payload["meta"], payload["records"]
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("fork"),
                                 initializer=_init, initargs=(bench, args.grader_timeout)) as pool:
            scored = list(pool.map(score_item, records, chunksize=8))

        runs = meta["runs"]
        run_acc = [sum(c[k] for c, _ in scored) / len(scored) for k in range(runs)]
        all_tokens = [t for r in records for t in r["tokens"]]
        truncated = sum(f == "length" for r in records for f in r["finish"])
        summary = {**meta, "accuracy": statistics.mean(run_acc),
                   "accuracy_std": statistics.stdev(run_acc) if runs > 1 else 0.0, "accuracy_runs": run_acc,
                   "avg_tokens": sum(all_tokens) / len(all_tokens), "truncated_rate": truncated / len(all_tokens),
                   "grader_timeouts": sum(t for _, t in scored)}
        slim = [{"answer": r["answer"], "groups": r["groups"], "correct": c, "tokens": r["tokens"],
                 "truncated": [f == "length" for f in r["finish"]], "response_0_tail": r["responses"][0][-1200:]}
                for r, (c, _) in zip(records, scored)]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"summary": summary, "records": slim}, ensure_ascii=False))
        print(f"[score] {bench:<14} {cond:<16} acc={summary['accuracy']:.4f}±{summary['accuracy_std']:.4f} "
              f"tokens={summary['avg_tokens']:.0f} trunc={summary['truncated_rate']:.1%} timeouts={summary['grader_timeouts']}", flush=True)


if __name__ == "__main__":
    main()
