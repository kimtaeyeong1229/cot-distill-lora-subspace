#!/usr/bin/env python3
"""Per-problem correctness rate pi(x) of a teacher sampler, from k extra samples (filtering correction)."""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import clean_response, is_correct, read_jsonl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--samples", nargs="+", required=True)
ap.add_argument("--output", required=True)
a = ap.parse_args()
hits, n = defaultdict(int), defaultdict(int)
for f in a.samples:
    for r in read_jsonl(f):
        n[r["unique_id"]] += 1
        ok = r.get("finish_reason") == "stop" and is_correct(r["problem"], clean_response(r["cot_response"]), r["answer"])
        hits[r["unique_id"]] += int(ok)
out = {str(u): {"k": n[u], "correct": hits[u]} for u in n}
Path(a.output).write_text(json.dumps(out))
print(len(out), sum(v["correct"] for v in out.values()) / max(1, sum(v["k"] for v in out.values())))
