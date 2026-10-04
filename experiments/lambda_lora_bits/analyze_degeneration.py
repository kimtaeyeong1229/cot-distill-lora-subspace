#!/usr/bin/env python3
"""How often do edited students end in a repetition loop?

Grafted students (Table 9) turn reflective markers on but rarely finish, so their markers could come
from text that repeats itself rather than from thinking. This script counts loops in the existing
generations, with no new generation.

Definition. A response loops if, in its last 300 words (whitespace tokens), at least half of the word
10-grams repeat a 10-gram that already occurred earlier in those 300 words. Responses shorter than 100
words never count as loops. A paragraph that restates a formula once or twice stays far below one half;
a response that cycles through the same sentence or two for its last 300 words is close to 1.

Responses. As for the marker density, each problem's first sampled response is used: the first 200
MATH-500 problems, seeds 0-2 (600 responses per cell and condition). Unedited lambda0 / lambda1 adapters
come from the stored first response (eval_math500[_7680], field response_0), edits from
eval_main_s<seed>. The beta = 4 and 8 grafts exist only for seed 0 (eval_graft_beta, 200 responses).

Writes experiments/degeneration.json.
"""

import gzip
import json
import statistics as st
from pathlib import Path

from analyze_main_density import CELLS
from config import EVAL_PROBLEMS as PROBLEMS, EVAL_SAMPLES as SAMPLES, EXP as E, RESULTS, SEEDS

TAIL, N, MIN_WORDS, CUT = 300, 10, 100, 0.5
EDITS = {"rm_pre": "lambda0.0_rm_pre", "graft1": "graft1", "graft2": "graft2", "graft_other2": "graft_orth2",
         "graft_whole2": "graft_full2", "ampn4": "ampn4"}
BETA = ["graft4", "graft8", "graft_orth4", "graft_orth8", "graft_full4", "graft_full8"]


def repeat_share(text):
    words = text.split()
    if len(words) < MIN_WORDS:
        return 0.0
    tail = words[-TAIL:]
    seen, rep, total = set(), 0, 0
    for i in range(len(tail) - N + 1):
        g = tuple(tail[i:i + N])
        rep += g in seen
        seen.add(g)
        total += 1
    return rep / max(total, 1)


def rate(texts):
    shares = [repeat_share(t) for t in texts]
    return {"loop_rate": sum(s >= CUT for s in shares) / len(shares), "mean_repeat_share": st.mean(shares),
            "responses": len(shares)}


def base_texts(tag, bench, lam, seed):
    recs = json.loads((E / tag / bench / f"lambda{lam}_seed{seed}.json").read_text())["records"][:PROBLEMS]
    return [r["response_0"] for r in recs]


def edit_texts(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(l)["text"] for i, l in enumerate(f) if i % SAMPLES == 0]


def summarise(per_seed):
    vals = [p["loop_rate"] for p in per_seed]
    return {"mean": st.mean(vals), "sd": st.stdev(vals) if len(vals) > 1 else 0.0, "seeds": vals,
            "mean_repeat_share": st.mean(p["mean_repeat_share"] for p in per_seed)}


def main():
    out = {}
    for cell, (tag, bench) in CELLS.items():
        row = {"base0": summarise([rate(base_texts(tag, bench, "0.0", s)) for s in SEEDS]),
               "base1": summarise([rate(base_texts(tag, bench, "1.0", s)) for s in SEEDS])}
        for name, file in EDITS.items():
            runs = [E / tag / f"eval_main_s{s}" / f"{file}.samples.jsonl.gz" for s in SEEDS]
            runs = [p for p in runs if p.exists()]
            if runs:
                row[name] = summarise([rate(edit_texts(p)) for p in runs])
        for name in BETA:
            p = E / tag / "eval_graft_beta" / f"{name}.samples.jsonl.gz"
            if p.exists():
                row[name.replace("orth", "other").replace("full", "whole") + "_seed0"] = summarise([rate(edit_texts(p))])
        out[cell] = row
        print(f"{cell:8s} " + " ".join(f"{k} {v['mean']:.2f}" for k, v in row.items()), flush=True)
    (RESULTS / "degeneration.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
