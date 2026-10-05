#!/usr/bin/env python3
"""Section 5 (Tables 6-9) under the shared student setting, pooled over seeds 0-2.

Each seed's edited adapters (runs_edit_s<seed>, region built without that seed's edited adapters) were
evaluated on the first 200 MATH-500 problems x 2 samples (eval_main_s<seed>). As for the bases in
analyze_main_density.py, the reflective-marker density is counted on each problem's first sample and
tokens, truncation, completion and accuracy use both samples. The lambda0 / lambda1 bases come from
experiments/main_density.json, so an unedited adapter has one value everywhere in the paper.

Run after analyze_main_density.py; writes experiments/main_edits.json.
"""

import gzip
import json
from pathlib import Path

from config import CELL_DIR as CELLS, EVAL_SAMPLES as SAMPLES, EXP as E, HEAD_WORDS as HEAD, REFLECTIVE, RESULTS, SEEDS

VARIANTS = ["lambda0.0_rm_pre", "lambda0.0_rm_decisive", "lambda0.0_rm_scale", "lambda1.0_rm_pre",
            "graft1", "graft_orth1", "graft_full1", "graft2", "graft_orth2", "graft_full2",
            "ampn1", "ampn4", "ampV",
            # added for the review: early markers of the energy-matched top removal and the length-matched cut
            "lambda0.0_gs_energy", "lambda0.0_budget"]


def density(texts, head=None):
    words = [t.split()[:head] if head else t.split() for t in texts]
    markers = sum(len(REFLECTIVE.findall(" ".join(w))) for w in words)
    return markers / max(sum(map(len, words)), 1) * 1000


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--base", default=str(RESULTS / "main_density.json"))
    ap.add_argument("--output", default=str(RESULTS / "main_edits.json"))
    a = ap.parse_args()
    base = json.loads(Path(a.base).read_text())
    out = {}
    for cell, tag in CELLS.items():
        row = {"base0": base[cell].get("0.0"), "base1": base[cell].get("1.0")}
        for v in VARIANTS:
            recs, accs = [], []
            for seed in a.seeds:
                path = E / tag / f"eval_main_s{seed}" / f"{v}.samples.jsonl.gz"
                if not path.exists():
                    continue
                recs += [json.loads(l) for l in gzip.open(path, "rt")]
                accs.append(json.loads(path.with_name(f"{v}.json").read_text())["accuracy"])
            if not recs:
                continue
            first = [r["text"] for i, r in enumerate(recs) if i % SAMPLES == 0]
            row[v] = {"seeds": len(accs), "responses": len(first),
                      "markers_per_1k": density(first), "head_markers_per_1k": density(first, HEAD),
                      "avg_tokens": sum(r["tokens"] for r in recs) / len(recs),
                      "completed": sum(r["finish_reason"] == "stop" for r in recs) / len(recs),
                      "accuracy": sum(accs) / len(accs)}
        out[cell] = row
        done = [v for v in VARIANTS if v in row]
        print(f"{cell:8} {len(done)}/{len(VARIANTS)} conditions, seeds {sorted({row[v]['seeds'] for v in done}) or '-'}")
    Path(a.output).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
