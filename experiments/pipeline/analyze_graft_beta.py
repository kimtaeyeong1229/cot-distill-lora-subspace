#!/usr/bin/env python3
import gzip
import json
from config import CELL_DIR as CELLS, EXP as E, HEAD_WORDS as HEAD, REFLECTIVE, RESULTS


def summarise(path):
    records = [json.loads(l) for l in gzip.open(path, "rt")]
    markers = words = 0
    for r in records:
        head = r["text"].split()[:HEAD]
        markers += len(REFLECTIVE.findall(" ".join(head)))
        words += len(head)
    return {
        "head_markers_per_1k": markers / max(words, 1) * 1000,
        "completed": sum((r["finish_reason"] == "stop" for r in records)) / len(records),
    }


def main():
    out = {}
    for cell, tag in CELLS.items():
        base = json.loads((E / tag / "eval_ablation_loo" / "lambda0.0_base.json").read_text())
        row = {"lambda0_markers_per_1k": base["reflective_per_1k_words"]}
        for variant in ("graft", "graft_orth", "graft_full"):
            for beta in (2, 4, 8):
                name = f"{variant}{beta}"
                samples = E / tag / "eval_graft_beta" / f"{name}.samples.jsonl.gz"
                if not samples.exists():
                    continue
                summary = json.loads(samples.with_name(f"{name}.json").read_text())
                row[name] = {
                    **summarise(samples),
                    "accuracy": summary["accuracy"],
                    "truncated": summary["truncated_rate"],
                }
        out[cell] = row
        g, o = (row.get("graft2", {}), row.get("graft_orth2", {}))
        print(
            f"{cell:8} shared {g.get('head_markers_per_1k', float('nan')):6.2f} ({g.get('completed', float('nan')):.2f} done) | other {o.get('head_markers_per_1k', float('nan')):5.2f} ({o.get('completed', float('nan')):.2f} done)"
        )
    (RESULTS / "graft_beta.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
