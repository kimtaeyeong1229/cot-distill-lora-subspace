#!/usr/bin/env python3
import gzip
import json
import math
import pandas as pd
from common import is_correct
from config import CELL_NAMES as CELLS, EVAL_PROBLEMS as PROBLEMS, EVAL_SAMPLES as SAMPLES, EXP as E
from config import LAMBDAS as LAMS, RESULTS as X, SEEDS, STUDENT_MODEL as MODELS


def per_unit():
    ret = json.loads((X / "lambda0_retention.json").read_text())
    lc = json.loads((X / "length_confound.json").read_text())
    out = {}
    for cell in CELLS:
        kept = ret[cell]["kept"]
        ivs = lc["pairs"][cell[0]]["intervals"]
        rows = []
        for i, iv in enumerate(ivs):
            a, b = (kept[LAMS[i]], kept[LAMS[i + 1]])
            d = iv["log_len_drop"]
            rows.append(
                {
                    "interval": f"{LAMS[i]}->{LAMS[i + 1]}",
                    "class": iv["class"],
                    "log_len_drop": d,
                    "kept_ratio": b / a,
                    "loss_per_unit": -math.log(b / a) / d if abs(d) >= 0.05 else None,
                }
            )
        out[cell] = rows
    return out


def untrained_stats(answers):
    out = {}
    for short, name in MODELS.items():
        n = c = done = done_c = 0
        for s in SEEDS:
            path = E / "base_eval" / name / f"eval_main_s{s}" / "base.samples.jsonl.gz"
            rows = [json.loads(line) for line in gzip.open(path, "rt")]
            assert len(rows) == PROBLEMS * SAMPLES, path
            for i, r in enumerate(rows):
                ok = is_correct(r["problem"], r["text"], answers[i // SAMPLES])
                fin = r["finish_reason"] == "stop"
                n += 1
                c += ok
                done += fin
                done_c += ok and fin
        out[short] = {"accuracy": c / n, "truncated": 1 - done / n, "accuracy_completed": done_c / done}
    return out


def main():
    rows = pd.read_parquet(E / "data" / "math500.parquet").to_dict("records")[:PROBLEMS]
    answers = [str(r["answer"]) for r in rows]
    me = json.loads((X / "main_edits.json").read_text())
    unt = untrained_stats(answers)
    l1 = {}
    for cell in CELLS:
        b1, u = (me[cell]["base1"], unt[cell.split("/")[1]])
        l1[cell] = {
            "lambda1": {k: b1[k] for k in ("accuracy", "truncated", "accuracy_completed")},
            "untrained": u,
            "d_accuracy_pp": 100 * (b1["accuracy"] - u["accuracy"]),
            "d_accuracy_completed_pp": 100 * (b1["accuracy_completed"] - u["accuracy_completed"]),
            "d_truncated_pp": 100 * (b1["truncated"] - u["truncated"]),
        }
    out = {"lambda0_region_per_unit": per_unit(), "untrained": unt, "lambda1_vs_untrained": l1}
    (X / "additional_stats.json").write_text(json.dumps(out, indent=1))
    for cell, v in l1.items():
        print(
            cell,
            " ".join(
                (f"{k} {v[k]:+.1f}" for k in ("d_accuracy_pp", "d_accuracy_completed_pp", "d_truncated_pp"))
            ),
        )
    for cell in CELLS[3:6]:
        print(
            cell,
            [
                (
                    r["interval"],
                    r["class"],
                    None if r["loss_per_unit"] is None else round(r["loss_per_unit"], 2),
                )
                for r in out["lambda0_region_per_unit"][cell]
            ],
        )


if __name__ == "__main__":
    main()
